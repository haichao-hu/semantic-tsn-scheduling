from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback

from src.nc_engine.safety_validator import validate_schedule
from src.intent_ontology.types import FlowSemantics

from .tsn_env import TSNEnv, CSRLConfig
from .safety_shield import SafetyShield


# ============================================================
# Constraint Manager (Lagrangian dual)
# ============================================================


@dataclass
class ConstraintManager:
    """Manages NC-derived constraints as Lagrangian multipliers.

    Lagrangian: L(θ, λ) = J(θ) - λ · (c(s,a) - threshold)
    Dual ascent: λ ← max(0, λ + η · constraint_violation)
    """

    threshold: float = 0.0         # constraint threshold (zero violations)
    initial_lambda: float = 0.1    # initial multiplier value
    lr_lambda: float = 0.002       # dual ascent learning rate η
    max_lambda: float = 1.0        # cap: penalty must stay below the
                                   # completion reward scale, otherwise the
                                   # policy learns "reject everything"
    _lambda: float = field(init=False)

    def __post_init__(self) -> None:
        self._lambda = self.initial_lambda

    @property
    def value(self) -> float:
        return self._lambda

    def update(self, constraint_violation: float) -> float:
        """Update λ proportional to constraint violation (dual ascent).

        Returns the new λ value.
        """
        self._lambda = max(0.0, self._lambda + self.lr_lambda * constraint_violation)
        self._lambda = min(self._lambda, self.max_lambda)
        return self._lambda

    def reset(self) -> None:
        self._lambda = self.initial_lambda

    @property
    def is_constraint_active(self) -> bool:
        return self._lambda > 0

    def total_loss(self, reward_loss: float, constraint_violation: float) -> float:
        """Compute the Lagrangian loss: L = -reward_loss + λ · violation."""
        return -reward_loss + self._lambda * max(0.0, constraint_violation)


# ============================================================
# CSRL Agent
# ============================================================


class CSRLAgent:
    """Constrained PPO agent for semantic-aware TSN scheduling.

    Wraps stable-baselines3 PPO with:
    - Lagrangian constraint management from NC violations
    - Optional safety shield for action filtering
    - Periodic NC validation
    """

    def __init__(
        self,
        env: TSNEnv,
        constraint_manager: ConstraintManager | None = None,
        safety_shield: SafetyShield | None = None,
        ppo_kwargs: dict | None = None,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.env = env
        self.constraint_manager = constraint_manager or ConstraintManager()
        self.safety_shield = safety_shield
        self.device = device
        self.seed = seed

        # full reproducibility: torch, numpy, and SB3 share the seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        ppo_defaults = dict(
            policy="MlpPolicy",
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            seed=seed,
            verbose=0,
            device=device,
        )
        ppo_defaults.update(ppo_kwargs or {})
        self.model = PPO(env=env, **ppo_defaults)

        self._training_stats: dict[str, list[float]] = {
            "rewards": [],
            "completion_rates": [],
            "wcd_violations": [],
            "lambda_evolution": [],
            "constraint_violations": [],
        }

        self._safe_step_fn = None
        if self.safety_shield is not None and self.safety_shield.enabled:
            self._safe_step_fn = self._make_safe_step()

    def _make_safe_step(self):
        """Create a wrapped step function that applies the safety shield."""
        from src.intent_ontology.types import DecayType, SemanticCompressibility, UrgencyFunction
        from src.nc_engine.schedulability import SchedEntry

        original_step = self.env.step

        def safe_step(action: np.ndarray) -> tuple:
            if self.safety_shield is None:
                return original_step(action)

            decoded = self.env._decode_action(action)

            flows: list[FlowSemantics] = []
            entries: dict[str, SchedEntry] = {}
            for sf in self.env.sim_flows:
                flows.append(FlowSemantics(
                    flow_id=sf.flow_id,
                    task_id=sf.task_id,
                    priority_weight=sf.priority_weight,
                    delayable_boundary_us=int(sf.deadline_us),
                    urgency=UrgencyFunction(DecayType.STEP, value_plateau_us=0, decay_start_us=0),
                    compressibility=SemanticCompressibility(ratio=0.0),
                    stream_class=sf.stream_class,
                ))
                entries[sf.flow_id] = SchedEntry(
                    flow_id=sf.flow_id,
                    queue=int(sf.queue),
                    gate_start_us=float(sf.gate_start_us),
                    gate_size_us=float(sf.gate_size_us),
                    period_us=float(sf.period_us),
                    deadline_us=float(sf.deadline_us),
                    path=list(sf.path),
                    task_id=sf.task_id,
                    priority_weight=sf.priority_weight,
                    dispatch_phase_us=float(sf.phase_offset_us),
                )

            pw_map = {sf.flow_id: sf.priority_weight for sf in self.env.sim_flows}
            safe_decoded = self.safety_shield.filter_action(decoded, flows, None, pw_map, entries)

            # re-encode to action space
            safe_action = self._encode_action(safe_decoded)
            return original_step(safe_action)

        return safe_step

    def _encode_action(self, decoded: dict[str, dict]) -> np.ndarray:
        """Encode a decoded action dict back to the action space vector.

        Matches the env's 3-dim-per-flow parameterization
        (accept, dispatch_offset, gate_start_offset).
        """
        M = self.env.config.max_active_flows
        dim = self.env._action_dim_per_flow
        action = np.zeros(M * dim, dtype=np.float64)
        for i, sf in enumerate(self.env.sim_flows):
            if i >= M or sf.flow_id not in decoded:
                continue
            d = decoded[sf.flow_id]
            offset = i * dim
            action[offset] = 1.0 if d.get("accept", True) else -1.0
            action[offset + 1] = d.get("dispatch_offset_us", 0.0) / max(sf.period_us, 1.0) * 2.0 - 1.0
            # inverse of the env's relative gate_start parameterization
            rel = (d.get("gate_start_us", 0.0) - d.get("dispatch_offset_us", 0.0)) % max(sf.period_us, 1.0)
            action[offset + 2] = rel / max(sf.period_us, 1.0) * 2.0 - 1.0
        return np.clip(action, -1.0, 1.0)

    def train(
        self,
        total_timesteps: int = 10000,
        nc_validation_interval: int = 1000,
        log_interval: int = 100,
        callback: BaseCallback | None = None,
        use_shield_during_training: bool = False,
        warmup_ratio: float = 0.3,
    ) -> dict:
        """Train the PPO agent with periodic NC validation and constraint updates.

        By default the safety shield does NOT hard-filter actions during
        training: hard interception would corrupt the policy gradient
        (the policy would learn "accept → executed as reject").  Instead
        the NC constraint is enforced softly through the Lagrangian
        penalty (λ · violations subtracted from the reward).  The shield
        is a deployment-time component that vetoes unsafe actions before
        they reach the switch.

        The dual ascent is warmed up for ``warmup_ratio`` of the training
        budget: early training produces many deadline violations (the
        policy has not aligned windows yet), and letting λ react to them
        immediately would inflate the penalty faster than the policy can
        learn — a self-reinforcing conservative spiral.

        Parameters
        ----------
        total_timesteps : int
            Number of environment steps to train.
        nc_validation_interval : int
            Run NC validation every N steps.
        log_interval : int
            Log training statistics every N steps.
        callback : BaseCallback or None
            Optional SB3 callback.
        use_shield_during_training : bool
            Set True to ALSO hard-filter actions during training
            (not recommended; kept for ablation of the shield-in-training
            variant).
        warmup_ratio : float
            Fraction of the budget during which λ stays at its initial
            value while the policy explores.

        Returns
        -------
        stats : dict with training history.
        """
        remaining = total_timesteps
        step = 0
        warmup_steps = int(total_timesteps * warmup_ratio)

        if self._safe_step_fn is not None and use_shield_during_training:
            self.env.step = self._safe_step_fn

        # push the initial λ into the environment so the reward carries
        # the Lagrangian penalty from the first training step onward
        self.env.config.constraint_penalty = self.constraint_manager.value

        while remaining > 0:
            chunk = min(nc_validation_interval, remaining)
            self.model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                callback=callback,
            )
            step += chunk
            remaining -= chunk

            # dual ascent update (after warmup)
            if step >= warmup_steps and (step % nc_validation_interval == 0 or remaining == 0):
                violation = self._validate_current_schedule()
                self.constraint_manager.update(violation)
                # dual ascent: push the updated λ into the reward signal
                self.env.config.constraint_penalty = self.constraint_manager.value
                self._training_stats["wcd_violations"].append(violation)
                self._training_stats["lambda_evolution"].append(self.constraint_manager.value)
                self._training_stats["constraint_violations"].append(violation)

            # log reward
            if step % log_interval == 0:
                self._log_stats(step)

        return dict(self._training_stats)

    def _validate_current_schedule(self) -> float:
        """Compute the constraint violation signal for the dual ascent.

        Uses the actual deadline violations observed in the most recent
        environment step: a policy that fails to align windows produces
        missed deadlines, raising λ and thereby increasing the reward
        penalty — steering the policy back toward schedulable behavior.
        """
        return float(getattr(self.env, "_last_violation_count", 0))

    def _log_stats(self, step: int) -> None:
        """Accumulate training metrics."""
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            rewards = [ep.get("r", 0) for ep in self.model.ep_info_buffer]
            if rewards:
                self._training_stats["rewards"].append(float(np.mean(rewards)))

        n_accepted = sum(1 for f in self.env.sim_flows if f.accepted)
        n_with_data = sum(1 for f in self.env.sim_flows if len(f.e2e_delays) > 0)
        cr = n_with_data / max(n_accepted, 1)
        self._training_stats["completion_rates"].append(float(cr))

    def save(self, path: str) -> None:
        """Save model and constraints to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save(path)
        meta_path = path + ".meta.npz"
        np.savez(
            meta_path,
            lambda_=np.array([self.constraint_manager.value]),
            n_wcd_violations=np.array(self._training_stats.get("wcd_violations", [])),
            rewards=np.array(self._training_stats.get("rewards", [])),
        )

    def load(self, path: str) -> None:
        """Load model and constraints from disk."""
        self.model = PPO.load(path, env=self.env, device=self.device)
        meta_path = path + ".meta.npz"
        if os.path.exists(meta_path):
            meta = np.load(meta_path)
            self.constraint_manager._lambda = float(meta["lambda_"][0])

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Optional[np.ndarray]]:
        return self.model.predict(observation, deterministic=deterministic)
