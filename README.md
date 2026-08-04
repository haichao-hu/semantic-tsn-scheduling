# Semantic-Aware TSN Scheduling (CSRL)

A semantic-aware scheduler for Time-Sensitive Networking (TSN): task-intent ontology + Constrained Safe Reinforcement Learning (CSRL) + Network Calculus safety shield.

> **Paper**: `paper/main.pdf` (EN) / `paper/main-zh.pdf` (ZH)
>
> **CI**: [![test](https://github.com/haichao-hu/semantic-tsn-scheduling/actions/workflows/test.yml/badge.svg)](https://github.com/haichao-hu/semantic-tsn-scheduling/actions/workflows/test.yml)

## What is this?

TSN gives industrial networks deterministic timing through gate-controlled schedules, but today's schedulers only optimize network metrics — latency, jitter, throughput — without knowing what the data means to the application. A sensor stream that tolerates 10 ms during inspection becomes safety-critical within an emergency stop; the network cannot see the difference.

This framework bridges that gap with three components:

1. **Intent ontology** — six industrial task types mapped to flow semantics (priority, urgency, deadline) and then to GCL/CBS parameters.
2. **CSRL scheduler** — a constrained PPO optimizing semantic-weighted task completion, with a Network Calculus worst-case delay bound coupled into the learning objective through a Lagrangian multiplier.
3. **NC safety shield** — every scheduling action is validated against deterministic schedulability constraints (window fits frame, per-port window mutual exclusion, aligned deadline satisfiability) before it reaches the switch. The check is policy-independent: an unsafe action never executes no matter what the learned policy outputs.

## Key results (3-switch line topology, 10 ms hyperperiod)

| Scenario | CSRL | Static GCL | Note |
|---|---|---|---|
| 5 / 8 flows | 100% | 100% | parity (classical schedulers are near-optimal at small scale) |
| 12 flows | 97.5% (multi-seed 92.8% ± 3.6%) | 100% | close to static; λ rises 0.12 → 0.67 |
| Scarce: 10 same-period flows, 1 port | 100% ± 0 | **41.7% ± 0** | learning-based coordination beats greedy fragmentation |

The Lagrangian multiplier λ rises monotonically with flow density (0.12 at 5 flows → 0.67 at 12 flows), demonstrating that the formal constraint pressure is genuinely transmitted into the learned policy.

## Repository layout

```
src/
  intent_ontology/   # task-intent ontology + intent → GCL/CBS mapping
  csrl/              # TSNEnv simulator, PPO agent, Lagrangian, safety shield
  nc_engine/         # network-calculus engine + online schedulability check
  experiments/       # experiment framework (runner / baselines / ablation)
experiments/
  run_full.py        # 4 experiment modes + scarce-resource mode
  run_matrix.py      # multi-seed matrix
  run_deploy_gap.py  # deployment-gap experiments (shield veto vs training)
  make_fig2.py       # regenerates paper Figure 2 from curated data
  results/           # curated v2 results (paper data)
paper/               # LaTeX sources (EN + ZH), compiled PDFs
tests/               # 250 tests
.github/workflows/   # CI (pytest on push/PR, CPU torch)
```

## Install & run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# on machines without a GPU, install CPU-only torch first to avoid the
# multi-GB CUDA wheel:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu

# tests (fast subset; full suite incl. training tests takes ~10 min)
pytest tests/ -q -m "not slow"
pytest tests/ -q

# experiments
python experiments/run_full.py --mode default --flows 5 --train 60000
python experiments/run_full.py --mode large   --flows 12 --train 60000
python experiments/run_full.py --mode dynamic --flows 8  --train 60000
python experiments/run_full.py --mode ablation --flows 8 --train 60000
python experiments/run_full.py --mode scarcity           --train 120000

# multi-seed matrix
python experiments/run_matrix.py --modes default large --seeds 123,456,789

# regenerate paper Figure 2 from curated data
python experiments/make_fig2.py
```

Experiment outputs are written to `experiments/run_*/` (gitignored); curated results for the paper live in `experiments/results/`.

## Reproducibility notes

- All training is seeded (torch, numpy, PPO); seeds 42 / 123 / 456 / 789 used in the paper.
- Completion is measured as completed packets over expected packets (`T_hp / T_i` per accepted step); rejected or unscheduled flows count as incomplete.
- The OMNeT++/INET simulation (`simulation/`) is a separate dependency — see `simulation/README.md`; it is not required for the Python pipeline.

## Roadmap & status

Internal tracking documents (`STATUS.md`, `ROADMAP.md`, research application
reports) are maintained outside this repository.

## License

Apache 2.0 — see [LICENSE](LICENSE).
