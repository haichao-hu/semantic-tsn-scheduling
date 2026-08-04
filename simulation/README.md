# Simulation (OMNeT++ / INET)

The Python pipeline in `src/` does not require OMNeT++. This directory is
reserved for cross-validation against the INET framework's TSN models
(frame preemption, guard bands, packet-level timing).

## Current status

- `inet/` — the INET framework source tree (compiled `libINET.so` present
  locally). **Not tracked in git**; clone separately:
  `git clone --depth 1 --branch v6.4.0 https://github.com/inet-framework/inet.git simulation/inet`
- No custom simulation cases yet. A minimal 3-switch line-topology TAS case
  (validating the scarce-scenario phase-staggering result) is planned —
  see `ROADMAP.md` (P2).

## Requirements

- OMNeT++ 6.x (https://omnetpp.org)
- INET 6.4+ (clone into `simulation/inet/`)
