# rlsched — re-implementation of "Learning Collaborative and Robust Scheduling Policies for Microservice Backends under Uncertainty"

Status: **smoke test only** (`./smoke_test.sh`). No full training or evaluation has been run.

## Repository layout

The folders follow the paper's pipeline, in order:

```
rlsched/
├── data/                         inputs (read-only)
│   ├── train-ticket/             FudanSELab/train-ticket, pruned to ts-*-service/**/*.java → static call graph
│   ├── zenodo/                   Zenodo 6979726 traces + Prometheus metrics (tt.zip and extracted ex/)
│   └── trainticket_profile.json  calibrated service profile (written by calibration/)
│
├── calibration/
│   └── calibrate.py              data/ → trainticket_profile.json (latency, call ratios, request mix, CPU)
│
├── simulator/                    stand-in for KuTARA (not publicly available)
│   ├── env.py                    Gymnasium env TrainTicketSched-v0: state Eq.1, action Eq.2, reward Eq.3,
│   │                             disturbance set Ω(ε) Eq.5, constraints Eq.7
│   └── adversary.py              AdversarialDisturbance wrapper: worst-case ξ, Eq.6
│
├── training/
│   └── ppo_robust.py             robust + Lagrangian-constrained PPO (adapted from CleanRL ppo.py), Table 1
│
├── evaluation/                   stand-in for L2SBench (not publicly available)
│   └── bench.py                  Table 2 metrics, Fig.3 recovery, Fig.4 ε-sensitivity; baselines
│
├── outputs/                      generated artifacts
│   ├── checkpoints/              trained policies (*.pt)
│   ├── runs/                     TensorBoard logs
│   └── results/                  benchmark JSON
│
├── docs/数学基础.md               math background, with links into the code
└── smoke_test.sh                 runs the whole pipeline end to end at tiny scale
```

## How the paper maps to the code
| Paper | Code |
|---|---|
| Eq.1 state `[q; u; d; l]` | `MicroserviceSchedEnv._obs` in `simulator/env.py`. 30 services and 6 nodes give a 133-dimensional observation. |
| Eq.2 joint action | `MultiDiscrete`: per-node routing weight (6) + per-service scale ±1 (30) + per-service CPU quota ±1 (30). |
| Eq.3 reward | `-a·L/SLA - b·imbalance - c·(scaling ops + allocated share) + d·served ratio` |
| Eq.5 disturbance ξ | traffic surge, single-node degradation, CPU jitter; Ω(ε) is `sample_disturbance` in `simulator/env.py` |
| Eq.6 worst case | `simulator/adversary.py`: at each step, the worst of K sampled ξ on a cloned simulator (greedy min over ξ) |
| Eq.7 constraints | `info["constraints"]` (capacity, queue bound, SLA); Lagrangian + dual ascent in `training/ppo_robust.py` |
| Table 1 | hidden (128, 64) for actor and critic, Adam lr 1e-3, γ 0.99, minibatch 64, 200 episodes × 100 steps (defaults in `training/ppo_robust.py`) |
| Table 2, Fig.3, Fig.4 | `evaluation/bench.py` |

## Running
Run every command from the repository root. The Python scripts are run as modules (`-m`) so they can import `simulator`.

```bash
./smoke_test.sh                                                                                    # tiny end-to-end check

# Full runs (not executed)
.venv/bin/python -m training.ppo_robust --perturb-eps 0.3 --save-path outputs/checkpoints/robust.pt                      # Table 1 defaults
.venv/bin/python -m training.ppo_robust --perturb-eps 0 --lagrangian False --save-path outputs/checkpoints/nominal.pt  # ablation
for lr in 3e-4 1e-3 3e-3; do .venv/bin/python -m training.ppo_robust --learning-rate $lr --save-path outputs/checkpoints/lr$lr.pt; done  # Fig.2
.venv/bin/python -m evaluation.bench --ckpt outputs/checkpoints/robust.pt --episodes 10                                   # Table 2 / Fig.3 / Fig.4
```

## Caveats
- `simulator/` and `evaluation/` are stand-ins for KuTARA and L2SBench written here, not the original tools.
- Baselines are static, HPA-threshold and random. The cited systems [8]–[14] are not re-implemented.
- The simulator is a fluid queue model calibrated on a single Zenodo scenario. It is not a live Kubernetes deployment of Train-Ticket.
