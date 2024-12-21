#!/usr/bin/env bash
# Smoke test only: tiny calibration + 2-episode PPO + 1-episode benchmark. No full training.
# Pipeline: data/ -> calibration/ -> simulator/ -> training/ -> evaluation/ -> outputs/
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
SCEN="data/zenodo/ex/anomalies_microservice_trainticket_version_configurations/ts-order-service_mongodb_4.2.2_2022-07-12"
echo "[1/4] calibration: train-ticket source + Zenodo 6979726 -> data/trainticket_profile.json"
$PY -I calibration/calibrate.py --scenario-dir "$SCEN" --out data/trainticket_profile.json
echo "[2/4] simulator check"
$PY -c "import gymnasium as gym, numpy as np, simulator; e=gym.make('TrainTicketSched-v0'); o,_=e.reset(seed=0); o,r,*_=e.step(e.action_space.sample()); assert np.isfinite(o).all() and np.isfinite(r); print('env ok', o.shape, e.action_space.nvec.size)"
echo "[3/4] training: robust PPO, 2 episodes"
$PY -m training.ppo_robust --episodes 2 --perturb-eps 0.3 --no-cuda --save-path outputs/checkpoints/smoke_agent.pt 2>&1 | grep -v -i warn
echo "[4/4] evaluation: benchmark harness, 1 episode x 30 steps"
$PY -m evaluation.bench --ckpt outputs/checkpoints/smoke_agent.pt --episodes 1 --max-steps 30 --eps-grid 0,0.3,0.6 --out outputs/results/smoke_bench.json 2>&1 | grep -v -i warn
echo "SMOKE TEST PASSED"
