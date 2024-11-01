"""L2SBench-substitute: evaluation harness for learned/heuristic microservice schedulers.

NOTE: "L2SBench" could not be located publicly; this harness plays its role. It runs policies on
TrainTicketSched-v0 and reports the paper's Table-2 metrics:
  Lat (ms) | Thr (% served; req/s also stored) | RU (%) | SR (%) | SLA (%) | CPU (%) | Mem (%) | Cost
and the two robustness studies:
  * recovery   (Fig.3): scripted fault window, per-step latency/throughput trace
  * sensitivity(Fig.4): throughput vs perturbation strength eps

Baselines are generic re-implementations (static, K8s-HPA-style threshold, random), NOT the cited
systems [8]-[14] of the paper, which are not reproducible from the paper alone.
"""
import argparse
import json
import os
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import simulator  # noqa: E402,F401
from simulator import AdversarialDisturbance  # noqa: E402
from simulator.env import sample_disturbance  # noqa: E402


# ---------------- policies ----------------
class StaticPolicy:
    name = "static"

    def __call__(self, env, obs):
        return np.ones(env.action_space.shape, dtype=int)


class RandomPolicy:
    name = "random"

    def __call__(self, env, obs):
        return env.action_space.sample()


class HPAPolicy:
    """Kubernetes-HPA-like rule: scale up if util>hi, down if util<lo; neutral routing/quota."""
    name = "hpa"

    def __init__(self, hi=0.7, lo=0.3):
        self.hi, self.lo = hi, lo

    def __call__(self, env, obs):
        u = env.unwrapped
        a = np.ones(env.action_space.shape, dtype=int)
        scale = np.where(u.util_s > self.hi, 2, np.where(u.util_s < self.lo, 0, 1))
        a[u.M:u.M + u.S] = scale
        return a


class PPOPolicy:
    name = "ppo"

    def __init__(self, ckpt, deterministic=True):
        from training.ppo_robust import Agent
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.agent = Agent(d["obs_dim"], d["nvec"])
        self.agent.load_state_dict(d["state_dict"])
        self.agent.eval()
        self.det = deterministic

    def __call__(self, env, obs):
        with torch.no_grad():
            a, *_ = self.agent.get_action_and_value(torch.as_tensor(obs[None], dtype=torch.float32),
                                                    deterministic=self.det)
        return a[0].numpy()


# ---------------- runner ----------------
def run_episode(policy, env_kwargs, eps=0.0, seed=0, adversarial=False, trace=False):
    env = gym.make("TrainTicketSched-v0", **env_kwargs)
    if adversarial and eps > 0:
        env = AdversarialDisturbance(env, eps=eps)
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    rng = np.random.default_rng(seed + 12345)
    rows, ret = [], 0.0
    done = False
    while not done:
        if not adversarial and eps > 0:            # random (non-adversarial) disturbance from Omega(eps)
            u.set_disturbance(sample_disturbance(rng, eps, u.M))
        a = policy(env, obs)
        obs, r, term, trunc, info = env.step(a)
        ret += r
        rows.append(info)
        done = term or trunc
    st = u.stats
    m = {
        "Lat": float(np.mean([x["lat_ms"] for x in rows])),
        "Thr": 100 * st["completed"] / max(st["arrived"], 1),   # % of arrived requests served
        "Thr_rps": float(np.mean([x["thr_rps"] for x in rows])),
        "RU": 100 * float(np.mean([x["ru"] for x in rows])),
        "SR": 100 * (1 - (st["dropped"] + st["rejected"]) / max(st["arrived"] + st["sched_ops"], 1)),
        "SLA": 100 * float(np.mean([x["sla_ok"] for x in rows])),
        "CPU": 100 * float(np.mean([x["cpu_alloc"] for x in rows])),
        "Mem": 100 * float(np.mean([x["mem_alloc"] for x in rows])),
        "Cost": float(np.sum([x["cost"] for x in rows])),
        "Return": float(ret),
    }
    if trace:
        m["trace"] = [{"t": i, "lat": x["lat_ms"], "thr": x["thr_rps"]} for i, x in enumerate(rows)]
    return m


def agg(ms):
    keys = [k for k in ms[0] if k != "trace"]
    return {k: float(np.mean([m[k] for m in ms])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="PPO checkpoint from training/ppo_robust.py --save-path")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--eps-grid", default="0,0.2,0.4,0.6")
    ap.add_argument("--out", default="outputs/results/bench.json")
    a = ap.parse_args()

    policies = [StaticPolicy(), RandomPolicy(), HPAPolicy()]
    if a.ckpt:
        policies.append(PPOPolicy(a.ckpt))
    kw = {"max_steps": a.max_steps}
    res = {"table2": {}, "recovery": {}, "sensitivity": {}}

    hdr = f"{'Method':8s}" + "".join(f"{k:>9s}" for k in ["Lat", "Thr", "RU", "SR", "SLA", "CPU", "Mem", "Cost"])
    print("== Table 2 (nominal) ==\n" + hdr)
    for p in policies:
        m = agg([run_episode(p, kw, seed=s) for s in range(a.episodes)])
        res["table2"][p.name] = m
        print(f"{p.name:8s}" + "".join(f"{m[k]:9.2f}" for k in ["Lat", "Thr", "RU", "SR", "SLA", "CPU", "Mem", "Cost"]))

    # Fig.3: fault window in the middle third: surge x2 + one node degraded to 40%
    t0, t1 = a.max_steps // 3, 2 * a.max_steps // 3
    deg = np.ones(6); deg[0] = 0.4
    fault = [{"t0": t0, "t1": t1, "xi": {"surge": 2.0, "degrade": deg, "jitter": 0.1}}]
    print(f"== Recovery (fault in [{t0},{t1})) ==")
    for p in policies:
        m = run_episode(p, {**kw, "fault_schedule": fault}, seed=0, trace=True)
        lat = np.array([x["lat"] for x in m["trace"]])
        pre = lat[:t0].mean()
        post = lat[t1:]
        rec = next((i for i, v in enumerate(post) if v <= 1.1 * pre), None)
        res["recovery"][p.name] = {"trace": m["trace"], "pre_lat": float(pre), "peak_lat": float(lat[t0:t1].max()),
                                   "steps_to_recover": rec}
        print(f"{p.name:8s} pre_lat={pre:9.1f} peak={lat[t0:t1].max():9.1f} steps_to_recover={rec}")

    # Fig.4: throughput vs perturbation strength
    grid = [float(x) for x in a.eps_grid.split(",")]
    print("== Sensitivity: Thr vs eps ==\n" + f"{'Method':8s}" + "".join(f"{e:>9.2f}" for e in grid))
    for p in policies:
        row = [agg([run_episode(p, kw, eps=e, seed=s) for s in range(a.episodes)])["Thr"] for e in grid]
        res["sensitivity"][p.name] = dict(zip(map(str, grid), row))
        print(f"{p.name:8s}" + "".join(f"{v:9.2f}" for v in row))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
