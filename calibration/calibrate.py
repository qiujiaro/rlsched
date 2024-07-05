"""Build a Train-Ticket service profile used to calibrate the scheduling simulator.

Sources
  * FudanSELab/train-ticket source tree  -> static service dependency graph
    (every `http://ts-xxx-service` literal inside ts-yyy-service/**.java is an edge yyy -> xxx)
  * Zenodo record 6979726 ("Anomalies in Microservice Architecture (train-ticket)")
      - Jaeger traces   -> per-service span latency, dynamic call edges + call ratios, request mix
      - Prometheus cAdvisor CPU -> per-service CPU demand (cores)

Output: data/trainticket_profile.json
"""
import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

URL_RE = re.compile(r"http://(ts-[a-z0-9-]+-service)")


def static_graph(tt_root):
    services = sorted(d for d in os.listdir(tt_root)
                      if d.startswith("ts-") and d.endswith("-service") and os.path.isdir(os.path.join(tt_root, d)))
    edges = Counter()
    for s in services:
        for f in glob.glob(os.path.join(tt_root, s, "**", "*.java"), recursive=True):
            with open(f, errors="ignore") as fh:
                for callee in URL_RE.findall(fh.read()):
                    if callee != s:
                        edges[(s, callee)] += 1
    return services, edges


def parse_traces(trace_dir, max_files=None):
    span_dur = defaultdict(list)       # service -> [ms]
    dyn_edges = Counter()              # (caller, callee) -> count
    entry = Counter()                  # root-span service -> count
    files = sorted(glob.glob(os.path.join(trace_dir, "*.json")))[:max_files]
    n_traces = 0
    for f in files:
        try:
            with open(f) as fh:
                doc = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        for tr in doc.get("data") or []:
            n_traces += 1
            procs = {k: v.get("serviceName", "?") for k, v in (tr.get("processes") or {}).items()}
            spans = {sp["spanID"]: sp for sp in tr.get("spans", [])}
            for sp in spans.values():
                svc = procs.get(sp.get("processID"), "?")
                span_dur[svc].append(sp.get("duration", 0) / 1000.0)
                parents = [r["spanID"] for r in sp.get("references", []) if r.get("refType") == "CHILD_OF"]
                if not parents or parents[0] not in spans:
                    entry[svc] += 1
                    continue
                psvc = procs.get(spans[parents[0]].get("processID"), "?")
                if psvc != svc:
                    dyn_edges[(psvc, svc)] += 1
    return span_dur, dyn_edges, entry, n_traces


def parse_cpu(metric_file):
    """Mean CPU cores per ts-* service from container_cpu_usage_seconds_total (counter -> rate)."""
    out = defaultdict(list)
    if not metric_file:
        return {}
    with open(metric_file) as fh:
        doc = json.load(fh)
    for series in doc.get("data", {}).get("result", []):
        m = series["metric"]
        pod, cont = m.get("pod", ""), m.get("container", "")
        if not pod.startswith("ts-") or cont in ("POD", ""):
            continue
        svc = re.match(r"(ts-[a-z0-9-]+?-service)", pod)
        if not svc:
            continue
        v = np.array([[float(t), float(x)] for t, x in series["values"]])
        if len(v) < 2:
            continue
        rate = np.diff(v[:, 1]) / np.maximum(np.diff(v[:, 0]), 1e-6)
        rate = rate[rate >= 0]
        if len(rate):
            out[svc.group(1)].append(float(rate.mean()))
    return {k: float(np.sum(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tt-root", default="data/train-ticket")
    ap.add_argument("--scenario-dir", required=True, help="one extracted Zenodo scenario folder")
    ap.add_argument("--out", default="data/trainticket_profile.json")
    ap.add_argument("--max-trace-files", type=int, default=None)
    a = ap.parse_args()

    services, s_edges = static_graph(a.tt_root)
    tdir = glob.glob(os.path.join(a.scenario_dir, "Traces_*"))[0]
    cpu_file = (glob.glob(os.path.join(a.scenario_dir, "Monitoring_*", "*container_cpu_usage_seconds_total.json")) or [None])[0]
    span_dur, d_edges, entry, n_traces = parse_traces(tdir, a.max_trace_files)
    cpu = parse_cpu(cpu_file)

    idx = {s: i for i, s in enumerate(services)}
    prof = {"services": services, "source": {"tt_root": a.tt_root, "scenario": os.path.basename(a.scenario_dir.rstrip("/")),
                                             "n_traces": n_traces}}
    lat = []
    for s in services:
        d = np.array(span_dur.get(s, []))
        lat.append({"n": int(d.size),
                    "p50_ms": float(np.percentile(d, 50)) if d.size else None,
                    "p95_ms": float(np.percentile(d, 95)) if d.size else None,
                    "mean_ms": float(d.mean()) if d.size else None})
    prof["latency"] = lat
    prof["cpu_cores"] = [cpu.get(s) for s in services]
    # call ratio: how many downstream calls to callee per request served by caller
    served = {s: max(len(span_dur.get(s, [])), 1) for s in services}
    edges = {}
    for (u, v), c in s_edges.items():
        edges[(u, v)] = {"static": c, "dyn": 0, "ratio": 0.0}
    for (u, v), c in d_edges.items():
        if u in idx and v in idx:
            e = edges.setdefault((u, v), {"static": 0, "dyn": 0, "ratio": 0.0})
            e["dyn"] = c
            e["ratio"] = c / served[u]
    prof["edges"] = [[idx[u], idx[v], e["static"], e["dyn"], e["ratio"]] for (u, v), e in sorted(edges.items())]
    tot = sum(c for s, c in entry.items() if s in idx) or 1
    prof["entry_mix"] = [entry.get(s, 0) / tot for s in services]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(prof, fh, indent=1)
    n_dyn = sum(1 for e in edges.values() if e["dyn"])
    n_lat = sum(1 for x in lat if x["n"])
    print(f"services={len(services)} static_edges={len(s_edges)} dyn_edges={n_dyn} traces={n_traces} "
          f"services_with_latency={n_lat} services_with_cpu={sum(v is not None for v in prof['cpu_cores'])} -> {a.out}")


if __name__ == "__main__":
    main()
