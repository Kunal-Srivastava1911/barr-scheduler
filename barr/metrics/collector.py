"""Evaluation parameters of Table IV (implementation plan), computed from the
event log of one simulation run.

    TAT_j = C_j - A_j              WT_j = TAT_j - S_j - I_j
    SD_j  = TAT_j / (S_j + I_j)    J    = (sum SD)^2 / (n sum SD^2)
"""

import numpy as np

from .fairness import jain_index

# name -> (label, unit, better)
METRIC_INFO = {
    "mean_wt": ("Mean waiting time", "ms", "lower"),
    "mean_tat": ("Mean turnaround time", "ms", "lower"),
    "max_wt": ("Maximum waiting time", "ms", "lower"),
    "mean_resp": ("Mean response time", "ms", "lower"),
    "p95_resp": ("P95 response time", "ms", "lower"),
    "n_cs": ("Context switches", "", "lower"),
    "cs_share": ("Switch overhead share", "", "lower"),
    "throughput": ("Throughput", "proc/s", "higher"),
    "util": ("Useful CPU utilisation", "", "higher"),
    "mean_sd": ("Mean slowdown", "", "lower"),
    "jain": ("Jain fairness index", "", "higher"),
    "first_q": ("Bursts done in first quantum", "", "higher"),
    "pred_mae": ("Prediction MAE", "ms", "lower"),
    "pred_under": ("Under-predicted share", "", "lower"),
    "makespan": ("Makespan", "ms", "lower"),
}


def collect(res):
    procs = [p for p in res.procs if p.completion is not None]
    n = len(procs)
    if n == 0:
        return {k: float("nan") for k in METRIC_INFO}
    tat = np.array([p.completion - p.arrival for p in procs])
    service = np.array([p.cpu_total for p in procs])
    io = np.array([p.io_total for p in procs])
    wt = tat - service - io
    sd = tat / np.maximum(service + io, 1e-9)

    bursts = res.bursts
    resp = np.array([b["first"] - b["ready"] for b in bursts]) if bursts else np.zeros(1)
    first_q = np.mean([not b["preempted"] for b in bursts]) if bursts else float("nan")
    T = max(res.end - res.start, 1e-9)

    out = {
        "mean_wt": float(wt.mean()),
        "mean_tat": float(tat.mean()),
        "max_wt": float(wt.max()),
        "mean_resp": float(resp.mean()),
        "p95_resp": float(np.percentile(resp, 95)),
        "n_cs": int(res.n_cs),
        "cs_share": float(res.n_cs * res.cs_cost / T),
        "throughput": float(n / T * 1000.0),
        "util": float(service.sum() / T),
        "mean_sd": float(sd.mean()),
        "jain": float(jain_index(sd)),
        "first_q": float(first_q),
        "makespan": float(T),
    }
    preds = [(b["pred"], b["length"]) for b in bursts if b["pred"] and b["pred"] > 0]
    if preds:
        pr = np.array(preds)
        out["pred_mae"] = float(np.abs(pr[:, 0] - pr[:, 1]).mean())
        out["pred_under"] = float(np.mean(pr[:, 0] < pr[:, 1]))
    else:
        out["pred_mae"] = float("nan")
        out["pred_under"] = float("nan")
    return out
