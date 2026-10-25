"""Experiment runner: every scheduler on every workload of a configuration.

    python -m barr.experiments.run barr/experiments/configs/draft.json

Writes results/<name>/runs.csv (one row per scheduler x workload) and
results/<name>/summary.json (Table III style ratios to the best fixed RR,
bootstrap confidence intervals, Wilcoxon tests and the planned thresholds).
"""

import argparse
import csv
import json
import os
import time

import numpy as np

from ..analysis.stats import bootstrap_ci, geomean, wilcoxon, pct_change
from ..metrics.collector import collect
from ..predictors.train import get_or_train
from ..sched.registry import make, label, ORACLE
from ..sim.engine import simulate
from ..traces.load_linux import available_mixes, load_mix, window
from ..workloads.synthetic import generate

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
KEY_METRICS = ["mean_wt", "n_cs", "p95_resp", "max_wt"]


def workloads(cfg):
    """Yields (workload_id, group, processes)."""
    for fam in cfg.get("families", []):
        for load in cfg.get("loads", [0.7]):
            for seed in cfg.get("seeds", [1, 2, 3]):
                yield (f"{fam}/rho={load}/seed={seed}", "synthetic",
                       generate(fam, cfg.get("n_procs", 60), load, seed,
                                tuple(cfg.get("bursts", (10, 40)))))
    lin = cfg.get("linux")
    if lin:
        for path in available_mixes():
            mix = os.path.splitext(os.path.basename(path))[0]
            if lin.get("mixes") and mix not in lin["mixes"]:
                continue
            procs = load_mix(path)
            end = max(p.arrival for p in procs)
            test_start = 0.8 * end            # windows from the test period only
            span = end - test_start
            n_win = lin.get("windows", 2)
            for w in range(n_win):
                s = test_start + w * span / n_win
                ps = window(procs, s, lin.get("window_ms", 2000))
                if ps:
                    yield f"linux:{mix}/w{w}", "linux", ps


def run(cfg, log=print, progress=None, trained=None):
    name = cfg.get("name", "experiment")
    out_dir = os.path.join(RESULTS, name)
    os.makedirs(out_dir, exist_ok=True)
    if trained is None:
        trained, _ = get_or_train(log=log)
    scheds = cfg["schedulers"]
    wls = list(workloads(cfg))
    total = len(wls) * len(scheds)
    log(f"{len(wls)} workloads x {len(scheds)} schedulers = {total} runs")
    rows, done, t0 = [], 0, time.time()
    for wid, group, procs in wls:
        for s in scheds:
            sched = make(s, trained, cfg.get("barr"), cfg.get("barr_s"))
            m = collect(simulate(sched, procs, cfg.get("cs_cost", 0.1)))
            row = {"workload": wid, "group": group, "scheduler": s, **m}
            rows.append(row)
            done += 1
            if progress:
                progress(done, total, row)
        log(f"  [{done}/{total}] {wid} done ({time.time() - t0:.1f} s)")
    with open(os.path.join(out_dir, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, scheds)
    summary["config"] = cfg
    summary["seconds"] = time.time() - t0
    summary = clean(summary)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"results in {out_dir}")
    return rows, summary


def summarize(rows, scheds):
    """Ratios to the best fixed RR of each workload (lowest mean WT), geometric
    mean over workloads, per group; bootstrap CI of raw metrics; Wilcoxon tests."""
    rr = [s for s in scheds if s.startswith("RR-")]
    by_wl = {}
    for r in rows:
        by_wl.setdefault(r["workload"], {})[r["scheduler"]] = r
    groups = sorted({r["group"] for r in rows})
    out = {"groups": {}, "schedulers": [{"name": s, "label": label(s), "oracle": s in ORACLE} for s in scheds]}
    for g in groups:
        wls = [w for w, d in by_wl.items() if next(iter(d.values()))["group"] == g]
        best_rr = {}
        for w in wls:
            if rr:
                best_rr[w] = min(rr, key=lambda s: by_wl[w][s]["mean_wt"])
        ratios, raw = {}, {}
        for s in scheds:
            ratios[s], raw[s] = {}, {}
            for m in KEY_METRICS + ["mean_tat", "mean_resp", "jain", "first_q", "pred_mae"]:
                vals = [by_wl[w][s][m] for w in wls]
                raw[s][m] = bootstrap_ci(vals)
                if best_rr and m in KEY_METRICS:
                    ratios[s][m] = geomean([by_wl[w][s][m] / max(by_wl[w][best_rr[w]][m], 1e-9) for w in wls])
        best_counts = {}
        for w, s in best_rr.items():
            best_counts[s] = best_counts.get(s, 0) + 1
        tests = _tests(by_wl, wls, scheds, best_rr)
        out["groups"][g] = {"n_workloads": len(wls), "ratios": ratios, "raw": raw,
                            "best_rr": best_counts, "tests": tests,
                            "thresholds": _thresholds(by_wl, wls, scheds, best_rr)}
    return out


def _tests(by_wl, wls, scheds, best_rr):
    """Wilcoxon signed-rank: each BARR variant vs the strongest non-oracle
    baseline (lowest geometric mean WT among non-BARR, non-oracle policies)."""
    base = [s for s in scheds if not s.startswith("BARR") and s not in ORACLE]
    if not base or len(wls) < 2:
        return []
    strongest = min(base, key=lambda s: geomean([by_wl[w][s]["mean_wt"] for w in wls]))
    res = []
    for s in scheds:
        if not s.startswith("BARR") or s in ORACLE:
            continue
        for m in KEY_METRICS:
            x = [by_wl[w][s][m] for w in wls]
            y = [by_wl[w][strongest][m] for w in wls]
            wstat, p, n = wilcoxon(x, y)
            res.append({"scheduler": s, "baseline": strongest, "metric": m, "W": wstat, "p": p, "n": n,
                        "change_pct": pct_change(float(np.mean(x)), float(np.mean(y)))})
    return res


def _thresholds(by_wl, wls, scheds, best_rr):
    """The scheduling thresholds fixed in the implementation plan."""
    out = []
    qmax_rr = "RR-50" if "RR-50" in scheds else None
    for s in scheds:
        if not s.startswith("BARR") or s in ORACLE or not best_rr:
            continue
        cs = geomean([by_wl[w][s]["n_cs"] / max(by_wl[w][best_rr[w]]["n_cs"], 1) for w in wls])
        p95 = geomean([by_wl[w][s]["p95_resp"] / max(by_wl[w][best_rr[w]]["p95_resp"], 1e-9) for w in wls])
        item = {"scheduler": s, "cs_ratio": cs, "p95_ratio": p95,
                "cs_met": cs <= 0.80, "p95_met": p95 <= 1.10}
        if qmax_rr and not s.startswith("BARR-S"):
            mx = geomean([by_wl[w][s]["max_wt"] / max(by_wl[w][qmax_rr]["max_wt"], 1e-9) for w in wls])
            item.update({"maxwt_ratio_qmax": mx, "maxwt_met": mx <= 1.0 + 1e-9})
        out.append(item)
    return out


def clean(x):
    """JSON-safe copy: NaN / inf -> None, numpy scalars -> Python numbers."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def load_config(path):
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            import yaml  # optional
            return yaml.safe_load(f)
        return json.load(f)


def print_table(summary):
    for g, d in summary["groups"].items():
        print(f"\n{g}: ratio to best fixed RR (geometric mean over {d['n_workloads']} workloads)")
        print(f"{'scheduler':22s}" + "".join(f"{m:>10s}" for m in KEY_METRICS))
        for s, r in d["ratios"].items():
            print(f"{label(s):22s}" + "".join(f"{r.get(m, float('nan')):10.2f}" for m in KEY_METRICS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=os.path.join(os.path.dirname(__file__), "configs", "draft.json"))
    a = ap.parse_args()
    _, summary = run(load_config(a.config))
    print_table(summary)


if __name__ == "__main__":
    main()
