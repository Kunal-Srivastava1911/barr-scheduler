"""Demonstration: Gantt charts of RR (q = 4 ms), BARR and BARR-ORACLE on a
four process mixed compute bound / interactive workload (final review, Fig. 3).

    python -m barr.demo
"""

from .metrics.collector import collect
from .predictors.base import OraclePredictor
from .predictors.ema import EMAPredictor
from .sched.barr import BARR
from .sched.rr import RoundRobin
from .sim.engine import simulate
from .workloads.synthetic import demo_workload

SCALE = 2  # characters per ms


def gantt(res, width_ms):
    pids = sorted(p.pid for p in res.procs)
    lines = []
    for pid in pids:
        row = [" "] * int(width_ms * SCALE + 1)
        for spid, _, cs_start, start, end in res.slices:
            if spid != pid:
                continue
            for x in range(int(round(start * SCALE)), int(round(end * SCALE))):
                if x < len(row):
                    row[x] = "#"
        lines.append(f"  P{pid} |{''.join(row)}|")
    axis = "".join(f"{t:<{5 * SCALE}d}" for t in range(0, int(width_ms) + 1, 5))
    lines.append("       " + axis + " ms")
    return "\n".join(lines)


def main():
    work = demo_workload()
    c = 0.1
    runs = [
        (RoundRobin(4), "RR q=4"),
        (BARR(EMAPredictor(alpha=0.5, tau0=5.0), q_max=50, name="BARR-EMA"), "BARR-EMA"),
        (BARR(OraclePredictor(), q_max=50, name="BARR-ORACLE"), "BARR-ORACLE"),
    ]
    results = [(name, simulate(s, work, c)) for s, name in runs]
    width = max(r.end for _, r in results)
    for name, res in results:
        print(f"\n{name}  (context switch cost {c} ms)")
        print(gantt(res, width))
    print(f"\n{'policy':14s}{'mean WT':>9s}{'mean TAT':>10s}{'P95 resp':>10s}{'switches':>10s}{'1st-q %':>9s}")
    for name, res in results:
        m = collect(res)
        print(f"{name:14s}{m['mean_wt']:9.2f}{m['mean_tat']:10.2f}{m['p95_resp']:10.2f}"
              f"{m['n_cs']:10d}{m['first_q'] * 100:9.0f}")


if __name__ == "__main__":
    main()
