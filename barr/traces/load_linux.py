"""Load parsed Linux bursts (common schema CSV, or Parquet when pandas and
pyarrow are installed) and turn them into simulator workloads.

Schema: pid, source, arrival, burst_index, burst_length, blocked_before,
priority, preempted. Times are in milliseconds.
"""

import csv
import glob
import os
from collections import defaultdict

from ..sim.process import Process

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data", "linux")


def read_rows(path):
    if path.endswith(".parquet"):
        import pandas as pd  # optional
        return pd.read_parquet(path).to_dict("records")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def rows_to_procs(rows, pid_offset=0):
    """One process per (source, pid); the I/O wait after burst i is the
    blocked time recorded before burst i+1."""
    by = defaultdict(list)
    for r in rows:
        by[(r["source"], int(r["pid"]))].append(r)
    procs = []
    for k, ((source, pid), rs) in enumerate(sorted(by.items())):
        rs.sort(key=lambda r: int(r["burst_index"]))
        lengths = [float(r["burst_length"]) for r in rs]
        blocked = [float(r["blocked_before"]) for r in rs]
        io = blocked[1:] + [0.0]
        procs.append(Process(pid_offset + k + 1, float(rs[0]["arrival"]), list(zip(lengths, io)),
                             int(float(rs[0].get("priority", 0) or 0)),
                             name=f"{source}-{pid}", kind=source))
    t0 = min((p.arrival for p in procs), default=0.0)
    for p in procs:
        p.arrival -= t0
    return procs


def available_mixes(data_dir=DATA_DIR):
    files = glob.glob(os.path.join(data_dir, "*.csv")) + glob.glob(os.path.join(data_dir, "*.parquet"))
    return sorted(files)


def load_mix(path):
    return rows_to_procs(read_rows(path))


def window(procs, start, length):
    """Processes whose first burst arrives in [start, start + length), re-based to 0."""
    sel = [p for p in procs if start <= p.arrival < start + length]
    out = []
    for p in sel:
        out.append(Process(p.pid, p.arrival - start, list(p.bursts), p.priority, p.name, p.kind))
    return out
