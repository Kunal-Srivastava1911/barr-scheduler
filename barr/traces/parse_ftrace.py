"""Burst extraction from an ftrace text trace (sched_switch, sched_wakeup,
sched_process_fork).

A run interval of a thread is the time between its switch-in and its next
switch-out. When a thread is switched out while still runnable (state R) it was
preempted and its burst continues in the next interval, so consecutive
intervals are merged until the thread is switched out sleeping (S, D) or exits.
The time from that switch-out to the next wakeup is the blocked time before the
next burst. The idle task and kernel threads are removed; with root_pid set,
only threads that descend from the workload root process are kept.

    python -m barr.traces.parse_ftrace trace.txt --root 1234 --source compile -o bursts.csv
"""

import argparse
import csv
import re

LINE = re.compile(r"^\s*(?P<task>.+?)-(?P<pid>\d+)\s+\[(?P<cpu>\d+)\].*?\s(?P<ts>\d+\.\d+):\s+"
                  r"(?P<event>\w+):\s+(?P<body>.*)$")
KV = re.compile(r"(\w+)=(\S+)")
KERNEL_PREFIXES = ("kworker", "ksoftirqd", "migration", "rcu_", "kthreadd", "idle_inject",
                   "cpuhp", "watchdog", "swapper", "irq/", "kcompactd", "khugepaged")

COLUMNS = ["pid", "source", "arrival", "burst_index", "burst_length", "blocked_before",
           "priority", "preempted"]


def _kv(body):
    """Key=value pairs; for sched_switch the part after '==>' is prefixed 'next_'."""
    return dict(KV.findall(body))


class _Thread:
    __slots__ = ("pid", "comm", "prio", "first", "run_start", "acc", "preempted",
                 "sleep_start", "blocked", "bursts")

    def __init__(self, pid, comm, prio):
        self.pid, self.comm, self.prio = pid, comm, prio
        self.first = None
        self.run_start = None
        self.acc = 0.0
        self.preempted = False
        self.sleep_start = None
        self.blocked = 0.0
        self.bursts = []   # (start_ms, length_ms, blocked_before_ms, preempted)


def parse_lines(lines, root_pid=None):
    threads = {}
    keep = {root_pid} if root_pid is not None else None
    burst_start = {}

    def th(pid, comm="", prio=120):
        if pid not in threads:
            threads[pid] = _Thread(pid, comm, prio)
        return threads[pid]

    for line in lines:
        m = LINE.match(line)
        if not m:
            continue
        ts = float(m["ts"]) * 1000.0
        ev, kv = m["event"], _kv(m["body"])
        if ev == "sched_process_fork":
            if keep is not None and int(kv.get("pid", -1)) in keep:
                keep.add(int(kv["child_pid"]))
        elif ev == "sched_wakeup":
            t = th(int(kv["pid"]), kv.get("comm", ""))
            if t.sleep_start is not None:
                t.blocked = ts - t.sleep_start
                t.sleep_start = None
        elif ev == "sched_switch":
            prev, nxt = int(kv["prev_pid"]), int(kv["next_pid"])
            if prev != 0:
                t = th(prev, kv.get("prev_comm", ""), int(kv.get("prev_prio", 120)))
                if t.run_start is not None:
                    t.acc += ts - t.run_start
                    t.run_start = None
                    state = kv.get("prev_state", "S")
                    if state.startswith("R"):
                        t.preempted = True
                    else:
                        t.bursts.append((burst_start.get(prev, ts), t.acc, t.blocked, t.preempted))
                        t.acc, t.preempted, t.blocked = 0.0, False, 0.0
                        t.sleep_start = ts if state[0] in "SDI" else None
            if nxt != 0:
                # the 'next_*' keys appear after '==>' in the body
                comm = kv.get("next_comm", "")
                t = th(nxt, comm, int(kv.get("next_prio", 120)))
                if t.acc == 0.0 and t.run_start is None:
                    burst_start[nxt] = ts
                t.run_start = ts
                if t.first is None:
                    t.first = ts

    out = []
    for t in threads.values():
        if keep is not None and t.pid not in keep:
            continue
        if t.comm.startswith(KERNEL_PREFIXES) or not t.bursts:
            continue
        out.append(t)
    return out


def to_rows(threads, source):
    rows = []
    for t in threads:
        for i, (start, length, blocked, pre) in enumerate(t.bursts):
            rows.append({"pid": t.pid, "source": source, "arrival": round(start, 6),
                         "burst_index": i, "burst_length": round(length, 6),
                         "blocked_before": round(blocked, 6), "priority": t.prio - 120,
                         "preempted": int(pre)})
    return rows


def parse_file(path, root_pid=None, source="linux"):
    with open(path, encoding="utf-8", errors="replace") as f:
        return to_rows(parse_lines(f, root_pid), source)


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace")
    ap.add_argument("--root", type=int, default=None, help="workload root pid")
    ap.add_argument("--source", default="linux")
    ap.add_argument("-o", "--out", default="bursts.csv")
    a = ap.parse_args()
    rows = parse_file(a.trace, a.root, a.source)
    write_csv(rows, a.out)
    print(f"{len(rows)} bursts from {len({r['pid'] for r in rows})} threads -> {a.out}")


if __name__ == "__main__":
    main()
