"""Discrete event simulator: one CPU, one ready queue, pluggable scheduler.

The scheduler interface (see barr.sched.base) has on_ready, pick_next,
on_preempt and on_burst_end; the engine never knows which policy it runs.
Every slice and every completed burst is written to a log, and all metrics are
computed afterwards from that log (barr.metrics.collector).
"""

import heapq
from dataclasses import dataclass, field

from .events import ARRIVAL, IO_DONE, SLICE_END, EPS
from .process import clone_workload


@dataclass
class SimResult:
    scheduler: str
    procs: list
    slices: list = field(default_factory=list)   # (pid, burst_idx, cs_start, start, end)
    bursts: list = field(default_factory=list)   # dicts, one per completed burst
    queue_log: list = field(default_factory=list)  # (t, running_pid, [queued pids])
    n_cs: int = 0
    cs_cost: float = 0.0
    start: float = 0.0
    end: float = 0.0


class _Slice:
    __slots__ = ("p", "cs_start", "start", "run", "live")

    def __init__(self, p, cs_start, start, run):
        self.p, self.cs_start, self.start, self.run, self.live = p, cs_start, start, run, True


class Simulator:
    def __init__(self, scheduler, cs_cost=0.1, record_queue=False):
        self.sched = scheduler
        self.c = cs_cost
        self.record_queue = record_queue

    def run(self, workload):
        procs = clone_workload(workload)
        res = SimResult(self.sched.name, procs, cs_cost=self.c)
        self.res = res
        self.heap = []
        self.seq = 0
        self.running = None
        self.last_pid = None
        self.order = 0
        for p in procs:
            self._push(p.arrival, ARRIVAL, p)
        res.start = min((p.arrival for p in procs), default=0.0)

        t = res.start
        while self.heap:
            t = self.heap[0][0]
            batch = []
            while self.heap and self.heap[0][0] <= t + EPS:
                batch.append(heapq.heappop(self.heap))
            batch.sort(key=lambda e: (e[1], e[2]))
            for _, kind, _, obj in batch:
                if kind == SLICE_END:
                    if obj.live:
                        self._end_slice(obj, t)
                else:
                    self._make_ready(obj, t)
            if self.running is not None and self.sched.preemptive:
                sl = self.running
                left = sl.p.remaining - max(0.0, t - sl.start)
                if self.sched.should_preempt(sl.p, left, t):
                    sl.live = False
                    sl.start = min(sl.start, t)      # preempted during the switch itself
                    sl.run = t - sl.start
                    self._end_slice(sl, t)
            if self.running is None:
                self._dispatch(t)
        res.end = max((p.completion for p in procs if p.completion is not None), default=t)
        return res

    # ------------------------------------------------------------------ events
    def _push(self, t, kind, obj):
        self.seq += 1
        heapq.heappush(self.heap, (t, kind, self.seq, obj))

    def _make_ready(self, p, t):
        p.remaining = p.burst
        p.used = 0.0
        p.ready_at = t
        p.dispatched = False
        p.preempted = False
        p.pred = 0.0
        self.order += 1
        p.order = self.order
        self.sched.on_ready(p, t)

    def _dispatch(self, t):
        p, q = self.sched.pick_next(t)
        if p is None:
            return
        cs = 0.0
        if self.last_pid is not None and p.pid != self.last_pid:
            cs = self.c
            self.res.n_cs += 1
        start = t + cs
        q = max(q, EPS)
        if not p.dispatched:
            p.dispatched = True
            p.first_dispatch = start
            p.first_quantum = q
            p.pred0 = p.pred
        run = min(q, p.remaining)
        sl = _Slice(p, t, start, run)
        self.running = sl
        self.last_pid = p.pid
        if self.record_queue:
            self.res.queue_log.append((t, p.pid, self.sched.snapshot()))
        self._push(start + run, SLICE_END, sl)

    def _end_slice(self, sl, t):
        p = sl.p
        self.running = None
        if sl.run > 0 or sl.start > sl.cs_start:
            self.res.slices.append((p.pid, p.idx, sl.cs_start, sl.start, sl.start + sl.run))
        p.remaining -= sl.run
        p.used += sl.run
        if p.remaining > EPS:
            p.preempted = True
            self.sched.on_preempt(p, t)
            return

        length = p.burst
        self.res.bursts.append({
            "pid": p.pid, "idx": p.idx, "ready": p.ready_at, "first": p.first_dispatch,
            "end": t, "length": length, "pred": p.pred0, "q1": p.first_quantum,
            "preempted": p.preempted,
        })
        p.history.append(length)
        self.sched.on_burst_end(p, t, length)
        io = p.bursts[p.idx][1]
        p.idx += 1
        if p.idx >= len(p.bursts):
            p.completion = t
            return
        p.blocked = io
        if io > EPS:
            self._push(t + io, IO_DONE, p)
        else:
            self._make_ready(p, t)


def simulate(scheduler, workload, cs_cost=0.1, record_queue=False):
    return Simulator(scheduler, cs_cost, record_queue).run(workload)
