"""Scheduler interface shared by every policy, including BARR."""

import math
from collections import deque

INF = math.inf


class Scheduler:
    name = "base"
    preemptive = False   # True if the policy may preempt on a new ready process

    def on_ready(self, p, t):
        """A burst of p became ready (arrival or end of I/O)."""
        raise NotImplementedError

    def pick_next(self, t):
        """Return (process, quantum) or (None, None) if the queue is empty."""
        raise NotImplementedError

    def on_preempt(self, p, t):
        """p used up its quantum (or was preempted) and still has work left."""
        raise NotImplementedError

    def on_burst_end(self, p, t, length):
        pass

    def should_preempt(self, p, remaining, t):
        return False

    def snapshot(self):
        return []


class FifoScheduler(Scheduler):
    """Helper base: a FIFO ready queue."""

    def __init__(self):
        self.queue = deque()

    def on_ready(self, p, t):
        self.queue.append(p)

    def on_preempt(self, p, t):
        self.queue.append(p)

    def snapshot(self):
        return [p.pid for p in self.queue]
