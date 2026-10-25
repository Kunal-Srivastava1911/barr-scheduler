from ..predictors.ema import EMAPredictor
from .base import Scheduler, INF


class SJF(Scheduler):
    """Non-preemptive shortest job first on the predicted next burst
    (exponential averaging by default, the textbook practical SJF)."""

    def __init__(self, predictor=None, name="SJF-EMA"):
        self.predictor = predictor or EMAPredictor()
        self.name = name
        self.queue = []

    def on_ready(self, p, t):
        p.pred = self.predictor.predict(p)
        self.queue.append(p)

    def pick_next(self, t):
        if not self.queue:
            return None, None
        p = min(self.queue, key=lambda x: (x.pred, x.order))
        self.queue.remove(p)
        return p, INF

    def on_preempt(self, p, t):
        self.queue.append(p)

    def on_burst_end(self, p, t, length):
        self.predictor.update(p, length)

    def snapshot(self):
        return [p.pid for p in sorted(self.queue, key=lambda x: (x.pred, x.order))]


class SRTF(Scheduler):
    """Preemptive shortest remaining time first with oracle burst lengths.
    Lower bound on mean waiting time; not achievable by a real scheduler."""

    name = "SRTF oracle"
    preemptive = True

    def __init__(self):
        self.queue = []

    def on_ready(self, p, t):
        p.pred = p.burst
        self.queue.append(p)

    def on_preempt(self, p, t):
        self.queue.append(p)

    def pick_next(self, t):
        if not self.queue:
            return None, None
        p = min(self.queue, key=lambda x: (x.remaining, x.order))
        self.queue.remove(p)
        return p, INF

    def should_preempt(self, p, remaining, t):
        return bool(self.queue) and min(x.remaining for x in self.queue) < remaining - 1e-9

    def snapshot(self):
        return [p.pid for p in sorted(self.queue, key=lambda x: (x.remaining, x.order))]
