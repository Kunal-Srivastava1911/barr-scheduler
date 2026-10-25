"""Dynamic quantum RR from the median remaining burst of the ready queue.

MedianRR follows Matarneh [2]: at the start of every round the quantum is set
to the median remaining burst of the processes in the ready queue, and each of
them is served once with that quantum. With reorder=True the queue is also
arranged in ascending order of remaining burst at the start of every round, as
in Behera et al. [3]. In oracle mode the true remaining burst is used, as in
the papers; in predicted mode it is replaced by the predicted burst minus the
CPU time already used, with a small positive floor.
"""

import numpy as np

from ..predictors.ema import EMAPredictor
from .base import FifoScheduler

FLOOR = 0.1  # ms


class _DynamicRR(FifoScheduler):
    def __init__(self, mode="oracle", predictor=None, q_min=FLOOR):
        super().__init__()
        self.mode = mode
        self.predictor = predictor or EMAPredictor()
        self.q_min = q_min

    def estimate(self, p):
        if self.mode == "oracle":
            return p.remaining
        return max(FLOOR, p.pred - p.used)

    def on_ready(self, p, t):
        p.pred = p.burst if self.mode == "oracle" else self.predictor.predict(p)
        self.queue.append(p)

    def on_burst_end(self, p, t, length):
        self.predictor.update(p, length)


class MedianRR(_DynamicRR):
    def __init__(self, mode="oracle", predictor=None, reorder=False):
        super().__init__(mode, predictor)
        self.reorder = reorder
        self.round_left = 0
        self.q = self.q_min
        base = "Behera RR" if reorder else "Median RR"
        self.name = f"{base} ({'oracle' if mode == 'oracle' else 'pred.'})"

    def pick_next(self, t):
        if not self.queue:
            return None, None
        if self.round_left <= 0:
            if self.reorder:
                self.queue = type(self.queue)(sorted(self.queue, key=lambda p: (self.estimate(p), p.order)))
            self.q = max(self.q_min, float(np.median([self.estimate(p) for p in self.queue])))
            self.round_left = len(self.queue)
        self.round_left -= 1
        return self.queue.popleft(), self.q
