"""Dynamic quantum RR from the mean remaining burst of the ready queue (Noon et
al. [4]); recomputed whenever the queue changes, i.e. at every dispatch."""

from .dyn_median import _DynamicRR


class MeanRR(_DynamicRR):
    def __init__(self, mode="oracle", predictor=None):
        super().__init__(mode, predictor)
        self.name = f"Mean RR ({'oracle' if mode == 'oracle' else 'pred.'})"

    def pick_next(self, t):
        if not self.queue:
            return None, None
        ests = [self.estimate(p) for p in self.queue]
        return self.queue.popleft(), max(self.q_min, sum(ests) / len(ests))
