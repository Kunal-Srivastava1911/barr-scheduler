"""Burst-Aware Adaptive Round Robin.

BARR is the composition of a burst predictor, a quantum controller and a ready
queue. The controller implements equation (3):

    q_i = min(q_max, max(q_min, (1 + delta)(p_i - u_i)))

where u_i is the CPU time already used in the burst. When a quantum expires the
prediction is raised to max(p_i, beta * u_i), so repeated under-prediction grows
the quantum geometrically up to q_max. A waiting process that has been passed by
more than A dispatches moves to the head of the queue (aging).

order="fifo" is the original BARR; order="spf" is BARR-S, which serves the
smallest predicted remaining burst first.
"""

from .base import Scheduler


class QuantumController:
    def __init__(self, delta=0.25, q_min=1.0, q_max=50.0, beta=2.0):
        self.delta, self.q_min, self.q_max, self.beta = delta, q_min, q_max, beta

    def quantum(self, p):
        q = (1 + self.delta) * (p.pred - p.used)
        return min(self.q_max, max(self.q_min, q))

    def on_expiry(self, p):
        p.pred = max(p.pred, self.beta * p.used)


class BARR(Scheduler):
    def __init__(self, predictor, delta=0.25, q_min=1.0, q_max=50.0, beta=2.0,
                 aging=64, order="fifo", name=None):
        self.predictor = predictor
        self.ctrl = QuantumController(delta, q_min, q_max, beta)
        self.aging = aging
        self.order = order
        self.queue = []
        self.dispatches = 0
        self.aged = 0
        tag = "BARR-S" if order == "spf" else "BARR"
        self.name = name or f"{tag}-{predictor.name.upper()}"

    def _enqueue(self, p):
        p.enq_seq = self.dispatches
        self.queue.append(p)

    def on_ready(self, p, t):
        p.pred = self.predictor.predict(p)
        self._enqueue(p)

    def on_preempt(self, p, t):
        self.ctrl.on_expiry(p)
        self._enqueue(p)

    def on_burst_end(self, p, t, length):
        self.predictor.update(p, length)

    def _key(self, p):
        return (p.pred - p.used, p.order) if self.order == "spf" else (p.enq_seq, p.order)

    def pick_next(self, t):
        if not self.queue:
            return None, None
        # aging: the process passed by the most dispatches, if above the threshold
        oldest = min(range(len(self.queue)), key=lambda i: self.queue[i].enq_seq)
        if self.aging is not None and self.dispatches - self.queue[oldest].enq_seq > self.aging:
            i = oldest
            if self.order == "spf":
                self.aged += 1
        elif self.order == "fifo":
            i = 0
        else:
            i = min(range(len(self.queue)), key=lambda j: self._key(self.queue[j]))
        p = self.queue.pop(i)
        self.dispatches += 1
        return p, self.ctrl.quantum(p)

    def snapshot(self):
        return [p.pid for p in sorted(self.queue, key=self._key)]
