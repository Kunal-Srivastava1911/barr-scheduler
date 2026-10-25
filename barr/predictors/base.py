"""Predictor interface: predict(p) -> next burst (ms), update(p, length)."""


class Predictor:
    name = "base"

    def predict(self, p):
        raise NotImplementedError

    def update(self, p, length):
        pass

    def fresh(self):
        """A copy with empty per-process state, for a new simulation run."""
        return self


class OraclePredictor(Predictor):
    """Returns the true burst length. Reference only; not achievable online."""
    name = "oracle"

    def predict(self, p):
        return p.burst


class LastValuePredictor(Predictor):
    name = "last"

    def __init__(self, default=5.0):
        self.default = default

    def predict(self, p):
        return p.history[-1] if p.history else self.default
