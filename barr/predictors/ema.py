"""Exponential averaging, equation (1): tau_{n+1} = alpha t_n + (1 - alpha) tau_n."""

from .base import Predictor


class EMAPredictor(Predictor):
    name = "ema"

    def __init__(self, alpha=0.5, tau0=5.0):
        self.alpha = alpha
        self.tau0 = tau0
        self.tau = {}

    def predict(self, p):
        return self.tau.get(p.pid, self.tau0)

    def update(self, p, length):
        tau = self.tau.get(p.pid, self.tau0)
        self.tau[p.pid] = self.alpha * length + (1 - self.alpha) * tau

    def fresh(self):
        return EMAPredictor(self.alpha, self.tau0)
