"""Wrapper that turns a regressor on log burst length into an online predictor.

Processes with no completed burst use the exponential-average default (cold
start), as in Step 1 of the BARR algorithm.
"""

import numpy as np

from ..features.build import FEATURE_ALPHA, TAU0, N_FEATURES, feature_vector, from_log
from .base import Predictor


class LearnedPredictor(Predictor):
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.tau = {}
        self._buf = np.zeros((1, N_FEATURES))

    def predict(self, p):
        if not p.history:
            return TAU0
        feature_vector(p.history, self.tau.get(p.pid, TAU0), p.blocked, p.priority, self._buf[0])
        return float(from_log(self.model.predict(self._buf)[0]))

    def update(self, p, length):
        tau = self.tau.get(p.pid, TAU0)
        self.tau[p.pid] = FEATURE_ALPHA * length + (1 - FEATURE_ALPHA) * tau

    def fresh(self):
        return LearnedPredictor(self.model, self.name)


class Ridge:
    """Linear (ridge) regression sanity model, closed form."""

    def __init__(self, l2=1.0):
        self.l2 = l2

    def fit(self, X, y, X_val=None, y_val=None):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        Z = np.column_stack([(X - self.mu) / self.sd, np.ones(len(X))])
        reg = self.l2 * np.eye(Z.shape[1])
        reg[-1, -1] = 0
        self.w = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
        return self

    def predict(self, X):
        return ((X - self.mu) / self.sd) @ self.w[:-1] + self.w[-1]


class KNN:
    """k nearest neighbours (median of neighbour targets) on standardised features."""

    def __init__(self, k=10, max_rows=4000, seed=0):
        self.k, self.max_rows, self.seed = k, max_rows, seed

    def fit(self, X, y, X_val=None, y_val=None):
        if len(X) > self.max_rows:
            idx = np.random.default_rng(self.seed).choice(len(X), self.max_rows, replace=False)
            X, y = X[idx], y[idx]
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        self.X = (X - self.mu) / self.sd
        self.y = y
        return self

    def predict(self, X):
        Z = (X - self.mu) / self.sd
        out = np.empty(len(Z))
        for s in range(0, len(Z), 256):
            d = ((Z[s:s + 256, None, :] - self.X[None, :, :]) ** 2).sum(-1)
            nn = np.argpartition(d, self.k, axis=1)[:, :self.k]
            out[s:s + 256] = np.median(self.y[nn], axis=1)
        return out
