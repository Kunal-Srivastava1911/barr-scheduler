"""One-layer LSTM over the last 16 log burst lengths (accuracy upper bound).

Optional: needs PyTorch. available() tells the caller whether it can be used.
"""

import numpy as np

from ..features.build import TAU0, to_log, from_log
from .base import Predictor

SEQ = 16

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def available():
    return torch is not None


def sequences(procs, split=(0.6, 0.2, 0.2)):
    """(X, y) per part: X[r] = last SEQ log bursts before burst i (zero padded)."""
    parts = {k: ([], []) for k in ("train", "val", "test")}
    for p in procs:
        lens = to_log(np.array([b[0] for b in p.bursts], dtype=float))
        n = len(lens) - 1
        if n < 1:
            continue
        a, b = int(round(n * split[0])), int(round(n * (split[0] + split[1])))
        for i in range(1, len(lens)):
            x = np.zeros(SEQ)
            w = lens[max(0, i - SEQ):i]
            x[SEQ - len(w):] = w
            part = "train" if i - 1 < a else "val" if i - 1 < b else "test"
            parts[part][0].append(x)
            parts[part][1].append(lens[i])
    return {k: (np.array(x).reshape(-1, SEQ), np.array(y)) for k, (x, y) in parts.items()}


class LSTMModel:
    def __init__(self, hidden=32, epochs=30, lr=3e-3, patience=4, seed=0):
        self.hidden, self.epochs, self.lr, self.patience, self.seed = hidden, epochs, lr, patience, seed

    def fit(self, X, y, X_val=None, y_val=None, log=print):
        torch.manual_seed(self.seed)
        self.net = _Net(self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        yt = torch.tensor(y, dtype=torch.float32)
        Xv = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
        yv = torch.tensor(y_val, dtype=torch.float32)
        best, state, since = np.inf, None, 0
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xt))
            for s in range(0, len(Xt), 512):
                b = perm[s:s + 512]
                opt.zero_grad()
                loss = (self.net(Xt[b]) - yt[b]).abs().mean()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = (self.net(Xv) - yv).abs().mean().item()
            if v < best - 1e-5:
                best, since = v, 0
                state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                since += 1
                if since >= self.patience:
                    break
        self.net.load_state_dict(state)
        self.net.eval()
        return self

    def predict(self, X):
        with torch.no_grad():
            return self.net(torch.tensor(X, dtype=torch.float32).unsqueeze(-1)).numpy()


if torch is not None:
    class _Net(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1]).squeeze(-1)


class LSTMPredictor(Predictor):
    name = "lstm"

    def __init__(self, model):
        self.model = model

    def predict(self, p):
        if not p.history:
            return TAU0
        w = to_log(np.asarray(p.history[-SEQ:], dtype=float))
        x = np.zeros((1, SEQ))
        x[0, SEQ - len(w):] = w
        return float(from_log(self.model.predict(x)[0]))

    def fresh(self):
        return LSTMPredictor(self.model)
