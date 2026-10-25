"""Feature builder.

For burst i of a process the features use only bursts 0..i-1 (no leakage):
the last K log burst lengths (most recent first, zero padded) with a fill
fraction as mask, their mean, variance, minimum and maximum, the exponential
average, the blocked time before burst i and the static priority. Lengths are
log transformed because bursts span several orders of magnitude.
"""

import numpy as np

K = 8
FEATURE_ALPHA = 0.5
TAU0 = 5.0
FEATURE_NAMES = ([f"lag{j + 1}" for j in range(K)] +
                 ["fill", "mean", "var", "min", "max", "ema", "blocked", "priority"])
N_FEATURES = len(FEATURE_NAMES)

_FLOOR = 1e-4  # ms; log values are shifted so the floor maps to 0


def to_log(x):
    return np.log10(np.maximum(x, _FLOOR)) + 4.0


def from_log(y):
    return np.power(10.0, np.asarray(y) - 4.0)


def feature_vector(history, tau, blocked, priority, out=None):
    """history: completed burst lengths (ms), oldest first; tau: EMA so far."""
    v = np.zeros(N_FEATURES) if out is None else out
    window = history[-K:]
    m = len(window)
    if m:
        lw = to_log(np.asarray(window[::-1], dtype=float))
        v[:m] = lw
        v[K + 1] = lw.mean()
        v[K + 2] = lw.var()
        v[K + 3] = lw.min()
        v[K + 4] = lw.max()
    v[K] = m / K
    v[K + 5] = to_log(tau)
    v[K + 6] = to_log(blocked + 1e-3)
    v[K + 7] = priority
    return v


def sequence_features(lengths, blocked, priority=0):
    """Feature rows for bursts 1..n-1 of one process; row r predicts lengths[r+1]."""
    n = len(lengths)
    X = np.zeros((max(n - 1, 0), N_FEATURES))
    tau = TAU0
    for i in range(1, n):
        tau = FEATURE_ALPHA * lengths[i - 1] + (1 - FEATURE_ALPHA) * tau
        feature_vector(lengths[:i], tau, blocked[i], priority, X[i - 1])
    return X


def build_dataset(procs, split=(0.6, 0.2, 0.2)):
    """Chronological per-process split into train / validation / test rows.

    Returns dict part -> (X, y_ms, meta) where meta rows are (pid, burst_index).
    """
    parts = {"train": ([], [], []), "val": ([], [], []), "test": ([], [], [])}
    for p in procs:
        lengths = [b[0] for b in p.bursts]
        blocked = [0.0] + [b[1] for b in p.bursts[:-1]]
        if len(lengths) < 2:
            continue
        X = sequence_features(lengths, blocked, p.priority)
        y = np.asarray(lengths[1:], dtype=float)
        n = len(y)
        a = int(round(n * split[0]))
        b = int(round(n * (split[0] + split[1])))
        for name, lo, hi in (("train", 0, a), ("val", a, b), ("test", b, n)):
            if hi > lo:
                parts[name][0].append(X[lo:hi])
                parts[name][1].append(y[lo:hi])
                parts[name][2].append(np.column_stack([np.full(hi - lo, p.pid), np.arange(lo, hi) + 1]))
    out = {}
    for name, (xs, ys, ms) in parts.items():
        if xs:
            out[name] = (np.vstack(xs), np.concatenate(ys), np.vstack(ms))
        else:
            out[name] = (np.zeros((0, N_FEATURES)), np.zeros(0), np.zeros((0, 2)))
    return out
