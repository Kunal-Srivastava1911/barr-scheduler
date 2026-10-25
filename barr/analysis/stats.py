"""Statistical treatment: bootstrap confidence intervals, Wilcoxon signed-rank
test (exact null distribution) and geometric means of ratios."""

import math

import numpy as np


def bootstrap_ci(values, n_boot=2000, level=0.95, seed=0):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(v, (n_boot, len(v)), replace=True).mean(1)
    a = (1 - level) / 2
    return float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def geomean(values):
    v = np.asarray([x for x in values if np.isfinite(x) and x > 0], dtype=float)
    return float(np.exp(np.log(v).mean())) if len(v) else float("nan")


def wilcoxon(x, y):
    """Two-sided Wilcoxon signed-rank test on paired samples.

    Zero differences are dropped; ties get average ranks. The p value comes
    from the exact permutation distribution of the rank sum (doubled ranks
    keep it in integers). Returns (W+, p, n).
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n == 0:
        return 0.0, 1.0, 0
    order = np.argsort(np.abs(d), kind="mergesort")
    ranks = np.empty(n)
    absd = np.abs(d)[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1] - absd[i] < 1e-12:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r2 = np.rint(ranks * 2).astype(int)
    w_plus2 = int(r2[d > 0].sum())
    total = int(r2.sum())
    # distribution of the sum of a random subset of r2
    dist = np.zeros(total + 1)
    dist[0] = 1.0
    for r in r2:          # doubled ranks are >= 2
        new = dist.copy()
        new[r:] += dist[:-r]
        dist = new
    dist /= 2.0 ** n
    lo = min(w_plus2, total - w_plus2)
    p = min(1.0, 2 * dist[:lo + 1].sum())
    return w_plus2 / 2.0, float(p), n


def pct_change(new, ref):
    return (new / ref - 1.0) * 100.0 if ref else float("nan")


def isfinite(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))
