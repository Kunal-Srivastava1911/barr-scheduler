"""Gradient boosted regression trees on the log burst length.

Uses XGBoost when it is installed (backend="xgboost"); otherwise a compact
histogram based implementation in NumPy with the same settings: shallow trees
(depth 4 to 6), at most 200 trees, squared, absolute or quantile loss, early
stopping on the validation part. The quantile loss with tau > 0.5 penalises
under-prediction more heavily, since an under-predicted burst costs a switch.
"""

import numpy as np

try:
    import xgboost as _xgb
except ImportError:  # pragma: no cover - optional dependency
    _xgb = None


def _loss_grad(loss, tau, pred, y):
    r = y - pred
    if loss == "l2":
        return pred - y
    q = 0.5 if loss == "l1" else tau
    return np.where(r > 0, -q, 1.0 - q)


def _leaf_value(loss, tau, resid, g, l2):
    if loss == "l2":
        return resid.mean() * len(resid) / (len(resid) + l2)
    q = 0.5 if loss == "l1" else tau
    return float(np.quantile(resid, q))


def _loss_value(loss, tau, pred, y):
    r = y - pred
    if loss == "l2":
        return float(np.mean(r * r))
    q = 0.5 if loss == "l1" else tau
    return float(np.mean(np.maximum(q * r, (q - 1) * r)))


class GBRT:
    def __init__(self, n_trees=200, depth=5, lr=0.1, loss="l1", tau=0.6, n_bins=32,
                 min_leaf=20, l2=1.0, subsample=0.8, patience=20, seed=0, backend="auto"):
        self.n_trees, self.depth, self.lr, self.loss, self.tau = n_trees, depth, lr, loss, tau
        self.n_bins, self.min_leaf, self.l2, self.subsample = n_bins, min_leaf, l2, subsample
        self.patience, self.seed = patience, seed
        self.backend = ("xgboost" if _xgb is not None else "numpy") if backend == "auto" else backend

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, X_val=None, y_val=None):
        if self.backend == "xgboost":
            return self._fit_xgb(X, y, X_val, y_val)
        rng = np.random.default_rng(self.seed)
        n, f = X.shape
        self.edges = []
        Xb = np.empty((n, f), dtype=np.int32)
        for j in range(f):
            e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.n_bins + 1)[1:-1]))
            self.edges.append(e)
            Xb[:, j] = np.searchsorted(e, X[:, j], side="left")
        self.base = float(np.median(y)) if self.loss != "l2" else float(y.mean())
        pred = np.full(n, self.base)
        val_pred = None if X_val is None or len(X_val) == 0 else np.full(len(X_val), self.base)
        n_nodes = 2 ** (self.depth + 1) - 1
        feats, thrs, vals = [], [], []
        best, best_iter, since = np.inf, 0, 0
        for it in range(self.n_trees):
            rows = np.flatnonzero(rng.random(n) < self.subsample) if self.subsample < 1 else np.arange(n)
            g = _loss_grad(self.loss, self.tau, pred[rows], y[rows])
            feat, thr, val = self._build_tree(Xb[rows], g, y[rows] - pred[rows], n_nodes)
            feats.append(feat), thrs.append(thr), vals.append(val)
            pred += self.lr * self._predict_tree(X, feat, thr, val)
            if val_pred is not None:
                val_pred += self.lr * self._predict_tree(X_val, feat, thr, val)
                v = _loss_value(self.loss, self.tau, val_pred, y_val)
                if v < best - 1e-6:
                    best, best_iter, since = v, it + 1, 0
                else:
                    since += 1
                    if since >= self.patience:
                        break
            else:
                best_iter = it + 1
        k = max(best_iter, 1)
        self.feat = np.array(feats[:k])
        self.thr = np.array(thrs[:k])
        self.val = np.array(vals[:k])
        self.n_used = k
        return self

    def _build_tree(self, Xb, g, resid, n_nodes):
        feat = np.full(n_nodes, -1, dtype=np.int32)
        thr = np.zeros(n_nodes)
        val = np.zeros(n_nodes)
        node = np.zeros(len(g), dtype=np.int64)
        nb = self.n_bins + 1
        first_leaf = 2 ** self.depth - 1
        active = [0]
        for _ in range(self.depth):
            if not active:
                break
            local = {nid: i for i, nid in enumerate(active)}
            lmap = np.full(n_nodes, -1, dtype=np.int64)
            for nid, i in local.items():
                lmap[nid] = i
            li = lmap[node]
            mask = li >= 0
            li_m, g_m = li[mask], g[mask]
            m = len(active)
            G = np.bincount(li_m, weights=g_m, minlength=m)
            H = np.bincount(li_m, minlength=m).astype(float)
            parent = G * G / (H + self.l2)
            best_gain = np.zeros(m)
            best_f = np.full(m, -1)
            best_b = np.zeros(m, dtype=np.int64)
            for j in range(Xb.shape[1]):
                key = li_m * nb + Xb[mask, j]
                Gh = np.bincount(key, weights=g_m, minlength=m * nb).reshape(m, nb).cumsum(1)
                Hh = np.bincount(key, minlength=m * nb).reshape(m, nb).cumsum(1).astype(float)
                GR, HR = G[:, None] - Gh, H[:, None] - Hh
                gain = Gh * Gh / (Hh + self.l2) + GR * GR / (HR + self.l2) - parent[:, None]
                gain[(Hh < self.min_leaf) | (HR < self.min_leaf)] = -np.inf
                gain[:, len(self.edges[j]):] = -np.inf   # last bin has no upper edge
                b = gain.argmax(1)
                gb = gain[np.arange(m), b]
                better = gb > best_gain + 1e-12
                best_gain[better], best_f[better], best_b[better] = gb[better], j, b[better]
            nxt = []
            for nid, i in local.items():
                if best_f[i] < 0:
                    continue
                j, b = best_f[i], best_b[i]
                feat[nid] = j
                thr[nid] = self.edges[j][b]
                rows = node == nid
                go_right = rows & (Xb[:, j] > b)
                node[rows] = 2 * nid + 1
                node[go_right] = 2 * nid + 2
                nxt += [2 * nid + 1, 2 * nid + 2]
            active = [a for a in nxt if a < n_nodes]
        for nid in np.unique(node):
            rows = node == nid
            val[nid] = _leaf_value(self.loss, self.tau, resid[rows], g[rows], self.l2)
        return feat, thr, val

    @staticmethod
    def _predict_tree(X, feat, thr, val):
        idx = np.zeros(len(X), dtype=np.int64)
        r = np.arange(len(X))
        while True:
            f = feat[idx]
            inner = f >= 0
            if not inner.any():
                break
            right = X[r, np.maximum(f, 0)] > thr[idx]
            idx = np.where(inner, 2 * idx + 1 + right, idx)
        return val[idx]

    # ------------------------------------------------------------------ predict
    def predict(self, X):
        if self.backend == "xgboost":
            return self.model.predict(X)
        T = len(self.feat)
        out = np.full(len(X), self.base)
        if T == 0:
            return out
        tr = np.arange(T)[:, None]
        r = np.arange(len(X))[None, :]
        idx = np.zeros((T, len(X)), dtype=np.int64)
        for _ in range(self.depth):
            f = self.feat[tr, idx]
            inner = f >= 0
            right = X[r, np.maximum(f, 0)] > self.thr[tr, idx]
            idx = np.where(inner, 2 * idx + 1 + right, idx)
        return out + self.lr * self.val[tr, idx].sum(0)

    def _fit_xgb(self, X, y, X_val, y_val):  # pragma: no cover - optional dependency
        obj = {"l2": "reg:squarederror", "l1": "reg:absoluteerror",
               "quantile": "reg:quantileerror"}[self.loss]
        kw = {"quantile_alpha": self.tau} if self.loss == "quantile" else {}
        self.model = _xgb.XGBRegressor(n_estimators=self.n_trees, max_depth=self.depth,
                                       learning_rate=self.lr, objective=obj, subsample=self.subsample,
                                       random_state=self.seed, early_stopping_rounds=self.patience
                                       if X_val is not None else None, **kw)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        self.n_used = getattr(self.model, "best_iteration", self.n_trees)
        return self
