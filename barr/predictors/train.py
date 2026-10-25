"""Train and evaluate burst predictors offline.

Data: synthetic workloads (and Linux bursts when a parsed trace data set is
present), split chronologically per process into 60 % train, 20 % validation
and 20 % test. Hyperparameters are tuned on the validation part only; the test
part is used once for the reported numbers.

    python -m barr.predictors.train            # train, evaluate, save models/
"""

import os
import pickle
import time

import numpy as np

from ..features.build import build_dataset, to_log, from_log
from ..sim.process import Process
from ..traces.load_linux import available_mixes, load_mix
from ..workloads.synthetic import FAMILIES, generate
from . import lstm as lstm_mod
from .base import LastValuePredictor
from .ema import EMAPredictor
from .gbrt import GBRT
from .learned import KNN, LearnedPredictor, Ridge

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT, "models", "predictors.pkl")
TRAIN_SEED = 1000   # training workloads never share seeds with evaluation workloads


def training_procs(families=FAMILIES, n_procs=150, seed=TRAIN_SEED, bursts=(40, 120),
                   include_linux=True):
    """Synthetic processes per family, plus parsed Linux bursts from data/linux
    when present (collected with barr/traces/mixes.sh)."""
    procs, pid = [], 1
    for f in families:
        ps = generate(f, n_procs=n_procs, load=0.7, seed=seed, bursts=bursts, start_pid=pid)
        pid += n_procs
        procs += ps
    if include_linux:
        for path in available_mixes():
            ps = [p for p in load_mix(path) if len(p.bursts) >= 5]
            for k, p in enumerate(ps):
                p.pid = pid + k
                p.name = f"linux:{p.name}"
            pid += len(ps)
            procs += ps
    return procs


def _scores(pred, y):
    return {
        "mae": float(np.mean(np.abs(pred - y))),
        "log_err": float(np.mean(np.abs(np.log(np.maximum(pred, 1e-4)) - np.log(np.maximum(y, 1e-4))))),
        "under": float(np.mean(pred < y)),
    }


def _replay(predictor, procs, part, split=(0.6, 0.2, 0.2)):
    """Run an online predictor over every burst sequence; score one part."""
    preds, ys = [], []
    for p in procs:
        q = Process(p.pid, 0, p.bursts, p.priority)
        n = len(p.bursts) - 1
        a, b = int(round(n * split[0])), int(round(n * (split[0] + split[1])))
        lo, hi = {"train": (0, a), "val": (a, b), "test": (b, n)}[part]
        for i, (cpu, io) in enumerate(p.bursts):
            q.idx = i
            if 1 <= i and lo <= i - 1 < hi:
                preds.append(predictor.predict(q))
                ys.append(cpu)
            predictor.update(q, cpu)
            q.history.append(cpu)
            q.blocked = io
    return np.array(preds), np.array(ys)


def latency_us(predictor, procs, n=2000):
    """Median and 99th percentile wall time of one predict() call, in µs."""
    samples = []
    pool = []
    for p in procs[:50]:
        q = Process(p.pid, 0, p.bursts, p.priority)
        q.history = [b[0] for b in p.bursts[:20]]
        q.blocked = p.bursts[19][1] if len(p.bursts) > 20 else 1.0
        pool.append(q)
    for i in range(n):
        q = pool[i % len(pool)]
        t0 = time.perf_counter_ns()
        predictor.predict(q)
        samples.append((time.perf_counter_ns() - t0) / 1000.0)
    s = np.array(samples)
    return float(np.median(s)), float(np.percentile(s, 99))


def train_all(procs=None, log=print, with_lstm=True, per_family=True):
    """Returns (predictors dict, report dict)."""
    procs = procs if procs is not None else training_procs()
    t0 = time.time()
    data = build_dataset(procs)
    Xtr, ytr, _ = data["train"]
    Xva, yva, _ = data["val"]
    Xte, yte, mte = data["test"]
    ltr, lva = to_log(ytr), to_log(yva)
    log(f"data set: {len(ytr)} train / {len(yva)} val / {len(yte)} test bursts "
        f"from {len(procs)} processes")

    report = {"n_train": len(ytr), "n_val": len(yva), "n_test": len(yte), "models": {}}
    preds = {}

    # exponential average: alpha chosen on validation
    best = None
    for a in (0.25, 0.5, 0.75):
        pv, yv = _replay(EMAPredictor(alpha=a), procs, "val")
        mae = float(np.mean(np.abs(pv - yv)))
        log(f"  EMA alpha={a}: val MAE {mae:.3f} ms")
        if best is None or mae < best[1]:
            best = (a, mae)
    ema = EMAPredictor(alpha=best[0])
    preds["ema"] = ema
    report["ema_alpha"] = best[0]
    preds["last"] = LastValuePredictor()

    # tabular models on the feature vector
    log("  fitting ridge ...")
    preds["linear"] = LearnedPredictor(Ridge().fit(Xtr, ltr), "linear")
    log("  fitting kNN (k=10) ...")
    preds["knn"] = LearnedPredictor(KNN().fit(Xtr, ltr), "knn")

    best = None
    for loss in ("l1", "l2", "quantile"):
        t1 = time.time()
        m = GBRT(loss=loss).fit(Xtr, ltr, Xva, lva)
        mae = float(np.mean(np.abs(from_log(m.predict(Xva)) - yva)))
        log(f"  GBRT loss={loss}: {m.n_used} trees, val MAE {mae:.3f} ms ({time.time() - t1:.1f} s)")
        if best is None or mae < best[1]:
            best = (m, mae, loss)
    preds["gbrt"] = LearnedPredictor(best[0], "gbrt")
    report["gbrt_loss"] = best[2]
    report["gbrt_backend"] = best[0].backend

    if with_lstm and lstm_mod.available():
        log("  training LSTM (32 hidden units, 16 bursts) ...")
        seq = lstm_mod.sequences(procs)
        model = lstm_mod.LSTMModel().fit(*seq["train"], *seq["val"])
        preds["lstm"] = lstm_mod.LSTMPredictor(model)
    elif with_lstm:
        log("  LSTM skipped: PyTorch is not installed (pip install torch)")

    # test scores: online replay over the test part for every predictor
    fam_of = {p.pid: p.name.split("-")[0] for p in procs}
    for name, pr in preds.items():
        pt, yt = _replay(pr.fresh(), procs, "test")
        sc = _scores(pt, yt)
        sc["lat_med_us"], sc["lat_p99_us"] = latency_us(pr.fresh(), procs)
        if per_family:
            sc["family"] = {}
            for fam in sorted(set(fam_of.values())):
                sub = [p for p in procs if fam_of[p.pid] == fam]
                pf, yf = _replay(pr.fresh(), sub, "test")
                sc["family"][fam] = _scores(pf, yf)
        report["models"][name] = sc
        log(f"  test {name:7s} MAE {sc['mae']:.3f} ms  log err {sc['log_err']:.3f}  "
            f"under {sc['under']:.2f}  latency {sc['lat_med_us']:.1f} us")
    ref = report["models"]["ema"]["mae"]
    for name, sc in report["models"].items():
        sc["mae_rel_ema"] = sc["mae"] / ref
    report["seconds"] = time.time() - t0
    return preds, report


def save(preds, report, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"predictors": preds, "report": report}, f)


def load(path=MODEL_PATH):
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
        return d["predictors"], d["report"]
    except Exception:
        return None, None


def get_or_train(log=print):
    preds, report = load()
    if preds is None:
        log("no trained predictors found; training on synthetic workloads ...")
        preds, report = train_all(log=log)
        save(preds, report)
    return preds, report


if __name__ == "__main__":
    p, r = train_all()
    save(p, r)
    print(f"saved to {MODEL_PATH} ({r['seconds']:.1f} s)")
