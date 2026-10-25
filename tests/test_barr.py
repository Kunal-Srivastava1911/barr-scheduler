"""BARR policy: quantum controller, the equivalence of Section IV-E and aging."""

import pytest

from barr.metrics import collect
from barr.predictors import EMAPredictor, OraclePredictor
from barr.sched import BARR, QuantumController, RoundRobin, make
from barr.sim import Process, simulate
from barr.workloads import FAMILIES, generate


def test_quantum_controller_bounds_and_growth():
    ctrl = QuantumController(delta=0.25, q_min=1.0, q_max=50.0, beta=2.0)
    p = Process(1, 0, [(100.0, 0.0)])
    p.pred, p.used = 4.0, 0.0
    assert ctrl.quantum(p) == pytest.approx(5.0)          # (1 + delta) * p
    p.pred = 0.1
    assert ctrl.quantum(p) == 1.0                         # q_min
    p.pred = 400.0
    assert ctrl.quantum(p) == 50.0                        # q_max
    p.pred, p.used = 4.0, 5.0                             # under-predicted, expired
    ctrl.on_expiry(p)
    assert p.pred == 10.0                                 # raised to beta * used


@pytest.mark.parametrize("family", FAMILIES)
def test_oracle_barr_equals_rr_with_qmax(family):
    """With exact predictions and delta >= 0, FIFO BARR is RR with q = q_max."""
    for seed in range(3):
        work = generate(family, n_procs=30, load=0.8, seed=seed)
        a = simulate(RoundRobin(50), work, 0.1)
        b = simulate(BARR(OraclePredictor(), q_max=50), work, 0.1)
        assert a.slices == b.slices
        assert collect(a) == pytest.approx(collect(b), nan_ok=True, rel=1e-12) or \
            collect(a)["mean_wt"] == collect(b)["mean_wt"]


def test_barr_s_aging_bounds_starvation():
    """A long process among a stream of short ones still gets the CPU."""
    work = [Process(0, 0.0, [(30.0, 0.0)])]
    work += [Process(i, 0.05 * i, [(0.5, 0.0)]) for i in range(1, 400)]
    res = simulate(BARR(OraclePredictor(), q_max=5, aging=16, order="spf"), work, 0.0)
    first = min(s[3] for s in res.slices if s[0] == 0)
    assert first < 20.0


def test_barr_records_predictions():
    work = generate("bimodal", n_procs=20, load=0.7, seed=5)
    m = collect(simulate(make("BARR-EMA"), work, 0.1))
    assert m["pred_mae"] > 0 and 0 <= m["pred_under"] <= 1
    ema = EMAPredictor(alpha=0.5, tau0=10)
    p = Process(1, 0, [(1, 0)])
    ema.update(p, 2.0)
    assert ema.predict(p) == pytest.approx(6.0)           # 0.5 * 2 + 0.5 * 10
