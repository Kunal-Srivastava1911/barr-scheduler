"""Dynamic quantum RR in oracle and predicted mode."""

import pytest

from barr.metrics import collect
from barr.sched import MeanRR, MedianRR, RoundRobin
from barr.sim import Process, simulate


def wl(bursts):
    return [Process(i + 1, 0.0, [(b, 0.0)]) for i, b in enumerate(bursts)]


def test_median_quantum_oracle():
    """Quanta follow the median of the remaining bursts in the ready queue."""
    res = simulate(MedianRR("oracle"), wl([10, 20, 30]), cs_cost=0.0)
    first = [(pid, end - start) for pid, _, _, start, end in res.slices][:3]
    assert first == [(1, 10), (2, 20), (3, 20)]           # q = median(10,20,30) = 20
    assert collect(res)["mean_wt"] == pytest.approx((0 + 10 + 30) / 3)


def test_mean_quantum_oracle():
    res = simulate(MeanRR("oracle"), wl([6, 12, 30]), cs_cost=0.0)
    assert res.slices[0][4] - res.slices[0][3] == pytest.approx(6)  # q = 16, burst 6 done
    assert res.slices[1][4] - res.slices[1][3] == pytest.approx(12)  # q = (12 + 30)/2 = 21


def test_reorder_serves_shortest_first():
    res = simulate(MedianRR("oracle", reorder=True), wl([30, 5, 12]), cs_cost=0.0)
    assert [s[0] for s in res.slices][:2] == [2, 3]


def test_predicted_mode_degrades_without_oracle():
    """Once a burst exceeds its prediction its estimate falls to the floor and
    drags the median quantum down, multiplying context switches."""
    work = wl([40, 45, 50, 55])
    oracle = collect(simulate(MedianRR("oracle"), work, 0.1))
    pred = collect(simulate(MedianRR("pred"), work, 0.1))
    rr = collect(simulate(RoundRobin(5), work, 0.1))
    assert pred["n_cs"] > 3 * oracle["n_cs"]
    assert pred["n_cs"] > rr["n_cs"]
