"""Textbook checks: worked examples of Silberschatz et al., Operating System
Concepts, ch. 5 (no context switch cost)."""

import pytest

from barr.metrics import collect
from barr.predictors import OraclePredictor
from barr.sched import FCFS, SJF, SRTF, RoundRobin
from barr.sim import Process, simulate


def wl(spec):
    return [Process(i + 1, a, [(b, 0.0)]) for i, (a, b) in enumerate(spec)]


@pytest.mark.parametrize("sched, spec, expected", [
    (FCFS(), [(0, 24), (0, 3), (0, 3)], 17.0),
    (SJF(OraclePredictor()), [(0, 6), (0, 8), (0, 7), (0, 3)], 7.0),
    (SRTF(), [(0, 8), (1, 4), (2, 9), (3, 5)], 6.5),
    (RoundRobin(4), [(0, 24), (0, 3), (0, 3)], 17 / 3),
])
def test_average_waiting_time(sched, spec, expected):
    m = collect(simulate(sched, wl(spec), cs_cost=0.0))
    assert m["mean_wt"] == pytest.approx(expected)


def test_rr_gantt_chart():
    res = simulate(RoundRobin(4), wl([(0, 24), (0, 3), (0, 3)]), cs_cost=0.0)
    order = [(pid, start, end) for pid, _, _, start, end in res.slices]
    assert order[:3] == [(1, 0, 4), (2, 4, 7), (3, 7, 10)]
    assert order[-1] == (1, 26, 30)
