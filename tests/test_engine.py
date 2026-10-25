"""Invariants on random workloads (seeded property tests) and the cross-check
of the event driven RR against an independent loop."""

import random

import pytest

from barr.metrics import collect
from barr.sched import ALL, make, rr_loop, RoundRobin
from barr.sim import Process, simulate
from barr.workloads import FAMILIES, generate

CASES = 25


def random_single_burst(rng, n):
    return [Process(i + 1, round(rng.uniform(0, 30), 2), [(round(rng.uniform(0.5, 20), 2), 0.0)])
            for i in range(n)]


@pytest.mark.parametrize("seed", range(CASES))
def test_rr_matches_independent_loop(seed):
    rng = random.Random(seed)
    work = random_single_burst(rng, rng.randint(1, 12))
    q = rng.choice([1, 2, 3.5, 5, 10])
    res = simulate(RoundRobin(q), work, cs_cost=0.0)
    ref = rr_loop(work, q)
    for p in res.procs:
        assert p.completion == pytest.approx(ref[p.pid])


@pytest.mark.parametrize("name", ALL)
def test_invariants_every_scheduler(name, trained):
    for seed in range(4):
        fam = FAMILIES[seed % len(FAMILIES)]
        work = generate(fam, n_procs=15, load=0.8, seed=seed, bursts=(3, 10))
        res = simulate(make(name, trained), work, cs_cost=0.1)
        # every process completes
        assert all(p.completion is not None for p in res.procs)
        # total service equals the input bursts
        served = {}
        for pid, _, cs_start, start, end in res.slices:
            assert 0 <= cs_start <= start <= end
            served[pid] = served.get(pid, 0.0) + (end - start)
        for p in work:
            assert served[p.pid] == pytest.approx(p.cpu_total, rel=1e-9, abs=1e-6)
        # the CPU never runs two processes at once
        sl = sorted(res.slices, key=lambda s: s[2])
        for a, b in zip(sl, sl[1:]):
            assert b[2] >= a[4] - 1e-9
        # no negative waiting or response time
        m = collect(res)
        assert m["mean_wt"] >= -1e-6 and m["mean_resp"] >= -1e-9
