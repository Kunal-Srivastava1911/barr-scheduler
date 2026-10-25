"""Synthetic workload generators.

A process is an alternating sequence of CPU bursts and I/O waits. Arrivals form
a Poisson process whose rate is set so that the offered CPU load equals rho.
Each process has its own burst scale, so its history carries signal about its
next burst, as in real programs. Families:

    exp       exponential bursts around a per-process mean
    bimodal   short interactive processes and long compute processes
    pareto    heavy tailed (Pareto, alpha 1.5) bursts
    phase     the burst distribution switches part way through the process
"""

import numpy as np

from ..sim.process import Process

FAMILIES = ("exp", "bimodal", "pareto", "phase")

MIN_BURST = 0.01   # ms
MAX_BURST = 1000.0


def _clip(x):
    return float(min(MAX_BURST, max(MIN_BURST, x)))


def _process_bursts(family, rng, n):
    """Returns (cpu bursts, io waits, kind)."""
    if family == "exp":
        mean = rng.lognormal(np.log(4.0), 0.9)
        cpu = rng.exponential(mean, n)
        kind = "interactive" if mean < 2 else "compute" if mean > 10 else "mixed"
        io_mean = 8.0
    elif family == "bimodal":
        if rng.random() < 0.6:
            kind, med, io_mean = "interactive", 0.6, 15.0
        else:
            kind, med, io_mean = "compute", 25.0, 2.0
        cpu = rng.lognormal(np.log(med), 0.5, n)
        flip = rng.random(n) < 0.08                     # occasional burst of the other kind
        cpu[flip] = rng.lognormal(np.log(25.0 if kind == "interactive" else 0.6), 0.5, flip.sum())
    elif family == "pareto":
        xm = rng.lognormal(np.log(0.8), 0.7)
        cpu = (rng.pareto(1.5, n) + 1.0) * xm
        kind = "interactive" if xm < 0.5 else "compute" if xm > 1.6 else "mixed"
        io_mean = 6.0
    elif family == "phase":
        short, long_ = rng.lognormal(np.log(0.8), 0.4), rng.lognormal(np.log(20.0), 0.4)
        first, second = (short, long_) if rng.random() < 0.5 else (long_, short)
        cut = int(rng.integers(n // 3, max(n // 3 + 1, 2 * n // 3)))
        cpu = np.concatenate([rng.exponential(first, cut), rng.exponential(second, n - cut)])
        kind = "mixed"
        io_mean = 6.0
    else:
        raise ValueError(f"unknown workload family {family!r}")
    io = rng.exponential(io_mean * rng.lognormal(0, 0.4), n)
    return [_clip(c) for c in cpu], [float(i) for i in io], kind


def generate(family, n_procs=50, load=0.7, seed=0, bursts=(10, 40), start_pid=1):
    rng = np.random.default_rng([seed, FAMILIES.index(family) if family in FAMILIES else 99])
    procs = []
    for k in range(n_procs):
        n = int(rng.integers(bursts[0], bursts[1] + 1))
        cpu, io, kind = _process_bursts(family, rng, n)
        io[-1] = 0.0
        procs.append(Process(start_pid + k, 0.0, list(zip(cpu, io)), priority=0,
                             name=f"{family}-{start_pid + k}", kind=kind))
    total_cpu = sum(p.cpu_total for p in procs)
    horizon = total_cpu / load
    arrivals = np.sort(rng.uniform(0, horizon, n_procs))
    arrivals -= arrivals[0]
    for p, a in zip(procs, arrivals):
        p.arrival = float(a)
    return procs


def demo_workload():
    """Four process demo of the final review (Fig. 3): two compute bound
    processes and two interactive ones."""
    return [
        Process(0, 0.0, [(9.0, 2.0), (9.0, 0.0)], name="P0 compute", kind="compute"),
        Process(1, 0.5, [(8.0, 3.0), (8.0, 0.0)], name="P1 compute", kind="compute"),
        Process(2, 1.0, [(1.0, 4.0), (1.5, 4.0), (1.0, 0.0)], name="P2 interactive", kind="interactive"),
        Process(3, 2.0, [(1.5, 5.0), (1.0, 5.0), (1.5, 0.0)], name="P3 interactive", kind="interactive"),
    ]
