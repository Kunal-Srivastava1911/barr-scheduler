"""Burst parser on a hand-built ftrace snippet, leakage check of the feature
builder, predictor sanity and statistics."""

import numpy as np
import pytest

from barr.analysis import wilcoxon, bootstrap_ci
from barr.features import build_dataset, sequence_features, to_log
from barr.metrics import jain_index
from barr.predictors import GBRT
from barr.sim import Process
from barr.traces.parse_ftrace import parse_lines, to_rows

SNIPPET = """\
# tracer: nop
            bash-100   [000] d..2.  10.000000: sched_process_fork: comm=bash pid=100 child_comm=bash child_pid=101
          <idle>-0     [000] d..2.  10.001000: sched_switch: prev_comm=swapper/0 prev_pid=0 prev_prio=120 prev_state=R ==> next_comm=work next_pid=101 next_prio=120
            work-101   [000] d..2.  10.004000: sched_switch: prev_comm=work prev_pid=101 prev_prio=120 prev_state=R ==> next_comm=kworker/0:1 next_pid=7 next_prio=120
     kworker/0:1-7     [000] d..2.  10.005000: sched_switch: prev_comm=kworker/0:1 prev_pid=7 prev_prio=120 prev_state=I ==> next_comm=work next_pid=101 next_prio=120
            work-101   [000] d..2.  10.007000: sched_switch: prev_comm=work prev_pid=101 prev_prio=120 prev_state=S ==> next_comm=swapper/0 next_pid=0 next_prio=120
          <idle>-0     [000] d..2.  10.017000: sched_wakeup: comm=work pid=101 prio=120 target_cpu=000
          <idle>-0     [000] d..2.  10.017500: sched_switch: prev_comm=swapper/0 prev_pid=0 prev_prio=120 prev_state=R ==> next_comm=work next_pid=101 next_prio=120
            work-101   [000] d..2.  10.018500: sched_switch: prev_comm=work prev_pid=101 prev_prio=120 prev_state=S ==> next_comm=other next_pid=555 next_prio=120
           other-555   [000] d..2.  10.019000: sched_switch: prev_comm=other prev_pid=555 prev_prio=120 prev_state=S ==> next_comm=swapper/0 next_pid=0 next_prio=120
"""


def test_parser_merges_preempted_intervals():
    threads = parse_lines(SNIPPET.splitlines(), root_pid=100)
    rows = to_rows(threads, "test")
    assert [r["pid"] for r in rows] == [101, 101]          # kworker and unrelated thread removed
    first, second = rows
    assert first["burst_length"] == pytest.approx(5.0)     # 3 ms + 2 ms merged across preemption
    assert first["preempted"] == 1
    assert second["burst_length"] == pytest.approx(1.0)
    assert second["blocked_before"] == pytest.approx(10.0)  # sleep 10.007 -> wakeup 10.017
    assert second["preempted"] == 0


def test_no_feature_uses_a_later_burst():
    rng = np.random.default_rng(0)
    lengths = list(rng.exponential(5, 30))
    blocked = list(rng.exponential(3, 30))
    X = sequence_features(lengths, blocked)
    for i in range(1, 30):
        changed = lengths[:i] + [x * 7 + 1 for x in lengths[i:]]
        changed_b = blocked[:i + 1] + [x * 3 for x in blocked[i + 1:]]
        X2 = sequence_features(changed, changed_b)
        assert np.array_equal(X[i - 1], X2[i - 1])           # row i-1 predicts burst i


def test_chronological_split():
    p = Process(1, 0, [(float(i + 1), 1.0) for i in range(51)])
    d = build_dataset([p])
    assert len(d["train"][1]) == 30 and len(d["val"][1]) == 10 and len(d["test"][1]) == 10
    assert d["train"][1].max() < d["val"][1].min() < d["test"][1].min()


def test_gbrt_learns_a_step_function():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (3000, 3))
    y = np.where(X[:, 0] > 0.5, 3.0, 1.0) + 0.05 * rng.standard_normal(3000)
    m = GBRT(n_trees=60, depth=3, loss="l2", backend="numpy").fit(X[:2000], y[:2000], X[2000:], y[2000:])
    assert np.mean(np.abs(m.predict(X[2000:]) - y[2000:])) < 0.1
    assert to_log(1.0) == pytest.approx(4.0)


def test_statistics():
    _, p, n = wilcoxon([2, 3, 4, 5, 6], [1, 1, 1, 1, 1])
    assert n == 5 and p == pytest.approx(2 / 32)            # exact two-sided p
    _, p, _ = wilcoxon([1, 2, 3, 4], [1.5, 1.0, 3.5, 3.0])
    assert p >= 0.5
    mean, lo, hi = bootstrap_ci([1, 2, 3, 4, 5])
    assert lo <= mean <= hi and mean == 3
    assert jain_index([2, 2, 2]) == pytest.approx(1.0)
    assert jain_index([1, 0, 0, 0]) == pytest.approx(0.25)
