"""Process model: a process is a list of (CPU burst, I/O wait) pairs."""

from dataclasses import dataclass, field


@dataclass
class Process:
    pid: int
    arrival: float
    bursts: list                 # [(cpu_ms, io_ms_after), ...]; last io is ignored
    priority: int = 0
    name: str = ""
    kind: str = ""               # workload class label, e.g. "interactive"

    # run-time state, owned by the engine and the schedulers
    idx: int = 0                 # index of the current burst
    remaining: float = 0.0       # CPU still needed by the current burst
    used: float = 0.0            # CPU already used by the current burst
    blocked: float = 0.0         # blocked time before the current burst
    ready_at: float = 0.0        # time the current burst became ready
    dispatched: bool = False     # current burst has been dispatched at least once
    preempted: bool = False      # current burst has been preempted at least once
    pred: float = 0.0            # predicted length of the current burst
    pred0: float = 0.0           # prediction at first dispatch (before any raise)
    first_dispatch: float = 0.0  # start of the first slice of the current burst
    first_quantum: float = 0.0   # quantum of the first dispatch of the current burst
    enq_seq: int = 0             # dispatch counter when the process was queued (aging)
    order: int = 0               # FIFO tie breaker
    history: list = field(default_factory=list)   # completed burst lengths
    completion: float = None

    @property
    def burst(self):
        """True length of the current CPU burst (oracle information)."""
        return self.bursts[self.idx][0]

    @property
    def cpu_total(self):
        return sum(b[0] for b in self.bursts)

    @property
    def io_total(self):
        return sum(b[1] for b in self.bursts[:-1])


def clone_workload(procs):
    """Fresh copies with clean run-time state, so one workload can feed many runs."""
    return [Process(p.pid, p.arrival, [tuple(b) for b in p.bursts], p.priority, p.name, p.kind)
            for p in procs]
