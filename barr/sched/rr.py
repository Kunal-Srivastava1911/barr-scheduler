from .base import FifoScheduler


class RoundRobin(FifoScheduler):
    def __init__(self, quantum):
        super().__init__()
        self.q = float(quantum)
        self.name = f"RR q={quantum:g}"

    def pick_next(self, t):
        if not self.queue:
            return None, None
        return self.queue.popleft(), self.q


def rr_loop(workload, quantum):
    """Independent fixed-quantum RR written as a plain loop without the event
    engine (no context switch cost, no I/O). Used to cross-check the engine.
    Returns {pid: completion time}."""
    procs = sorted(workload, key=lambda p: (p.arrival, p.pid))
    rem = {p.pid: p.bursts[0][0] for p in procs}
    t, i, queue, done = 0.0, 0, [], {}
    while len(done) < len(procs):
        while i < len(procs) and procs[i].arrival <= t + 1e-9:
            queue.append(procs[i].pid)
            i += 1
        if not queue:
            t = procs[i].arrival
            continue
        pid = queue.pop(0)
        run = min(quantum, rem[pid])
        t += run
        rem[pid] -= run
        while i < len(procs) and procs[i].arrival <= t + 1e-9:
            queue.append(procs[i].pid)
            i += 1
        if rem[pid] > 1e-9:
            queue.append(pid)
        else:
            done[pid] = t
    return done
