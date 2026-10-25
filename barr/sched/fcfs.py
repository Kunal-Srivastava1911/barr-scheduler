from .base import FifoScheduler, INF


class FCFS(FifoScheduler):
    name = "FCFS"

    def pick_next(self, t):
        if not self.queue:
            return None, None
        return self.queue.popleft(), INF
