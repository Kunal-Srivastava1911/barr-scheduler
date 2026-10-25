"""Event kinds. At one instant arrivals and I/O completions are handled before a
slice end, so a newly ready process queues ahead of the one being preempted
(the textbook round robin convention)."""

ARRIVAL = 0
IO_DONE = 1
SLICE_END = 2

NAMES = {ARRIVAL: "arrival", IO_DONE: "io_done", SLICE_END: "slice_end"}

EPS = 1e-9
