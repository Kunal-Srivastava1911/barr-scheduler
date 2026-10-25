#!/usr/bin/env bash
# Record scheduler events with ftrace (tracefs) while one workload command runs.
#
#   sudo ./collect.sh <name> <seconds> <command...>
#
# Writes out/<name>.trace (ftrace text) and out/<name>.root (workload root pid).
# Run on bare metal with a fixed CPU frequency where possible: hypervisor
# scheduling adds noise to the measured run intervals.
set -euo pipefail

NAME=$1; SECS=$2; shift 2
T=/sys/kernel/tracing
[ -d "$T/events" ] || T=/sys/kernel/debug/tracing
OUT=${OUT:-out}
mkdir -p "$OUT"

echo 0 > "$T/tracing_on"
echo > "$T/trace"
echo 65536 > "$T/buffer_size_kb"
echo nop > "$T/current_tracer"
for ev in sched_switch sched_wakeup sched_wakeup_new sched_process_fork; do
  echo 1 > "$T/events/sched/$ev/enable"
done

echo 1 > "$T/tracing_on"
setsid "$@" &
ROOT=$!
echo "$ROOT" > "$OUT/$NAME.root"
sleep "$SECS"
kill -TERM -- -"$ROOT" 2>/dev/null || true
wait "$ROOT" 2>/dev/null || true
echo 0 > "$T/tracing_on"

cat "$T/trace" > "$OUT/$NAME.trace"
for ev in sched_switch sched_wakeup sched_wakeup_new sched_process_fork; do
  echo 0 > "$T/events/sched/$ev/enable"
done
echo "trace: $OUT/$NAME.trace  root pid: $ROOT"
