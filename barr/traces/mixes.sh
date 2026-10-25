#!/usr/bin/env bash
# Five workload mixes, 40 s each, three runs each, then burst extraction.
#   sudo ./mixes.sh            (needs: make gcc gzip xz ffmpeg ab python3 curl)
# Output: data/linux/<mix>.csv in the common burst schema.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT"
export OUT="$ROOT/data/raw"
mkdir -p "$OUT" "$ROOT/data/linux" /tmp/barr
SECS=${SECS:-40}

interactive() { for i in $(seq 1 400); do ls -la /usr/bin >/dev/null; grep -c a /etc/services >/dev/null; sleep 0.$((RANDOM % 5)); done; }
compile()     { cd /tmp/barr; [ -d lua ] || (curl -sL https://www.lua.org/ftp/lua-5.4.6.tar.gz | tar xz && mv lua-5.4.6 lua);
                cd lua; while true; do make clean >/dev/null; make -j"$(nproc)" >/dev/null 2>&1; done; }
compress()    { head -c 200M /dev/urandom > /tmp/barr/blob;
                while true; do gzip -c /tmp/barr/blob >/dev/null; xz -T1 -c -1 /tmp/barr/blob | head -c 1 >/dev/null;
                ffmpeg -loglevel quiet -f lavfi -i testsrc=duration=5:size=640x360 -c:v libx264 -f null -; done; }
http()        { (cd /tmp/barr && python3 -m http.server 8099 >/dev/null 2>&1 &); sleep 1;
                while true; do ab -q -n 2000 -c 8 http://127.0.0.1:8099/ >/dev/null 2>&1; done; }
mixed()       { interactive & compile & compress & http & wait; }
export -f interactive compile compress http mixed

for mix in interactive compile compress http mixed; do
  for run in 1 2 3; do
    "$HERE/collect.sh" "$mix-$run" "$SECS" bash -c "$mix"
    python3 -m barr.traces.parse_ftrace "$OUT/$mix-$run.trace" --root "$(cat "$OUT/$mix-$run.root")" \
      --source "$mix" -o "$OUT/$mix-$run.csv"
  done
  (head -n1 "$OUT/$mix-1.csv"; for run in 1 2 3; do tail -n +2 "$OUT/$mix-$run.csv"; done) > "$ROOT/data/linux/$mix.csv"
done
pkill -f "http.server 8099" || true
echo "done: $ROOT/data/linux"
