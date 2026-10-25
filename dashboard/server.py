"""Live dashboard server (standard library only).

    python -m dashboard            # http://127.0.0.1:8765

Endpoints
    GET  /api/meta                 workloads, schedulers, defaults, predictor status
    POST /api/simulate             run schedulers on one workload, return the logs
    POST /api/jobs                 start a background job: train | experiment | tests
    GET  /api/jobs/<id>/events     Server-Sent Events stream of a job
    GET  /api/results/<name>       summary of a finished experiment
    GET  /results/<name>/runs.csv  raw per-run metrics
"""

import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from barr.experiments import run as exp_run                         # noqa: E402
from barr.metrics.collector import collect, METRIC_INFO               # noqa: E402
from barr.predictors import train as train_mod                        # noqa: E402
from barr.predictors import lstm as lstm_mod                          # noqa: E402
from barr.sched.registry import ALL, ORACLE, DEFAULT_BARR, DEFAULT_BARR_S, make, label  # noqa: E402
from barr.sim.engine import simulate                                  # noqa: E402
from barr.traces.load_linux import available_mixes                   # noqa: E402
from barr.workloads.synthetic import FAMILIES, generate, demo_workload  # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
CONFIGS = os.path.join(ROOT, "barr", "experiments", "configs")
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".csv": "text/csv; charset=utf-8",
         ".json": "application/json", ".svg": "image/svg+xml"}

STATE = {"trained": None, "report": None, "train_job": None}
JOBS = {}
LOCK = threading.Lock()


def r4(x):
    return round(float(x), 4)


# ---------------------------------------------------------------- background jobs
class Job:
    def __init__(self, kind):
        self.id = uuid.uuid4().hex[:10]
        self.kind = kind
        self.events = []
        self.done = False
        self.cond = threading.Condition()
        self.started = time.time()

    def emit(self, type_, **data):
        with self.cond:
            self.events.append({"type": type_, "t": round(time.time() - self.started, 2), **data})
            if type_ in ("done", "error"):
                self.done = True
            self.cond.notify_all()

    def log(self, msg):
        self.emit("log", msg=str(msg))


def start_job(kind, params):
    job = Job(kind)
    JOBS[job.id] = job
    target = {"train": _train_job, "experiment": _experiment_job, "tests": _tests_job}[kind]

    def runner():
        try:
            target(job, params or {})
        except Exception as e:  # report, never kill the server
            job.log(traceback.format_exc())
            job.emit("error", msg=f"{type(e).__name__}: {e}")

    threading.Thread(target=runner, daemon=True).start()
    return job


def _train_job(job, params):
    n = int(params.get("n_procs", 150))
    job.log(f"generating training workloads: {n} processes per family ...")
    procs = train_mod.training_procs(n_procs=n)
    preds, report = train_mod.train_all(procs, log=job.log, with_lstm=params.get("lstm", True))
    train_mod.save(preds, report)
    with LOCK:
        STATE["trained"], STATE["report"] = preds, report
    job.emit("done", report=exp_run.clean(report))


def _experiment_job(job, cfg):
    trained = _ensure_trained(job.log)
    total = {"n": 0}

    def progress(done, n, row):
        total["n"] = n
        job.emit("progress", done=done, total=n,
                 row={k: (r4(v) if isinstance(v, float) and v == v else v) for k, v in row.items()
                      if k in ("workload", "group", "scheduler", "mean_wt", "n_cs", "p95_resp", "max_wt")})

    _, summary = exp_run.run(cfg, log=job.log, progress=progress, trained=trained)
    job.emit("done", summary=summary, name=cfg.get("name", "experiment"))


def _tests_job(job, params):
    cmd = [sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", "tests"]
    job.log("$ python -m pytest -v tests")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        job.log(line.rstrip("\n"))
    rc = proc.wait()
    job.emit("done", returncode=rc)


def _ensure_trained(log=print):
    with LOCK:
        if STATE["trained"] is not None:
            return STATE["trained"]
    preds, report = train_mod.get_or_train(log=log)
    with LOCK:
        STATE["trained"], STATE["report"] = preds, report
    return preds


# ---------------------------------------------------------------- API
def meta():
    rep = STATE["report"]
    configs = {}
    for f in sorted(os.listdir(CONFIGS)):
        if f.endswith(".json"):
            with open(os.path.join(CONFIGS, f)) as fh:
                configs[f[:-5]] = json.load(fh)
    results = sorted(d for d in os.listdir(exp_run.RESULTS)
                     if os.path.exists(os.path.join(exp_run.RESULTS, d, "summary.json"))) \
        if os.path.isdir(exp_run.RESULTS) else []
    tj = STATE["train_job"]
    return {
        "families": list(FAMILIES) + ["demo"],
        "schedulers": [{"name": s, "label": label(s), "oracle": s in ORACLE} for s in ALL],
        "barr": DEFAULT_BARR, "barr_s": DEFAULT_BARR_S,
        "metrics": {k: {"label": v[0], "unit": v[1], "better": v[2]} for k, v in METRIC_INFO.items()},
        "trained": STATE["trained"] is not None,
        "train_job": tj.id if tj is not None and not tj.done else None,
        "report": exp_run.clean(rep) if rep else None,
        "lstm_available": lstm_mod.available(),
        "linux_mixes": [os.path.basename(p) for p in available_mixes()],
        "configs": configs,
        "results": results,
    }


def api_simulate(body):
    fam = body.get("family", "bimodal")
    if fam == "demo":
        work = demo_workload()
    else:
        n = max(2, min(int(body.get("n_procs", 12)), 60))
        lo, hi = body.get("bursts", [3, 8])
        work = generate(fam, n, float(body.get("load", 0.7)), int(body.get("seed", 1)),
                        (max(1, int(lo)), max(int(lo), min(int(hi), 60))))
    c = float(body.get("cs_cost", 0.1))
    names = body.get("schedulers") or ["RR-5", "BARR-EMA", "BARR-S-EMA"]
    needs_model = any(s.endswith(("GBRT", "LINEAR", "KNN", "LSTM")) for s in names)
    trained = STATE["trained"]
    if needs_model and trained is None:
        return 409, {"error": "Learned predictors are still training; pick EMA/ORACLE variants or wait."}
    out = []
    for s in names[:6]:
        try:
            sched = make(s, trained, body.get("barr"), body.get("barr_s"))
        except KeyError as e:
            return 400, {"error": str(e)}
        res = simulate(sched, work, c, record_queue=True)
        m = collect(res)
        out.append({
            "name": s, "label": label(s), "oracle": s in ORACLE,
            "metrics": exp_run.clean(m),
            "slices": [[pid, idx, r4(a), r4(b), r4(e)] for pid, idx, a, b, e in res.slices],
            "bursts": [[b["pid"], b["idx"], r4(b["ready"]), r4(b["first"]), r4(b["end"]),
                        r4(b["length"]), r4(b["pred"] or 0), int(b["preempted"]), r4(b["q1"])]
                       for b in res.bursts],
            "queue": [[r4(t), pid, q] for t, pid, q in res.queue_log],
            "completion": {p.pid: r4(p.completion) for p in res.procs},
            "end": r4(res.end),
        })
    procs = [{"pid": p.pid, "name": p.name or f"P{p.pid}", "kind": p.kind or "mixed",
              "arrival": r4(p.arrival), "cpu": r4(p.cpu_total), "io": r4(p.io_total),
              "n_bursts": len(p.bursts), "bursts": [[r4(a), r4(b)] for a, b in p.bursts]}
             for p in work]
    return 200, {"procs": procs, "results": out, "cs_cost": c}


def api_result(name):
    path = os.path.join(exp_run.RESULTS, os.path.basename(name), "summary.json")
    if not os.path.exists(path):
        return 404, {"error": "no such result"}
    with open(path) as f:
        return 200, {"name": name, "summary": json.load(f)}


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path):
        if not os.path.isfile(path):
            return self._json(404, {"error": "not found"})
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", TYPES.get(os.path.splitext(path)[1], "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            return self._file(os.path.join(STATIC, "index.html"))
        if path.startswith("/static/"):
            return self._file(os.path.join(STATIC, os.path.basename(path)))
        if path == "/api/meta":
            return self._json(200, meta())
        if path.startswith("/api/results/"):
            return self._json(*api_result(path.rsplit("/", 1)[1]))
        if path.startswith("/results/") and path.endswith("/runs.csv"):
            name = os.path.basename(os.path.dirname(path))
            return self._file(os.path.join(exp_run.RESULTS, name, "runs.csv"))
        if path.startswith("/api/jobs/") and path.endswith("/events"):
            return self._stream(path.split("/")[3])
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._body()
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})
        if path == "/api/simulate":
            try:
                return self._json(*api_simulate(body))
            except Exception as e:
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})
        if path == "/api/jobs":
            kind = body.get("kind")
            if kind not in ("train", "experiment", "tests"):
                return self._json(400, {"error": "kind must be train, experiment or tests"})
            job = start_job(kind, body.get("params"))
            if kind == "train":
                STATE["train_job"] = job
            return self._json(200, {"id": job.id})
        return self._json(404, {"error": "not found"})

    def _stream(self, job_id):
        job = JOBS.get(job_id)
        if job is None:
            return self._json(404, {"error": "no such job"})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        sent = 0
        try:
            while True:
                with job.cond:
                    while sent >= len(job.events) and not job.done:
                        job.cond.wait(timeout=15)
                        if sent >= len(job.events) and not job.done:
                            break
                    batch = job.events[sent:]
                    finished = job.done and sent + len(batch) >= len(job.events)
                if batch:
                    for ev in batch:
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    sent += len(batch)
                else:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
                if finished:
                    break
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


class Server(ThreadingHTTPServer):
    # SO_REUSEADDR on Windows lets two servers share a port silently; refuse instead
    allow_reuse_address = False
    daemon_threads = True


def bind(host, port, tries=20):
    for p in range(port, port + tries):
        try:
            return Server((host, p), Handler), p
        except OSError:
            print(f"port {p} is in use, trying {p + 1}")
    raise SystemExit(f"no free port in {port}-{port + tries - 1}")


def main(host="127.0.0.1", port=8765, open_browser=True):
    preds, report = train_mod.load()
    STATE["trained"], STATE["report"] = preds, report
    if preds is None:
        STATE["train_job"] = start_job("train", {})
        print("no trained predictors yet: training in the background (about 20 s)", flush=True)
    srv, port = bind(host, port)
    url = f"http://{host}:{port}/"
    print(f"BARR dashboard running at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
