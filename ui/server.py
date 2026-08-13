"""
EvoSynth Web Server
Serves the UI and exposes API endpoints to launch/stop the Python optimizer.

  POST /api/run     { budget, k_iters, elite_pct, inject_pct, swap_thr, step_delay }
  GET  /api/status  → { running, progress }
  POST /api/stop    → stops the running optimization
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time

from flask import Flask, jsonify, request, send_from_directory

# ── Paths ──────────────────────────────────────────────────────────────────
UI_DIR     = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.normpath(os.path.join(UI_DIR, "..", "Final_year Implementation"))
STATE_FILE = os.path.join(UI_DIR, "state.json")
RUNNER_PY  = os.path.join(UI_DIR, "runner.py")   # generated per-run

app = Flask(__name__, static_folder=UI_DIR, static_url_path="")

# ── Process state ──────────────────────────────────────────────────────────
_lock      = threading.Lock()
_proc      = None           # subprocess.Popen
_running   = False
_last_cfg  = {}

# ── Static files ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(UI_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(UI_DIR, path)

# ── Status ─────────────────────────────────────────────────────────────────
@app.route("/api/state")
def api_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    # If file doesn't exist but we are running, return an initializing state
    with _lock:
        running = _running
    if running:
        return jsonify({
            "phase": "INITIALIZATION",
            "evals_used": 0,
            "budget": 100,
            "best_fitness": 0,
            "best_algo": "",
            "best_params": {},
            "composite": {},
            "t1_islands": {},
            "t2_islands": {},
            "events": [{"type": "info", "msg": "Generating initial population... This may take a while for real datasets."}]
        })
    return jsonify({"error": "not running"}), 404

@app.route("/api/status")
def api_status():
    with _lock:
        running = _running
    progress = 0
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
            b = s.get("budget", 1) or 1
            e = s.get("evals_used", 0)
            progress = round(e / b * 100, 1)
        except Exception:
            pass
    return jsonify({"running": running, "progress": progress})

# ── Start run ───────────────────────────────────────────────────────────────
@app.route("/api/run", methods=["POST"])
def api_run():
    global _proc, _running, _last_cfg

    data       = request.get_json(force=True) or {}
    budget     = int(data.get("budget",     300))
    k_iters    = int(data.get("k_iters",    5))
    elite_pct  = float(data.get("elite_pct",  0.20))
    inject_pct = float(data.get("inject_pct", 0.20))
    swap_thr   = float(data.get("swap_thr",   0.05))
    step_delay = float(data.get("step_delay", 0.08))
    eval_type  = data.get("evaluator_type", "xgboost")

    with _lock:
        if _running:
            return jsonify({"ok": False, "msg": "Already running"}), 409
        _last_cfg = data

    # Write a placeholder state file so the UI doesn't 404 while generating the initial population
    init_state = {
        "phase": "INITIALIZATION",
        "evals_used": 0,
        "budget": budget,
        "best_fitness": 0,
        "best_algo": "",
        "best_params": {},
        "composite": {},
        "t1_islands": {},
        "t2_islands": {},
        "events": [{"type": "info", "msg": f"Generating initial population... Training models via {eval_type.upper()}."}]
    }
    with open(STATE_FILE, "w") as f:
        json.dump(init_state, f)

    if eval_type == "xgboost":
        imports = '''
from automl_engine.core.search_space import get_xgboost_search_space
from automl_engine.core.evaluator import XGBoostRealEvaluator
space     = get_xgboost_search_space()
evaluator = XGBoostRealEvaluator()
'''
    else:
        imports = '''
from automl_engine.core.search_space import get_cnn_search_space
from automl_engine.core.evaluator import CNNDummyEvaluator
space     = get_cnn_search_space()
evaluator = CNNDummyEvaluator()
'''

    # Write a self-contained runner script so we avoid import caching issues
    runner_src = f"""
import sys
sys.path.insert(0, {repr(ENGINE_DIR)})
from automl_engine.engine.controller import DynamicOptimizer
{imports}

optimizer = DynamicOptimizer(
    space, evaluator,
    budget={budget},
    k_iters={k_iters},
    step_delay={step_delay},
    swap_threshold={swap_thr},
    stagnation_limit=2,
    elite_pct={elite_pct},
    inject_pct={inject_pct},
)
optimizer.run()
"""
    with open(RUNNER_PY, "w") as f:
        f.write(runner_src)

    def _run():
        global _proc, _running
        log_path = os.path.join(UI_DIR, "runner.log")
        logf = None
        try:
            with _lock:
                _running = True
            logf = open(log_path, "w")
            proc = subprocess.Popen(
                [sys.executable, "-u", RUNNER_PY],
                stdout=logf,
                stderr=logf,
                cwd=UI_DIR,
            )
            with _lock:
                _proc = proc
            proc.wait()
        except Exception as e:
            print(f"[server] Runner error: {e}", flush=True)
        finally:
            if logf:
                logf.close()
            with _lock:
                _running = False
                _proc    = None

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({"ok": True, "msg": "Optimization started"})

# ── Stop run ────────────────────────────────────────────────────────────────
@app.route("/api/stop", methods=["POST"])
def api_stop():
    global _proc
    with _lock:
        p = _proc
    if p and p.poll() is None:
        try:
            p.send_signal(signal.SIGTERM)
        except Exception:
            pass
    return jsonify({"ok": True})

# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 58)
    print("  EvoSynth — Web Server")
    print("  Dashboard: http://localhost:8000")
    print("  Press Ctrl+C to stop")
    print("=" * 58, flush=True)
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
