"""Judge-facing page for "Teach an Assistive Hand" (local only, 127.0.0.1).

The page polls /api/state at 10 Hz; buttons and keys POST commands that the control loop applies at
its next tick (never blocking physics). Learning actions reuse the background training workflow.
Styling reuses the console's design system (dashboard/static/console.css, served read-only).
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..data.episodes import EpisodeStore
from . import learning as L

STATIC = Path(__file__).parent / "static"
CONSOLE_CSS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "console.css"


class LearnJobs:
    """Evaluations run in a background thread; training runs as a background subprocess (teaching.py)."""

    def __init__(self):
        self.running: dict = {}

    def evaluate(self, name: str) -> str:
        if name in self.running:
            return "already evaluating"

        def job():
            try:
                L.evaluate_candidate(name, workers=2)
                self.running[name] = "done"
            except Exception as e:                    # surfaced in the candidate list
                self.running[name] = f"failed: {e}"
        self.running[name] = "evaluating"
        threading.Thread(target=job, daemon=True).start()
        return "evaluation started (development scenarios)"


def episode_payload(demo_dir: str, eid: str) -> dict:
    ep = EpisodeStore(demo_dir).load(eid)
    a = ep.arrays
    n = len(ep)
    k = max(1, n // 300)
    sl = slice(None, None, k)
    r = lambda x: [None if not np.isfinite(v) else round(float(v), 4) for v in np.asarray(x)[sl]]
    return {"meta": {k_: ep.meta.get(k_) for k_ in ("id", "source", "label", "mode", "controller", "seed", "length",
                                                     "termination", "success", "created_unix", "raw_recording",
                                                     "glove_calibration_id", "policy", "paired")},
            "t": [round(i * ep.meta["dt"], 3) for i in range(n)][sl],
            "human_g": r(a["human_g"]), "human_effort": r(a["human_actions"][:, 1]), "final_effort": r(a["actions"][:, 1]),
            "final_closure": r(a["actions"][:, 0]), "assist_effort": r(a["assist_effort"]),
            "normal_index": r(a["normal_index"]), "normal_middle": r(a["normal_middle"]), "load_n": r(a["load_n"]),
            "disp_mm": r(a["disp_m"] * 1000), "tags": list(np.asarray(a["tags"])[sl]),
            "simulated_note": "contact, load and displacement are SIMULATED (PyBullet)"}


def make_handler(app, jobs: LearnJobs, demo_dir: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)).encode())

        def _local(self) -> bool:
            if self.client_address[0] not in ("127.0.0.1", "::1"):
                self._json({"error": "local access only"}, 403)
                return False
            return True

        def do_GET(self):
            if not self._local():
                return
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/console.css":
                    self._send(200, CONSOLE_CSS.read_bytes() if CONSOLE_CSS.exists() else b"", "text/css")
                elif u.path == "/api/state":
                    self._json(app.get_state())
                elif u.path == "/api/episodes":
                    store = EpisodeStore(demo_dir)
                    rows = [store.load_meta(i) for i in store.ids()]
                    self._json([{k: m.get(k) for k in ("id", "source", "label", "mode", "seed", "length", "termination",
                                                       "created_unix", "paired")} for m in rows][-60:])
                elif u.path == "/api/episode":
                    self._json(episode_payload(demo_dir, q["id"]))
                elif u.path == "/api/learn/candidates":
                    self._json({"candidates": L.candidates(), "active": L.active_checkpoint(), "jobs": jobs.running})
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:
                self._json({"error": repr(e)}, 500)

        def do_POST(self):
            if not self._local():
                return
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            u = urlparse(self.path)
            try:
                if u.path == "/api/cmd":
                    app.submit(body["cmd"], body.get("value"))
                    self._json({"queued": body["cmd"]})
                elif u.path == "/api/learn/snapshot":
                    out = L.snapshot([demo_dir], body["name"], ids=body.get("ids") or None)
                    self._json({"snapshot": str(out), "manifest": json.loads((out / "manifest.json").read_text())})
                elif u.path == "/api/learn/train":
                    init = L.active_checkpoint() if body.get("fine_tune") else None
                    d = L.train(body["snapshot"], body["name"], init_from=init, epochs=body.get("epochs"))
                    self._json({"candidate": str(d), "training": "fine-tune from " + init if init else "from scratch"})
                elif u.path == "/api/learn/evaluate":
                    self._json({"status": jobs.evaluate(body["name"])})
                elif u.path == "/api/learn/activate":
                    rec = L.activate(body["name"])
                    app.submit("reload_policy")
                    self._json({"active": rec})
                else:
                    self._json({"error": "not found"}, 404)
            except SystemExit as e:
                self._json({"error": str(e)}, 400)
            except Exception as e:
                self._json({"error": repr(e)}, 400)
    return H


def serve(app, port: int, demo_dir: str) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, LearnJobs(), demo_dir))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
