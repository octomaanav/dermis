"""Local dashboard: python -m dermis dashboard [--source synthetic|replay|tcp|...]

Standard library HTTP server bound to 127.0.0.1. It shows only what is implemented: live
telemetry through the shared ingestion pipeline, telemetry recording, baseline capture,
recorded episodes (REPLAY inspection), training runs, evaluation results, and a small job
runner for scripted collection / training / evaluation (subprocesses of `python -m dermis`).
JobManager is the extension point for real orchestration later.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..telemetry.calibration import BaselineCapture, Calibration
from ..telemetry.pipeline import TelemetryPipeline
from ..telemetry.runner import TelemetryRunner
from ..telemetry.sources import open_source

STATIC = Path(__file__).parent / "static"


def _f(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


class JobManager:
    ALLOWED = {"collect", "train", "evaluate", "smoke-test"}

    def __init__(self, log_dir: Path):
        self.jobs: dict[str, dict] = {}
        self.log_dir = log_dir

    def start(self, args: list[str]) -> dict:
        if not args or args[0] not in self.ALLOWED:
            raise ValueError(f"job must be one of {sorted(self.ALLOWED)}")
        jid = uuid.uuid4().hex[:8]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = self.log_dir / f"job-{jid}.log"
        proc = subprocess.Popen([sys.executable, "-m", "dermis", *args], stdout=open(log, "w"),
                                stderr=subprocess.STDOUT)
        self.jobs[jid] = {"id": jid, "args": args, "log": str(log), "proc": proc, "started": time.time()}
        return self.view(jid)

    def view(self, jid: str) -> dict:
        j = self.jobs[jid]
        rc = j["proc"].poll()
        tail = Path(j["log"]).read_text(errors="replace").splitlines()[-15:] if Path(j["log"]).exists() else []
        return {"id": jid, "args": j["args"], "state": "running" if rc is None else ("done" if rc == 0 else f"failed ({rc})"),
                "started": j["started"], "log_tail": tail}

    def list(self) -> list:
        return [self.view(j) for j in sorted(self.jobs, key=lambda k: -self.jobs[k]["started"])]


class Dashboard:
    def __init__(self, a):
        self.a = a
        src = open_source(a.source, path=a.path, rate=a.rate, loop=a.loop if a.source != "replay" else True,
                          host=a.host, port=a.port, serial_port=a.serial_port, wire=a.wire, faults=a.faults)
        self.runner = TelemetryRunner(src, TelemetryPipeline(pod_order=[p for p in a.pods.split(",") if p])).start()
        self.calibration = Calibration.load(a.calibration) if a.calibration else None
        self.capturing = False
        self.jobs = JobManager(Path(a.runs_root) / "jobs")
        self.roots = [Path(a.data_root), Path(a.runs_root), Path(a.results_root), Path(a.recordings)]

    # -- telemetry ---------------------------------------------------------------

    def status(self) -> dict:
        snap = self.runner.latest()
        cal = self.calibration
        return {
            "mode": "TELEMETRY MONITOR",
            "source": self.runner.label,
            "source_kind": self.a.source,
            "source_stale": self.runner.source_stale(),
            "source_finished": self.runner.finished,
            "source_error": repr(self.runner.error) if self.runner.error else None,
            "pipeline": self.runner.pipeline.stats(),
            "recent_events": list(self.runner.pipeline.events)[-12:],
            "recording": {"active": self.runner.rec_path is not None, "path": self.runner.rec_path,
                          "frames": self.runner.rec_frames},
            "calibration": None if cal is None else {
                "id": cal.calibration_id, "device_id": cal.device_id, "config_id": cal.config_id,
                "pods": list(cal.pods), "scale_ut": cal.scale_ut,
                "valid_for_stream": bool(snap and (snap.frame.device_id, snap.frame.config_id) == (cal.device_id, cal.config_id)),
                "note": cal.note},
            "capturing_baseline": self.capturing,
            "pods": list(self.runner.pipeline.pod_order),
            "units_note": "Magnetic field in microtesla. With a calibration: change from unloaded baseline (uT) "
                          "and normalized deformation (unitless). Not force.",
        }

    def telemetry(self, since: int) -> dict:
        rows = []
        for seq, s in self.runner.since(since)[-400:]:
            cal = self.calibration.apply(s, use_held=False) if self.calibration else None
            pods = []
            for i, p in enumerate(s.pods):
                state = ("missing" if not s.present[i] else "fresh" if s.mag_fresh[i] else
                         "stale" if s.mag_stale[i] else "invalid")
                imu = ("missing" if not s.present[i] else "fresh" if s.imu_fresh[i] else
                       "stale" if s.imu_stale[i] else "invalid")
                b = s.mag_raw[i]
                pods.append({
                    "mag": state, "imu": imu, "sat": bool(s.mag_saturated[i]),
                    "b_norm": _f(np.linalg.norm(b)) if np.isfinite(b).all() else None,
                    "db": _f(cal["delta_norm_ut"][i]) if cal is not None and cal["mask"][i] else None,
                    "deform": _f(cal["deformation"][i], 4) if cal is not None and cal["mask"][i] else None,
                    "acc_norm": _f(np.linalg.norm(s.acc_raw[i])) if np.isfinite(s.acc_raw[i]).all() else None,
                    "gyr_norm": _f(np.linalg.norm(s.gyr_raw[i]), 4) if np.isfinite(s.gyr_raw[i]).all() else None,
                    "age_ms": _f(s.mag_age_us[i] / 1000.0, 2),
                })
            rows.append({"seq": seq, "frame": s.frame.frame, "t": s.frame.t_device_us, "lost": s.lost_before,
                         "cfg": s.frame.config_id, "pods": pods})
        return {"rows": rows, "last": rows[-1]["seq"] if rows else since}

    def start_capture(self, frames: int) -> dict:
        """Blocking: collect the next `frames` snapshots (max 30 s) as the unloaded baseline."""
        cap = BaselineCapture(min_samples=min(50, frames // 2))
        hist = self.runner.since(-1)
        last = hist[-1][0] if hist else 0
        snaps = []
        self.capturing = True
        try:
            t0 = time.monotonic()
            while len(snaps) < frames and time.monotonic() - t0 < 30:
                new = self.runner.since(snaps[-1][0] if snaps else last)
                snaps += new
                time.sleep(0.02)
        finally:
            self.capturing = False
        for _, s in snaps[:frames]:
            cap.add(s)
        cal, report = cap.finish(scale_ut=500.0)
        out = Path(self.a.recordings).parent / "calibration" / f"{cal.calibration_id}.json"
        cal.save(out)
        self.calibration = cal
        return {"calibration_id": cal.calibration_id, "path": str(out), "frames": len(snaps[:frames]),
                "report": report, "source": self.runner.label}

    # -- files -------------------------------------------------------------------

    def _safe(self, path: str) -> Path:
        p = Path(path).resolve()
        if not any(str(p).startswith(str(r.resolve())) for r in self.roots):
            raise PermissionError("path outside dashboard roots")
        return p

    def episodes(self) -> list:
        out = []
        for root in (Path(self.a.data_root), Path(self.a.runs_root)):
            for j in sorted(root.rglob("*.json")) if root.exists() else []:
                if not (j.with_suffix(".npz")).exists():
                    continue
                try:
                    m = json.loads(j.read_text())
                except Exception:
                    continue
                out.append({"path": str(j), "dataset": str(j.parent), "id": m["id"], "label": m.get("label"),
                            "source": m.get("source"), "seed": m.get("seed"), "length": m.get("length"),
                            "termination": m.get("termination"), "success": m.get("success")})
        return out[-2000:]

    def episode(self, path: str) -> dict:
        from ..data.episodes import load_episode
        from ..sim.env import ACTION_NAMES, OBS_NAMES
        ep = load_episode(self._safe(path).with_suffix(""))
        step = max(1, len(ep) // 300)
        sl = slice(None, None, step)
        r = lambda a: np.round(a[sl], 4).tolist()
        return {"meta": ep.meta, "mode_label": "REPLAY (recorded data)", "t": r(ep.arrays["t"]),
                "obs": {n: r(ep.obs[:, i]) for i, n in enumerate(OBS_NAMES)},
                "actions": {n: r(ep.actions[:, i]) for i, n in enumerate(ACTION_NAMES)},
                "force_l": r(ep.arrays["force_l"]), "force_r": r(ep.arrays["force_r"]),
                "tags": ep.arrays["tags"][sl].tolist() if "tags" in ep.arrays else None}

    def runs(self) -> list:
        out = []
        root = Path(self.a.runs_root)
        for st in sorted(root.rglob("status.json")) if root.exists() else []:
            d = st.parent
            try:
                status = json.loads(st.read_text())
            except Exception:
                continue
            meta = json.loads((d / "metadata.json").read_text()) if (d / "metadata.json").exists() else {}
            log = []
            if (d / "train_log.csv").exists():
                import csv
                log = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(open(d / "train_log.csv"))]
            out.append({"path": str(d), "name": meta.get("name", d.name), "status": status, "log": log,
                        "features": meta.get("feature_groups"), "history": meta.get("history"),
                        "provenance": {k: meta.get("provenance", {}).get(k) for k in ("data_dir", "sources", "data_digest")},
                        "n_train": len(meta.get("provenance", {}).get("train_episode_ids", [])),
                        "n_val": len(meta.get("provenance", {}).get("val_episode_ids", []))})
        return out

    def results(self) -> dict:
        res, comps = [], []
        for root in (Path(self.a.results_root), Path(self.a.runs_root)):
            if not root.exists():
                continue
            for j in sorted(root.rglob("*.json")):
                if j.name == "comparison.json":
                    try:
                        c = json.loads(j.read_text())
                        comps.append({"path": str(j), "title": c["title"], "rows": c["rows"]})
                    except Exception:
                        pass
                    continue
                try:
                    d = json.loads(j.read_text())
                except Exception:
                    continue
                if isinstance(d, dict) and "summary" in d and "policy" in d:
                    res.append({"path": str(j), "suite": d["suite"], "policy": d["policy"], "summary": d["summary"]})
        return {"results": res, "comparisons": comps}


def make_handler(dash: Dashboard):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    body = (STATIC / "index.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif u.path == "/api/status":
                    self._json(dash.status())
                elif u.path == "/api/telemetry":
                    self._json(dash.telemetry(int(q.get("since", 0))))
                elif u.path == "/api/episodes":
                    self._json(dash.episodes())
                elif u.path == "/api/episode":
                    self._json(dash.episode(q["path"]))
                elif u.path == "/api/runs":
                    self._json(dash.runs())
                elif u.path == "/api/results":
                    self._json(dash.results())
                elif u.path == "/api/jobs":
                    self._json(dash.jobs.list())
                else:
                    self._json({"error": "not found"}, 404)
            except PermissionError as e:
                self._json({"error": str(e)}, 403)
            except Exception as e:
                self._json({"error": repr(e)}, 500)

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            try:
                if u.path == "/api/record/start":
                    path = Path(dash.a.recordings) / f"telemetry-{dash.runner.label.lower()}-{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
                    self._json({"path": dash.runner.start_recording(str(path))})
                elif u.path == "/api/record/stop":
                    self._json(dash.runner.stop_recording())
                elif u.path == "/api/calibration/capture":
                    self._json(dash.start_capture(int(body.get("frames", 200))))
                elif u.path == "/api/jobs":
                    self._json(dash.jobs.start([str(x) for x in body["args"]]))
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:
                self._json({"error": repr(e)}, 400)
    return H


def serve(a) -> int:
    dash = Dashboard(a)
    srv = ThreadingHTTPServer(("127.0.0.1", a.http_port), make_handler(dash))
    print(f"Dermis dashboard on http://127.0.0.1:{srv.server_address[1]}  (source: {dash.runner.label})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def serve_in_thread(a) -> tuple[ThreadingHTTPServer, Dashboard]:
    """For tests: start on an ephemeral port in a background thread."""
    dash = Dashboard(a)
    srv = ThreadingHTTPServer(("127.0.0.1", a.http_port), make_handler(dash))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, dash
