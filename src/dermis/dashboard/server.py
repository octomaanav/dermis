"""Dermis console: data capture and learning dashboard (standard-library HTTP, 127.0.0.1 only).

Two ways to run it, same UI (static/index.html):
    python -m dermis dashboard [--source synthetic|replay|tcp|...]   monitor + datasets / training / evaluation
    python -m dermis teleop --dashboard-port 8000 ...               + live robot, controller, sim camera

It shows only what is implemented and measured: live telemetry through the shared ingestion
pipeline (metrics.py), recordings, calibration, recorded episodes, training runs, evaluation
results, an optional fragile_cup live session, and a small job runner for scripted collection /
training / evaluation (subprocesses of `python -m dermis`).

Live data reaches the browser through /api/stream (server-sent events, 10 Hz, bounded payloads,
one stream per tab, automatic browser reconnect) or /api/live (same payload, polled). Buttons
POST to /api/control/<name>, which runs the real backend operation and returns its result.
POSTs must carry the X-Dermis header (blocks cross-site form posts) and come from loopback.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import re
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
from .metrics import DEFINITIONS
from .providers import CupWorker, RunnerProvider, TeleopProvider

STATIC = Path(__file__).parent / "static"
MAX_STREAMS = 6
STREAM_HZ = 10.0
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}


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
        if rc is not None and "ended" not in j:
            j["ended"] = time.time()
        tail = Path(j["log"]).read_text(errors="replace").splitlines()[-15:] if Path(j["log"]).exists() else []
        return {"id": jid, "args": j["args"], "state": "running" if rc is None else ("done" if rc == 0 else f"failed ({rc})"),
                "started": j["started"], "elapsed_s": round((j.get("ended") or time.time()) - j["started"], 1),
                "log_tail": tail}

    def list(self) -> list:
        return [self.view(j) for j in sorted(self.jobs, key=lambda k: -self.jobs[k]["started"])]


class FileViews:
    """Datasets, runs and results on disk (read-only), shared by both console modes."""

    a: argparse.Namespace
    roots: list

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
                if "id" not in m:
                    continue
                out.append({"path": str(j), "dataset": str(j.parent), "id": m["id"], "label": m.get("label"),
                            "source": m.get("source"), "seed": m.get("seed"), "length": m.get("length"),
                            "termination": m.get("termination"), "success": m.get("success"),
                            "task": m.get("task", "grasp_lift_place"), "kind": m.get("kind", "demonstration"),
                            "duration_s": None if m.get("length") is None or not m.get("dt")
                            else round(m["length"] * m["dt"], 2),
                            "bc_ready": m.get("bc_ready", m.get("kind") is None and m.get("task") is None)})
        return out[-2000:]

    def episode(self, path: str) -> dict:
        from ..data.episodes import load_episode
        from ..sim.env import ACTION_NAMES, CONTACT_EPS_N, EXCESS_FORCE_N, OBS_NAMES
        ep = load_episode(self._safe(path).with_suffix(""))
        n = len(ep)
        step = max(1, n // 400)
        sl = slice(None, None, step)
        r = lambda a: np.round(np.asarray(a, float)[sl], 4).tolist()
        obs_names = ep.meta.get("obs_names") or list(OBS_NAMES)
        act_names = ep.meta.get("action_names") or list(ACTION_NAMES)
        t = np.asarray(ep.arrays["t"], float) if "t" in ep.arrays else np.arange(n) * float(ep.meta.get("dt", 1 / 30))
        if n:
            t = t - t[0]
        logs = {k: r(v) for k, v in ep.arrays.items()
                if k not in ("obs", "actions", "t", "tags", "final_obs", "expert_actions", "force_l", "force_r")
                and getattr(v, "ndim", 0) == 1 and len(v) == n and np.issubdtype(np.asarray(v).dtype, np.number)}
        ann, notes = [], []
        fl, fr = ep.arrays.get("force_l"), ep.arrays.get("force_r")
        if fl is not None and fr is not None and n:
            touching = (np.asarray(fl) > CONTACT_EPS_N) & (np.asarray(fr) > CONTACT_EPS_N)
            if touching.any():
                k = int(np.argmax(touching))
                ann.append({"tick": k, "t": float(t[k]), "kind": "contact onset"})
                after = np.where(~touching[k:])[0]
                if len(after):
                    k2 = k + int(after[0])
                    ann.append({"tick": k2, "t": float(t[k2]), "kind": "release"})
            thr = float((ep.meta.get("task_config") or {}).get("excess_force_n", EXCESS_FORCE_N))
            over = np.maximum(fl, fr) > thr
            if over.any():
                k = int(np.argmax(over))
                ann.append({"tick": k, "t": float(t[k]), "kind": f"load > {thr:g} N (simulated threshold)"})
            notes.append(f"Contact onset / release: first tick where both simulated finger normal forces rise above / "
                         f"fall below {CONTACT_EPS_N} N (heuristic). Shear spikes are not labelled as slips.")
        outcome = ep.meta.get("outcome") or {}
        for e in outcome.get("events") or []:
            if isinstance(e, dict) and isinstance(e.get("tick"), int) and 0 <= e["tick"] - 1 < n:
                k = e["tick"] - 1
                ann.append({"tick": k, "t": float(t[k]), "kind": f"{e.get('event', 'event')} (env event)"})
        for m in ep.meta.get("markers") or []:
            k = min(int(m["tick"]), max(n - 1, 0))
            ann.append({"tick": k, "t": float(t[k]) if n else 0.0, "kind": f"marker: {m.get('text', '')}"})
        tags = ep.arrays.get("tags")
        return {"meta": ep.meta, "mode_label": "REPLAY (recorded data)", "t": r(t),
                "obs": {nm: r(ep.obs[:, i]) for i, nm in enumerate(obs_names) if i < ep.obs.shape[1]},
                "actions": {nm: r(ep.actions[:, i]) for i, nm in enumerate(act_names) if i < ep.actions.shape[1]},
                "force_l": r(fl) if fl is not None else None, "force_r": r(fr) if fr is not None else None,
                "logs": logs, "annotations": sorted(ann, key=lambda x: x["tick"]), "annotation_notes": notes,
                "tags": tags[sl].tolist() if tags is not None else None,
                "bc_ready": ep.meta.get("bc_ready", ep.meta.get("kind") is None and ep.meta.get("task") is None),
                "decimation": step}

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
            prov = meta.get("provenance", {})
            out.append({"path": str(d), "name": meta.get("name", d.name), "status": status, "log": log,
                        "features": meta.get("feature_groups"), "history": meta.get("history"),
                        "provenance": {k: prov.get(k) for k in ("data_dir", "sources", "data_digest")},
                        "n_train": len(prov.get("train_episode_ids", [])),
                        "n_val": len(prov.get("val_episode_ids", [])),
                        "task": "fragile_cup" if "cup" in d.parts else "grasp_lift_place",
                        "train_seconds": prov.get("train_seconds"), "trained_unix": prov.get("trained_unix"),
                        "has_checkpoint": (d / "model.pt").exists(), "error": status.get("error")})
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

    def evaluation(self) -> dict:
        """Results grouped by (task, scenario suite[, damage model]). Comparisons are only offered within a
        group; run-level variation only across runs whose names differ by a seed suffix _s<k>."""
        rows = []
        for root in (Path(self.a.results_root), Path(self.a.runs_root)):
            for j in sorted(root.rglob("*.json")) if root.exists() else []:
                if j.name in ("comparison.json", "status.json", "metadata.json") or j.with_suffix(".npz").exists():
                    continue
                try:
                    d = json.loads(j.read_text())
                except Exception:
                    continue
                if not isinstance(d, dict) or "summary" not in d or not ({"policy", "controller"} & set(d)):
                    continue
                s = d["summary"]
                who = d.get("controller") or d.get("policy") or {}
                term = s.get("terminations") or {}
                rows.append({
                    "path": str(j), "task": d.get("task", "grasp_lift_place"), "suite": d.get("suite"),
                    "name": who.get("name") or who.get("kind"), "label": who.get("label"), "kind": who.get("kind"),
                    "checkpoint": who.get("checkpoint"), "features": who.get("feature_groups"),
                    "damage_model": d.get("damage_model"), "attempts": s.get("attempts"),
                    "successes": s.get("successes"), "success_rate": s.get("success_rate"),
                    "drops": s.get("episodes_with_drop", term.get("drop", 0)),
                    "excess_load": s.get("episodes_with_excess_force",
                                         sum(v for k, v in term.items() if str(k).startswith("damage"))),
                    "timeouts": s.get("timeouts", term.get("timeout", 0)), "terminations": term,
                    "completion_s_median": s.get("completion_time_s_median"),
                    "completion_s_mean": s.get("completion_time_s_mean"),
                    "peak_force_n_median": s.get("peak_contact_force_n_median", s.get("peak_normal_n_median")),
                    "threshold_n": s.get("excess_force_threshold_n"), "wall_seconds": d.get("wall_seconds"),
                    "demos": (who.get("demonstrations") or {}).get("train_episodes")})
        groups: dict = {}
        for r in rows:
            key = f"{r['task']} | {r['suite']}" + (f" | {r['damage_model']}" if r["damage_model"] else "")
            groups.setdefault(key, []).append(r)
        variation = []
        for key, rs in groups.items():
            fam: dict = {}
            for r in rs:
                m = re.match(r"(.+)_s(\d+)$", str(r["name"]))
                if m and r["success_rate"] is not None:
                    fam.setdefault(m.group(1), []).append(r["success_rate"])
            for name, v in sorted(fam.items()):
                if len(v) >= 2:
                    variation.append({"group": key, "family": name, "runs": len(v), "success_rates": v,
                                      "min": min(v), "max": max(v), "mean": round(sum(v) / len(v), 4)})
        return {"groups": groups, "variation": variation,
                "notes": ["Compare policies only within one group: same task and scenario suite (and damage model).",
                          "Run-level variation = spread over training seeds; no confidence intervals or significance tests.",
                          "Low training loss is not task success; success comes only from AUTONOMOUS rollouts."]}

    def recordings(self) -> list:
        root = Path(self.a.recordings)
        out = []
        for f in sorted(root.glob("*.jsonl")) if root.exists() else []:
            if f.name.endswith(".recv.jsonl"):
                continue
            meta = {}
            mp = Path(str(f) + ".meta.json")
            if mp.exists():
                try:
                    meta = json.loads(mp.read_text())
                except Exception:
                    pass
            out.append({"path": str(f), "bytes": f.stat().st_size, "frames": meta.get("frames"),
                        "duration_s": meta.get("duration_s"), "task": meta.get("task"), "outcome": meta.get("outcome"),
                        "source": meta.get("source"), "kind": "raw telemetry (not a robot demonstration)"})
        return out[-500:]

    def inference_timing(self, checkpoint: str, n: int = 300) -> dict:
        """Measured on this machine: one policy step (feature history + MLP forward), CPU."""
        from ..bc.model import BCPolicy
        d = self._safe(checkpoint)
        pol = BCPolicy.load(d)
        obs = np.zeros(len(pol.meta["obs_names_all"]), np.float32)
        pol.reset()
        for _ in range(20):
            pol.act(obs)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            pol.act(obs)
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        return {"checkpoint": str(d), "n": n, "median_ms": round(ts[n // 2], 4), "p95_ms": round(ts[int(0.95 * n)], 4),
                "note": "one policy step incl. feature history, CPU, measured now; separate from training time"}


class _Observed:
    """Pass-through source wrapper: reports each message's frame number before ingestion, so late
    (out-of-order) frames can be matched to the gaps they fill. Keeps the source's label."""

    def __init__(self, src, dash):
        self.src, self.dash = src, dash
        self.label = getattr(src, "label", "UNKNOWN")

    def __getattr__(self, k):
        return getattr(self.src, k)

    def __iter__(self):
        for msg in self.src:
            cb, obj = self.dash.on_raw_frame, msg.obj
            if cb is not None and isinstance(obj, dict) and isinstance(obj.get("frame"), int):
                cb(obj["frame"])
            yield msg


class Dashboard(FileViews):
    """Standalone console: python -m dermis dashboard."""

    def __init__(self, a):
        self.a = a
        src = open_source(a.source, path=a.path, rate=a.rate, loop=a.loop if a.source != "replay" else True,
                          host=a.host, port=a.port, serial_port=a.serial_port, wire=a.wire, faults=a.faults)
        self.on_raw_frame = None                       # set by RunnerProvider: sees every frame number at the source
        self.runner = TelemetryRunner(_Observed(src, self), TelemetryPipeline(pod_order=[p for p in a.pods.split(",") if p])).start()
        self.calibration = Calibration.load(a.calibration) if a.calibration else None
        self.capturing = False
        self.last_capture_report = None
        self.jobs = JobManager(Path(a.runs_root) / "jobs")
        self.roots = [Path(a.data_root), Path(a.runs_root), Path(a.results_root), Path(a.recordings)]
        self.provider = RunnerProvider(self)
        cps = list(getattr(a, "cup_checkpoint", None) or [])
        ctl = getattr(a, "cup_controller", None) or ""
        ctl = [c for c in (ctl.split(",") if isinstance(ctl, str) else ctl) if c]
        if cps or ctl:
            self.roots.append(Path("runs/cup"))
            self.provider.cup = CupWorker(cps, ctl, seed=getattr(a, "cup_seed", None) or 4_000_000,
                                          hub=self.provider.hub).start()

    # -- telemetry (original endpoints) -------------------------------------------------

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


class TeleopConsole(FileViews):
    """Console hosted inside `python -m dermis teleop --dashboard-port N` (shares its robot and TCP server)."""

    def __init__(self, sess, data_root="data/episodes", runs_root="runs", results_root="results",
                 recordings="data/telemetry", feed_hz: float = 0.0):
        self.a = argparse.Namespace(data_root=data_root, runs_root=runs_root, results_root=results_root,
                                    recordings=recordings)
        self.roots = [Path(data_root), Path(runs_root), Path(results_root), Path(recordings)]
        self.jobs = JobManager(Path(runs_root) / "jobs")
        self.provider = TeleopProvider(sess, feed_hz, data_root, recordings)

    def status(self) -> dict:                                    # original /api/status shape, for old clients
        st = self.provider.status()
        return {"mode": st["control_mode"], "source": st["source"], "pipeline": self.provider.server.pipeline.stats(),
                "recording": st["recording"]["raw"], "calibration": st["calibration"]}

    def telemetry(self, since: int) -> dict:
        return {"rows": [], "last": since, "note": "use /api/live or /api/stream"}


# --------------------------------------------------------------------------------------
# live payload
# --------------------------------------------------------------------------------------

def live_payload(con, pod: str | None, cur: dict, full_status: bool = True) -> dict:
    prov = con.provider
    hub = prov.hub
    pods = list(hub.pods)
    pod = pod if pod in pods else (pods[0] if pods else None)
    samples, cur["s"] = hub.samples(pod, cur.get("s", 0), prov.baseline) if pod else ([], cur.get("s", 0))
    cmds, cur["c"] = hub.commands_since(cur.get("c", 0))
    events, cur["e"] = hub.events_since(cur.get("e", 0))
    cup = None if getattr(prov, "cup", None) is None else prov.cup.status()
    metrics = hub.metrics.summary()
    if hasattr(prov, "frame_age_ms"):
        metrics["frame_age_ms"] = prov.frame_age_ms()          # monitor: exclude the hub's polling delay
    return {"t": hub.rel(time.monotonic_ns()), "pod": pod, "pods_order": pods,
            "status": prov.status() if full_status else None, "metrics": metrics,
            "pods": hub.pod_summary(prov.baseline), "samples": samples, "commands": cmds, "events": events,
            "cup": cup, "has_baseline": {p: prov.baseline(p) is not None for p in pods}, "cursor": dict(cur)}


def make_handler(dash):
    streams = {"n": 0}
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _local(self) -> bool:
            try:
                return ipaddress.ip_address(self.client_address[0]).is_loopback
            except ValueError:
                return False

        def _json(self, obj, code=200):
            body = json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _bytes(self, body: bytes, ctype: str, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _stream(self, pod):
            with lock:
                if streams["n"] >= MAX_STREAMS:
                    return self._json({"error": f"too many live streams (max {MAX_STREAMS}); close other tabs"}, 429)
                streams["n"] += 1
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                self.close_connection = True
                cur = {}
                self.wfile.write(b"retry: 1000\n\n")
                k = 0
                while True:
                    payload = live_payload(dash, pod, cur, full_status=(k % 5 == 0))
                    self.wfile.write(b"data: " + json.dumps(payload, default=str).encode() + b"\n\n")
                    self.wfile.flush()
                    k += 1
                    time.sleep(1.0 / STREAM_HZ)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with lock:
                    streams["n"] -= 1

        def do_GET(self):
            if not self._local():
                return self._json({"error": "local access only"}, 403)
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html") or (u.path.startswith("/static/") and "/.." not in u.path):
                    name = "index.html" if u.path in ("/", "/index.html") else u.path[len("/static/"):]
                    f = (STATIC / name).resolve()
                    if not str(f).startswith(str(STATIC.resolve())) or not f.is_file():
                        return self._json({"error": "not found"}, 404)
                    return self._bytes(f.read_bytes(), CONTENT_TYPES.get(f.suffix, "application/octet-stream"))
                if u.path == "/api/status":
                    return self._json(dash.status())
                if u.path == "/api/telemetry":
                    return self._json(dash.telemetry(int(q.get("since", 0))))
                if u.path == "/api/episodes":
                    return self._json(dash.episodes())
                if u.path == "/api/episode":
                    return self._json(dash.episode(q["path"]))
                if u.path == "/api/runs":
                    return self._json(dash.runs())
                if u.path == "/api/results":
                    return self._json(dash.results())
                if u.path == "/api/jobs":
                    return self._json(dash.jobs.list())
                if u.path == "/api/stream":
                    return self._stream(q.get("pod"))
                if u.path == "/api/live":
                    cur = {k[1:]: int(v) for k, v in q.items() if k in ("cs", "cc", "ce")}
                    return self._json(live_payload(dash, q.get("pod"), cur))
                if u.path == "/api/console":
                    return self._json({"kind": dash.provider.kind, "status": dash.provider.status(),
                                       "definitions": DEFINITIONS, "pods": list(dash.provider.hub.pods)})
                if u.path == "/api/raw":
                    return self._json(dash.provider.raw())
                if u.path == "/api/frame":
                    return self._json({"frame": dash.provider.frame_dict(q.get("pod"))})
                if u.path == "/api/sim.jpg":
                    feed = dash.provider.feed if dash.provider.kind == "teleop" else (
                        dash.provider.cup.feed if dash.provider.cup else None)
                    jpg, _ = (None, 0) if feed is None else feed.jpeg()
                    return self._bytes(jpg, "image/jpeg") if jpg else self._json({"error": "no sim feed"}, 404)
                if u.path == "/api/evaluation":
                    return self._json(dash.evaluation())
                if u.path == "/api/recordings":
                    return self._json(dash.recordings())
                return self._json({"error": "not found"}, 404)
            except PermissionError as e:
                return self._json({"error": str(e)}, 403)
            except Exception as e:
                return self._json({"error": repr(e)}, 500)

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            if not self._local():
                return self._json({"error": "local access only"}, 403)
            new_api = u.path.startswith("/api/control/") or u.path == "/api/inference"
            if new_api and self.headers.get("X-Dermis") != "1":
                return self._json({"error": "missing X-Dermis header"}, 403)
            try:
                body = json.loads(raw or b"{}")
                if u.path.startswith("/api/control/"):
                    return self._json(dash.provider.control(u.path.rsplit("/", 1)[1], body))
                if u.path == "/api/inference":
                    return self._json(dash.inference_timing(str(body["checkpoint"])))
                if u.path == "/api/record/start":
                    return self._json(dash.provider.control("record_start", body))
                if u.path == "/api/record/stop":
                    return self._json(dash.provider.control("record_stop", body))
                if u.path == "/api/calibration/capture":
                    return self._json(dash.start_capture(int(body.get("frames", 200))))
                if u.path == "/api/jobs":
                    return self._json(dash.jobs.start([str(x) for x in body["args"]]))
                return self._json({"error": "not found"}, 404)
            except PermissionError as e:
                return self._json({"error": str(e)}, 403)
            except Exception as e:
                return self._json({"error": str(e) if isinstance(e, (ValueError, RuntimeError, TimeoutError)) else repr(e)}, 400)
    return H


def serve(a) -> int:
    dash = Dashboard(a)
    srv = ThreadingHTTPServer(("127.0.0.1", a.http_port), make_handler(dash))
    srv.daemon_threads = True
    print(f"Dermis console on http://127.0.0.1:{srv.server_address[1]}  (source: {dash.runner.label})", flush=True)
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
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, dash


def serve_teleop(sess, port: int, feed_hz: float = 0.0, **roots) -> tuple[ThreadingHTTPServer, TeleopConsole]:
    """Console inside the teleop process. HTTP runs in background threads; the provider's queued
    commands and camera renders run on the simulation thread via provider.service()."""
    con = TeleopConsole(sess, feed_hz=feed_hz, **roots)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(con))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="dermis-console", daemon=True).start()
    return srv, con
