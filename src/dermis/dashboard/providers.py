"""Where the dashboard gets live data, and how its buttons reach the real backend.

    TeleopProvider   inside `python -m dermis teleop --dashboard-port N`: the TCP receiver, the
                     arm/disarm controller and the PyBullet robot of that process. Commands are
                     queued and executed on the simulation (main) thread at the next control tick,
                     so the browser never touches PyBullet or the controller directly and robot
                     control never waits on the browser.
    RunnerProvider   inside `python -m dermis dashboard`: the existing TelemetryRunner source
                     (synthetic / replay / tcp / serial). No robot control (MONITOR).
    CupWorker        optional, in the dashboard process: a fragile_cup LiveSession owned by one
                     worker thread (it is not thread-safe), driven only through LiveSession.do().
                     Mode AUTONOMOUS: no human motor input object exists.

Camera feed: measured on an M2 Mac, PyBullet's software renderer takes ~1 s per 512x320 frame and
holds the GIL (it froze ingestion and control), and the GPU renderer (GUI window only) takes ~36 ms
at 320x200, longer than one 33 ms control tick. So the feed is OFF by default, GUI-only and opt-in
(--dashboard-feed-hz), rendered on the thread that owns the PyBullet client, and its cost is shown in
the console. The presentation view instead draws a schematic from measured simulator state, and the
native PyBullet window is meant to sit beside the browser.
"""
from __future__ import annotations

import io
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .live import LiveHub

FEED_SIZE = (320, 200)


class OwnerQueue:
    """Run callables on the thread that owns a resource; callers wait for the result (bounded)."""

    def __init__(self, maxsize: int = 32):
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)

    def call(self, fn, timeout: float = 5.0):
        box = {"done": threading.Event()}
        try:
            self.q.put_nowait((fn, box))
        except queue.Full:
            raise RuntimeError("control queue full; the simulation thread is busy") from None
        if not box["done"].wait(timeout):
            raise TimeoutError("the simulation thread did not run the command in time")
        if "error" in box:
            raise box["error"]
        return box["result"]

    def drain(self, limit: int = 8) -> None:
        for _ in range(limit):
            try:
                fn, box = self.q.get_nowait()
            except queue.Empty:
                return
            try:
                box["result"] = fn()
            except Exception as e:                    # reported to the caller, never kills the loop
                box["error"] = e
            box["done"].set()


class Feed:
    """Latest rendered camera frame (raw RGB) + lazily encoded JPEG."""

    def __init__(self):
        self._lock = threading.Lock()
        self.rgb = None
        self.id = 0
        self.t = None
        self.render_ms: list = []
        self._jpeg = (None, -1)

    def put(self, rgb: np.ndarray, render_ms: float) -> None:
        with self._lock:
            self.rgb, self.id, self.t = rgb, self.id + 1, time.time()
            self.render_ms = (self.render_ms + [render_ms])[-50:]

    def jpeg(self) -> tuple[bytes | None, int]:
        with self._lock:
            rgb, fid = self.rgb, self.id
            if self._jpeg[1] == fid:
                return self._jpeg[0], fid
        if rgb is None:
            return None, 0
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
        with self._lock:
            self._jpeg = (buf.getvalue(), fid)
        return buf.getvalue(), fid

    def stats(self) -> dict:
        with self._lock:
            r = sorted(self.render_ms)
            return {"frames": self.id, "age_s": None if self.t is None else round(time.time() - self.t, 2),
                    "render_ms_median": None if not r else round(r[len(r) // 2], 1),
                    "render_ms_max": None if not r else round(r[-1], 1)}


def render(client: int, camera: tuple, size=FEED_SIZE, hardware: bool = False) -> np.ndarray:
    """Camera image of an existing PyBullet client. Call on the thread that owns `client`."""
    import pybullet as p
    dist, yaw, pitch, target = camera
    view = p.computeViewMatrixFromYawPitchRoll(target, dist, yaw, pitch, 0, 2, physicsClientId=client)
    proj = p.computeProjectionMatrixFOV(50, size[0] / size[1], 0.01, 5, physicsClientId=client)
    w, h, rgba, _, _ = p.getCameraImage(size[0], size[1], view, proj, physicsClientId=client,
                                        renderer=p.ER_BULLET_HARDWARE_OPENGL if hardware else p.ER_TINY_RENDERER,
                                        flags=p.ER_NO_SEGMENTATION_MASK)
    return np.reshape(np.asarray(rgba, dtype=np.uint8), (h, w, 4))[:, :, :3].copy()


# --------------------------------------------------------------------------------------
# Teleop demonstrations (robot observation / action), separate from raw telemetry recordings
# --------------------------------------------------------------------------------------

class DemoRecorder:
    """Records what the robot measured and what was actually commanded each control tick.
    Saved as <id>.npz + <id>.json next to other episodes, with kind "teleop_demonstration".
    These are NOT behavioral-cloning episodes: the hand has no BC task, and the arm demo lacks the
    task's object/target observations. bc_ready is false and says why."""

    def __init__(self, root: str, robot: str, source: str, task: str, notes: str, dt: float, calib: dict | None):
        self.root, self.robot = Path(root), robot
        self.id = f"teleop-{robot}-{time.strftime('%Y%m%dT%H%M%S')}"
        self.meta = {"id": self.id, "kind": "teleop_demonstration", "task": task or "unlabeled", "notes": notes,
                     "source": source, "label": source, "robot": robot, "created_unix": time.time(), "dt": dt,
                     "calibration": calib, "bc_ready": False,
                     "bc_note": "teleop demonstration: robot state + issued commands for inspection. Not in the "
                                "grasp-lift-place BC observation format; not used for training."}
        self.t, self.obs, self.act, self.state, self.frame = [], [], [], [], []
        self.markers: list = []
        self.obs_names = self.act_names = None

    def add(self, t: float, obs: dict, act: dict, state: str, frame) -> None:
        if self.obs_names is None:
            self.obs_names, self.act_names = list(obs), list(act)
        self.t.append(t)
        self.obs.append([obs[k] for k in self.obs_names])
        self.act.append([act[k] for k in self.act_names])
        self.state.append(state)
        self.frame.append(-1 if frame is None else int(frame))

    def mark(self, text: str) -> int:
        self.markers.append({"tick": len(self.t), "text": text[:120]})
        return len(self.markers)

    def finish(self, outcome: str, note: str = "") -> dict:
        if not self.t:
            raise ValueError("nothing recorded")
        self.root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.root / f"{self.id}.npz", t=np.array(self.t), obs=np.array(self.obs, np.float32),
                            actions=np.array(self.act, np.float32), tags=np.array(self.state, dtype="U24"),
                            telemetry_frame=np.array(self.frame, np.int64))
        meta = {**self.meta, "length": len(self.t), "obs_names": self.obs_names, "action_names": self.act_names,
                "markers": self.markers, "termination": outcome or "unannotated", "success": outcome == "success",
                "outcome_note": note, "duration_s": round(self.t[-1] - self.t[0], 2)}
        (self.root / f"{self.id}.json").write_text(json.dumps(meta, indent=1))
        return {"id": self.id, "path": str(self.root / f"{self.id}.json"), "ticks": len(self.t)}


def _flatten(prefix: str, v) -> dict:
    v = list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v]
    return {f"{prefix}{i}" if len(v) > 1 else prefix: float(x) for i, x in enumerate(v)}


# --------------------------------------------------------------------------------------
# Teleop process
# --------------------------------------------------------------------------------------

class TeleopProvider:
    kind = "teleop"

    def __init__(self, sess, feed_hz: float = 0.0, data_root: str = "data/episodes", recordings: str = "data/telemetry"):
        self.sess = sess
        self.server = sess.server
        self.hub = LiveHub(tuple(self.server.pipeline.pod_order))
        self.ownerq = OwnerQueue()
        self.feed = Feed()
        self.feed_hz = feed_hz if sess.gui else 0.0          # headless = software renderer: far too slow
        self.feed_note = (None if feed_hz <= 0 else "GPU render of the live PyBullet scene on the sim thread"
                          if sess.gui else "disabled: headless rendering (~1 s/frame) would stall control")
        self._next_render = 0.0
        self.data_root, self.recordings = data_root, recordings
        self.demo: DemoRecorder | None = None
        self.rec_labels: dict = {}
        self._epoch = None
        self._state = None
        self._ev_seen = 0
        self.tick_ms: list = []
        self.server.observers.append(self.hub.on_snapshot)
        self.server.line_listeners.append(self._on_line)
        log = sess.ctrl.log
        sess.ctrl.log = lambda m: (log(m), self.hub.event("info", "controller", str(m).replace("[teleop] ", "")))
        self.cup = None

    # -- ingestion thread -------------------------------------------------------------

    def _on_line(self, verdict, frame, t_ns, proc_us, why) -> None:
        ep = self.server.connection_epoch
        if ep != self._epoch:
            self._epoch = ep
            self.hub.metrics.stream_reset(f"connection {ep}", t_ns, new_connection=True)
            self.hub.event("info", "connection", f"controller connected (epoch {ep}, {self.server.peer})")
        self.hub.on_line(verdict, frame, t_ns, proc_us, why)

    # -- main (simulation) thread, once per control tick ------------------------------

    def service(self, row: dict, tick_ms: float | None = None) -> None:
        if tick_ms is not None:
            self.tick_ms = (self.tick_ms + [tick_ms])[-300:]
        self._last_robot = {k: row[k] for k in ("joints", "palm_rp", "offset", "ee", "opening", "cmd_opening") if k in row}
        self._last_robot["robot"] = self.sess.robot.name
        cmd = {"state": row["state"]}
        extra = row.get("cmd") or {}
        if "flex" in extra:
            for pod, f in zip(self.sess.mapping.cfg.grip_pods, extra["flex"]):
                cmd[f"flex:{pod}"] = round(float(f), 4)
            cmd["offset_x"] = round(float(extra.get("offset", [0])[0]), 4)
        else:
            cmd["opening_cmd_m"] = round(float(row["cmd_opening"]), 5)
            if "opening" in row:
                cmd["opening_meas_m"] = round(float(row["opening"]), 5)
        self.hub.on_command(cmd)
        if row["state"] != self._state:
            if self._state is not None:
                self.hub.event("warn" if row["state"] != "ARMED" else "info", "control",
                               f"{self._state} -> {row['state']}" + (f": {row['reason']}" if row.get("reason") else ""))
            self._state = row["state"]
        pev = list(self.server.pipeline.events)
        n = self.server.pipeline.counters["stream_resets"] + self.server.pipeline.counters["restarts"] + \
            self.server.pipeline.counters["clock_resets"] + self.server.pipeline.counters["device_changes"]
        if n != self._ev_seen:
            self._ev_seen = n
            if pev and pev[-1]["event"] in ("stream_restart", "clock_reset", "device_change"):
                self.hub.metrics.stream_reset(pev[-1]["event"])
                self.hub.event("warn", "stream", json.dumps(pev[-1])[:200])
        if self.demo is not None:
            self.demo.add(time.monotonic(), self._obs(row), self._act(row), row["state"],
                          None if self.hub.latest is None else self.hub.latest.frame.frame)
        self.ownerq.drain()
        now = time.monotonic()
        if self.feed_hz > 0 and now >= self._next_render:
            self._next_render = now + 1.0 / self.feed_hz
            t0 = time.perf_counter()
            rgb = render(self.sess.robot.client, self._camera(), hardware=True)
            self.feed.put(rgb, (time.perf_counter() - t0) * 1000)

    def _camera(self):
        if self.sess.robot.name == "hand":
            from ..server.teleop.hand import CAMERA
            return CAMERA
        return (0.9, 50, -30, (0.52, 0.0, 0.15))

    def _obs(self, row) -> dict:
        if "joints" in row:
            return {**_flatten("joint", row["joints"]), **_flatten("tilt", row["palm_rp"]), **_flatten("offset", row["offset"])}
        return {**_flatten("ee", row["ee"]), "opening": row["opening"]}

    def _act(self, row) -> dict:
        extra = row.get("cmd") or {}
        if "flex" in extra:
            return {**_flatten("flex", extra["flex"]), **_flatten("attitude", extra["attitude"]),
                    **_flatten("offset_cmd", extra.get("offset", [0, 0, 0]))}
        return {**_flatten("v", row["cmd_v"]), "opening_cmd": row["cmd_opening"]}

    # -- any thread ---------------------------------------------------------------------

    def baseline(self, pod):
        cal = self.sess.ctrl.calib
        b = None if cal is None else cal.mag_baseline_ut.get(pod)
        return None if b is None else np.asarray(b, float)

    def status(self) -> dict:
        L = self.server.status()
        s = self.sess.ctrl.status()
        pipe = self.server.pipeline
        cal = self.sess.ctrl.calib
        latest = self.hub.latest
        mode = {"DISARMED": "DISARMED", "ARMED": "TELEOPERATION", "PAUSED": "TELEOPERATION"}[s.state]
        return {
            "app": "teleop", "source": L["source"] or ("NO SOURCE" if not L["accepted"] else "TCP-UNVERIFIED"),
            "source_note": "SYNTHETIC = generated development data" if L["source"] == "SYNTHETIC" else None,
            "control_mode": mode, "control_state": s.state, "control_reason": s.reason,
            "connection": {"state": L["state"], "peer": L["peer"], "epoch": L["epoch"], "listen": f"{L['host']}:{L['port']}"},
            "device_id": pipe.device_id, "config_id": pipe.config_id,
            "recording": {"raw": {"active": self.server.record_path is not None, "path": self.server.record_path,
                                  "frames": self.server.recording_frames,
                                  "duration_s": None if self.server.recording_started_ns is None else
                                  round((time.monotonic_ns() - self.server.recording_started_ns) / 1e9, 1)},
                          "demo": None if self.demo is None else {"id": self.demo.id, "ticks": len(self.demo.t),
                                                                  "task": self.demo.meta["task"], "markers": len(self.demo.markers)}},
            "calibration": self._calib_info(cal, latest),
            "calibrating": s.calibrating,
            "usable_age_ms": s.usable_age_ms, "frame_age_ms": L["last_accept_age_ms"],
            "control_valid": s.control_valid, "signals": s.signals,
            "robot": self.sess.robot.name, "mapping": self.sess.mapping.label,
            "watchdog_ms": round(self.sess.ctrl_cfg.watchdog_s * 1000), "auto": bool(getattr(self.sess, "auto", False)),
            "compat_normalized": L["normalized"],
            "control_loop": self._loop_stats(),
            "feed": {**self.feed.stats(), "hz": self.feed_hz, "note": self.feed_note},
            "robot_state": self._robot_state(),
        }

    def _robot_state(self) -> dict:
        """Measured simulator state for the presentation schematic (read-only; values copied each tick)."""
        return dict(getattr(self, "_last_robot", {}) or {})

    def _loop_stats(self) -> dict:
        t = sorted(self.tick_ms)
        return {"ticks": self.sess.tick_count, "overruns": self.sess.overruns,
                "tick_ms_median": None if not t else round(t[len(t) // 2], 2),
                "tick_ms_p95": None if not t else round(t[int(0.95 * (len(t) - 1))], 2),
                "budget_ms": round(1000 / self.sess.ctrl_cfg.control_hz, 1)}

    def _calib_info(self, cal, latest) -> dict | None:
        c = self.sess.ctrl
        col = c.collector
        progress = None if col is None else {"elapsed_s": round(col.elapsed(), 2), "duration_s": col.duration_s,
                                             "samples": len(col.snaps), "min_samples": col.min_samples,
                                             "skipped": col.skipped}
        req = [f"{k}:{p}" for k, p in self.sess.mapping.required_channels()]
        if cal is None:
            return {"active": False, "progress": progress, "required_channels": req}
        return {"active": True, "progress": progress, "required_channels": req, "path": cal.path,
                "id": Path(cal.path).stem if cal.path else None, "created_unix": cal.created_unix,
                "device_id": cal.device_id, "config_id": cal.config_id, "pod_ids": cal.pod_ids,
                "compatible": None if latest is None else cal.mismatch(latest) is None,
                "mismatch": None if latest is None else cal.mismatch(latest),
                "samples": cal.samples, "duration_s": cal.duration_s, "source": cal.source,
                "stationary_variation": {"mag_std_ut": cal.mag_std_ut, "gyro_rms_rads": cal.gyr_rms_rads,
                                         "note": "observed variation while held still during capture; not sensor accuracy"}}

    def raw(self) -> dict:
        last, acc = self.server.last_line, self.server.last_accepted_line
        now = time.monotonic_ns()
        f = None if self.hub.latest is None else self.hub.latest.frame
        return {"last_line": last, "last_accepted": acc,
                "age_ms": None if last is None else round((now - last["t_recv_ns"]) / 1e6, 1),
                "device": None if f is None else {"frame": f.frame, "t_device_us": f.t_device_us, "t_host_ns": f.t_host_ns,
                                                  "config_id": f.config_id, "flags": sorted(f.flags)},
                "timing_note": "t_device_us is the device clock; host receive time is host monotonic; "
                               "t_host_ns stays null without clock sync, so they are never compared."}

    def frame_dict(self, pod: str | None = None) -> dict | None:
        f = self.hub.latest
        return None if f is None else (f.frame.raw if pod is None else f.frame.raw.get("pods", {}).get(pod))

    def control(self, name: str, body: dict) -> dict:
        ctrl, sess = self.sess.ctrl, self.sess
        if name == "arm":
            def arm():
                sess._auto_paused = False                # an explicit arm also resumes --auto
                return ctrl.arm(self.server.status())
            return self._ok(*self.ownerq.call(arm))
        if name == "disarm":
            self.ownerq.call(lambda: sess.do("disarm"))
            return {"ok": True, "state": ctrl.state}
        if name == "calibrate":
            return self.ownerq.call(lambda: self._calibrate(bool(body.get("replace"))))
        if name == "record_start":
            labels = {k: str(body.get(k, ""))[:80] for k in ("task", "source_note", "operator")}
            path = Path(self.recordings) / f"telemetry-{(self.server.status()['source'] or 'none').lower()}-{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
            self.rec_labels = {**labels, "path": str(path), "started_unix": time.time()}
            return {"ok": True, "path": self.server.start_recording(str(path))}
        if name == "record_stop":
            out = self.server.stop_recording()
            meta = {**self.rec_labels, **out, "outcome": str(body.get("outcome", ""))[:40],
                    "note": str(body.get("note", ""))[:200], "source": self.server.status()["source"],
                    "kind": "raw_telemetry_recording", "device_id": self.server.pipeline.device_id,
                    "config_id": self.server.pipeline.config_id}
            if out.get("path"):
                Path(out["path"] + ".meta.json").write_text(json.dumps(meta, indent=1))
            return {"ok": True, **meta}
        if name == "demo_start":
            if self.demo is not None:
                raise ValueError("a demonstration is already being recorded")
            cal = ctrl.calib
            self.demo = DemoRecorder(str(Path(self.data_root) / f"teleop_{sess.robot.name}"), sess.robot.name,
                                     self.server.status()["source"] or "NONE", str(body.get("task", ""))[:60],
                                     str(body.get("notes", ""))[:200], 1.0 / sess.ctrl_cfg.control_hz,
                                     None if cal is None else {"path": cal.path, "device_id": cal.device_id, "config_id": cal.config_id})
            self.hub.event("info", "demo", f"recording demonstration {self.demo.id}")
            return {"ok": True, "id": self.demo.id}
        if name == "demo_mark":
            if self.demo is None:
                raise ValueError("no demonstration is being recorded")
            n = self.demo.mark(str(body.get("text", "marker")))
            self.hub.event("info", "marker", str(body.get("text", "marker")))
            return {"ok": True, "markers": n}
        if name == "demo_stop":
            if self.demo is None:
                raise ValueError("no demonstration is being recorded")
            d, self.demo = self.demo, None
            out = d.finish(str(body.get("outcome", "")), str(body.get("note", "")))
            self.hub.event("info", "demo", f"saved {out['id']} ({out['ticks']} ticks)")
            return {"ok": True, **out}
        raise ValueError(f"unknown control {name!r}")

    @staticmethod
    def _ok(ok, msg=None):
        return {"ok": bool(ok), "message": msg}

    def _calibrate(self, replace: bool) -> dict:
        """Main thread. Refuses unless the required channels are fresh right now; asks for confirmation
        before replacing an existing calibration."""
        ctrl = self.sess.ctrl
        if ctrl.collector is not None:
            raise ValueError("calibration already in progress")
        if ctrl.calib is not None and not replace:
            return {"ok": False, "needs_confirmation": True,
                    "message": "a calibration is active; confirm to replace it"}
        snap = self.hub.latest
        if snap is None:
            raise ValueError("refused: no telemetry received")
        age = self.hub.metrics.summary()["frame_age_ms"]
        if age is None or age > 500:
            raise ValueError(f"refused: no new frame accepted for {age:.0f} ms; a baseline needs live frames")
        why = self.sess.mapping.check_channels(snap, require_fresh=True)
        if why:
            raise ValueError(f"refused: required channel not usable ({why})")
        ok, msg = ctrl.start_calibration()
        return {"ok": ok, "message": msg}


# --------------------------------------------------------------------------------------
# Standalone dashboard (existing TelemetryRunner)
# --------------------------------------------------------------------------------------

class RunnerProvider:
    kind = "monitor"

    def __init__(self, dash):
        self.dash = dash
        self.runner = dash.runner
        self.hub = LiveHub(tuple(self.runner.pipeline.pod_order))
        self.feed = Feed()
        self._cursor = 0
        self._counts = {}
        self._stop = threading.Event()
        self.cup = None
        self._last_raw = None
        dash.on_raw_frame = self._raw_frame
        threading.Thread(target=self._poll, name="dash-poll", daemon=True).start()

    def _raw_frame(self, frame: int) -> None:
        """Source thread, before ingestion: a frame number behind the newest one seen is a late frame."""
        last = self._last_raw
        if last is not None:
            back = (last - frame) % (1 << 32)
            if 0 < back <= 64:                         # same reorder window as the pipeline
                self.hub.metrics.late_frame(frame)
                return
        self._last_raw = frame

    def _poll(self) -> None:
        """Feed the hub from the runner's history (it has no per-frame hooks). Rejections are taken
        from pipeline counter deltas, so their timestamps are poll times (+-50 ms)."""
        verdict_of = {"duplicates": "duplicate", "out_of_order": "out_of_order", "adapter_errors": "adapter_rejected"}
        while not self._stop.is_set():
            for seq, snap in self.runner.since(self._cursor):
                self._cursor = seq
                self.hub.on_line("accepted", snap.frame.frame, snap.frame.t_recv_ns)
                self.hub.on_snapshot(snap)
            c = self.runner.pipeline.counters
            now = time.monotonic_ns()
            for k, v in verdict_of.items():
                for _ in range(c[k] - self._counts.get(k, c[k])):
                    self.hub.on_line(v, None, now)
                self._counts[k] = c[k]
            r = c["restarts"] + c["device_changes"]
            if r != self._counts.get("resets", r):
                self.hub.metrics.stream_reset("stream restart / device change")
                self.hub.event("warn", "stream", "frame counter restart or device change")
            self._counts["resets"] = r
            time.sleep(0.05)

    def frame_age_ms(self) -> float | None:
        """Age of the reader's newest accepted frame, read directly (the hub is fed by a 50 ms poller,
        whose delay must not be counted as frame age)."""
        s = self.runner.latest()
        return None if s is None else round((time.monotonic_ns() - s.frame.t_recv_ns) / 1e6, 1)

    def source_label(self) -> str:
        """What the data actually is. Network/stdin sources carry no proof of origin, so they are
        SYNTHETIC (device id 'synthetic*', as the dev sender sends) or TCP-UNVERIFIED, never GLOVE;
        the same rule as the TCP teleop receiver. File sources are REPLAY, a USB serial device GLOVE."""
        kind = self.dash.a.source
        if kind in ("tcp", "stdin"):
            s = self.hub.latest
            if s is None:
                return "NO SOURCE"
            return "SYNTHETIC" if str(s.frame.device_id).startswith("synthetic") else "TCP-UNVERIFIED"
        return self.runner.label

    def baseline(self, pod):
        cal = self.dash.calibration
        if cal is None or pod not in getattr(cal, "pods", {}):
            return None
        return np.asarray(cal.pods[pod]["baseline_ut"], float)

    def status(self) -> dict:
        st = self.dash.status()
        cal = st["calibration"]
        r = st["recording"]
        cup = None if self.cup is None else self.cup.status()
        return {
            "app": "monitor", "source": self.source_label(),
            "source_note": {"SYNTHETIC": "SYNTHETIC = generated development data",
                            "REPLAY": "REPLAY = recorded data played back",
                            "TCP-UNVERIFIED": "network stream of unknown origin (not verified as the glove)"}.get(self.source_label()),
            "control_mode": "AUTONOMOUS" if cup and cup.get("running") else "NO ROBOT CONTROL",
            "control_state": "fragile_cup AUTONOMOUS (simulator)" if cup and cup.get("running") else "monitor only",
            "control_reason": None,
            "connection": {"state": "error" if st["source_error"] else "finished" if st["source_finished"]
                           else "stale" if st["source_stale"] else "receiving", "detail": st["source_error"]},
            "device_id": st["pipeline"]["device_id"], "config_id": st["pipeline"]["config_id"],
            "recording": {"raw": {"active": r["active"], "path": r["path"], "frames": r["frames"], "duration_s": None},
                          "demo": None},
            "calibration": None if cal is None else {"active": True, "id": cal["id"], "device_id": cal["device_id"],
                                                     "config_id": cal["config_id"], "compatible": cal["valid_for_stream"],
                                                     "pods": cal["pods"], "kind": "magnetic baseline (telemetry/calibration.py)",
                                                     "stationary_variation": getattr(self.dash, "last_capture_report", None)},
            "calibrating": st["capturing_baseline"],
            "usable_age_ms": None, "frame_age_ms": self.frame_age_ms(),
            "control_valid": None, "signals": {}, "robot": None, "mapping": None, "watchdog_ms": None,
            "control_loop": None, "feed": None if self.cup is None else {**self.cup.feed.stats(), "hz": self.cup.feed_hz},
        }

    def raw(self) -> dict:
        s = self.hub.latest
        f = None if s is None else s.frame
        ev = [e for e in self.runner.pipeline.events if e.get("event") in ("adapter_error", "bad_message")]
        return {"last_line": None if f is None else {"text": json.dumps(f.raw), "verdict": "accepted", "reason": None,
                                                     "frame": f.frame, "t_recv_ns": f.t_recv_ns, "processing_us": None},
                "last_rejection": ev[-1] if ev else None,
                "age_ms": None if f is None else round((time.monotonic_ns() - f.t_recv_ns) / 1e6, 1),
                "device": None if f is None else {"frame": f.frame, "t_device_us": f.t_device_us, "t_host_ns": f.t_host_ns,
                                                  "config_id": f.config_id, "flags": sorted(f.flags)},
                "timing_note": "Accepted frame as decoded (re-serialized); this source reader keeps no raw line text."}

    def frame_dict(self, pod: str | None = None):
        s = self.hub.latest
        return None if s is None else (s.frame.raw if pod is None else s.frame.raw.get("pods", {}).get(pod))

    def control(self, name: str, body: dict) -> dict:
        if name.startswith("cup_"):
            if self.cup is None:
                raise ValueError("no fragile_cup live session (start the dashboard with --cup-controller or --cup-checkpoint)")
            return self.cup.control(name[4:], body)
        if name == "calibrate":
            snap = self.hub.latest
            if snap is None:
                raise ValueError("refused: no telemetry received")
            if not any(snap.mag_fresh):
                raise ValueError("refused: no magnetometer channel is fresh")
            age = self.frame_age_ms()
            if age is None or age > 500:
                raise ValueError(f"refused: no new frame accepted for {age:.0f} ms (stream stopped, or only "
                                 "duplicates arriving); a baseline needs live frames")
            if self.dash.calibration is not None and not body.get("replace"):
                return {"ok": False, "needs_confirmation": True, "message": "a calibration is active; confirm to replace it"}
            out = self.dash.start_capture(int(body.get("frames", 200)))
            self.dash.last_capture_report = {"per_pod": out["report"],
                                             "note": "observed std while held still during capture; not sensor accuracy"}
            return {"ok": True, **out}
        if name == "record_start":
            path = Path(self.dash.a.recordings) / f"telemetry-{self.source_label().lower()}-{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
            self.rec_labels = {k: str(body.get(k, ""))[:80] for k in ("task", "source_note", "operator")}
            return {"ok": True, "path": self.runner.start_recording(str(path))}
        if name == "record_stop":
            out = self.runner.stop_recording()
            meta = {**getattr(self, "rec_labels", {}), **out, "outcome": str(body.get("outcome", ""))[:40],
                    "note": str(body.get("note", ""))[:200], "source": self.source_label(), "kind": "raw_telemetry_recording"}
            if out.get("path"):
                Path(out["path"] + ".meta.json").write_text(json.dumps(meta, indent=1))
            return {"ok": True, **meta}
        if name in ("arm", "disarm", "demo_start", "demo_stop", "demo_mark"):
            raise ValueError("no robot in this process: run `python -m dermis teleop --dashboard-port 8000` "
                             "for teleoperation and demonstrations")
        raise ValueError(f"unknown control {name!r}")


# --------------------------------------------------------------------------------------
# fragile_cup live session (AUTONOMOUS), owned by one worker thread
# --------------------------------------------------------------------------------------

class CupWorker:
    """Runs dermis.cup.live.LiveSession in AUTONOMOUS mode on its own thread at the control rate.
    Controllers are trained checkpoints (--cup-checkpoint) and/or the cup's SCRIPTED baselines
    (--cup-controller), labelled as such. Dashboard buttons map to LiveSession.do(); there is no
    human motor input in this mode."""

    def __init__(self, checkpoints=(), controllers=(), seed: int = 4_000_000, feed_hz: float = 0.0, hub: LiveHub | None = None):
        self.checkpoints, self.controllers, self.seed = list(checkpoints), list(controllers), seed
        self.feed, self.feed_hz = Feed(), feed_hz
        self.ownerq = OwnerQueue()
        self.hub = hub
        self.kinds: dict = {}
        self._snap: dict | None = None
        self._lock = threading.Lock()
        self.error: str | None = None
        self.running = False
        self.ticks = 0
        self.compare_running = False
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="cup-live", daemon=True)

    def start(self) -> "CupWorker":
        self.thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _build(self):
        from ..cup.controllers import PolicyController, scripted
        from ..cup.env import CupConfig, CupHoldEnv
        from ..cup.live import AUTONOMOUS, LiveSession
        policies = {}
        for c in self.checkpoints:
            from ..cup.train import CupPolicy
            policies[Path(c).name] = PolicyController(CupPolicy.load(c))
            self.kinds[Path(c).name] = "AUTONOMOUS (learned checkpoint)"
        for name in self.controllers:
            policies[name] = scripted(name)
            self.kinds[name] = "SCRIPTED baseline" + (" (uses privileged simulator state)" if "oracle" in name else "")
        if not policies:
            raise ValueError("no cup controllers")
        env = CupHoldEnv(CupConfig())                         # physics only, headless
        return env, LiveSession(env, AUTONOMOUS, self.seed, policies=policies)

    def _run(self) -> None:
        try:
            env, sess = self._build()
        except Exception as e:
            self.error = repr(e)
            return
        self.sess, self.running = sess, True
        dt = env.cfg.dt
        nxt, next_render = time.monotonic(), 0.0
        last_outcome = None
        while not self._stop.is_set():
            self.ownerq.drain()
            sess.tick()
            self.ticks += 1
            snap = sess.snapshot()
            snap["controller_kind"] = self.kinds.get(snap["active_policy"])
            with self._lock:
                self._snap = snap
            if self.hub is not None and snap["outcome"] != last_outcome and snap["outcome"]:
                self.hub.event("warn" if snap["outcome"] != "success" else "info", "cup", f"outcome: {snap['outcome']}")
            last_outcome = snap["outcome"]
            now = time.monotonic()
            if self.feed_hz > 0 and now >= next_render:
                next_render = now + 1.0 / self.feed_hz
                t0 = time.perf_counter()
                cx = getattr(env.cfg, "cup_xy", (0.5, 0.0))[0]
                self.feed.put(render(env.client, (0.45, 70, -15, (cx, 0.0, 0.12))), (time.perf_counter() - t0) * 1000)
            nxt += dt
            time.sleep(max(0.0, nxt - time.monotonic()))
            if time.monotonic() - nxt > 0.5:
                nxt = time.monotonic()

    def status(self) -> dict:
        with self._lock:
            snap = dict(self._snap) if self._snap else None
        return {"running": self.running, "error": self.error, "ticks": self.ticks, "snapshot": snap,
                "kinds": self.kinds, "compare_running": self.compare_running,
                "labels": {"applied_load_n": "load proxy (simulated): shown to humans, never a policy input",
                           "break_force_n": "simulated excessive-load threshold per finger; the cup is rigid and does not bend",
                           "finger_normal_n": "simulated fingertip normal contact force",
                           "cup_slip_mm": "cup displacement relative to the hand (simulated)"}}

    def control(self, name: str, body: dict) -> dict:
        if not self.running:
            raise ValueError(self.error or "cup session not running")
        if name in ("pour", "reset", "next", "switch"):
            res = self.ownerq.call(lambda: self.sess.do(name), timeout=5)
            return {"ok": True, "result": res if isinstance(res, (bool, type(None))) else "done"}
        if name == "select":
            target = body.get("policy")

            def sel():
                if target not in self.sess.policy_names:
                    raise ValueError(f"unknown controller {target!r}")
                while self.sess.active != target:
                    self.sess.do("switch")
                return self.sess.active
            return {"ok": True, "active": self.ownerq.call(sel)}
        if name == "compare":
            self.compare_running = True
            try:
                return {"ok": True, "results": self.ownerq.call(lambda: self.sess.paired_compare(), timeout=300)}
            finally:
                self.compare_running = False
        raise ValueError(f"unknown cup control {name!r}")
