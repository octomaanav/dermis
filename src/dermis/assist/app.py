"""DemoApp: the live "Teach an Assistive Hand" session (headless core; web UI and viewer attach to it).

One control loop at 30 Hz owns the physics. Human input (keyboard via the web page, or the glove via
a separate ingestion process), the web page and the 3-D viewer never block it: commands are queued
and applied at the next tick, the viewer is a separate process fed latest-only state.

Recorded data:
    data/episodes/assist_demo/            simulator demonstrations (obs, FINAL actions, human request, assistance)
    data/glove_raw/assist-*.jsonl         raw glove frames (only with glove input), linked by host time
    data/assist_sessions/<session>/       events.jsonl (mode switches, assistance changes, commands),
                                          paired.jsonl (paired trials: INTERACTIVE ILLUSTRATION, not a study)
Presentation results never go into runs/cup or runs/benchmark.
"""
from __future__ import annotations

import collections
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

from ..cup.env import CupScenario
from ..data.episodes import EpisodeStore
from . import learning as L
from .control import ASSISTED, AUTONOMOUS, MANUAL, MODES, ModeController, checkpoint_compatible
from .data import AssistRecorder
from .env import AssistHandEnv, default_config

KEY_STEP = 0.05


class KeyboardHuman:
    label = "KEYBOARD"

    def __init__(self):
        self.g = 0.35
        self.status = "ACTIVE"

    def command(self, cmd: str, value=None) -> None:
        if cmd == "g_up":
            self.g = min(1.0, self.g + KEY_STEP)
        elif cmd == "g_down":
            self.g = max(0.0, self.g - KEY_STEP)
        elif cmd == "open":
            self.g = 0.0
        elif cmd == "close":
            self.g = max(self.g, 0.35)
        elif cmd == "set_g" and value is not None:
            self.g = float(np.clip(float(value), 0.0, 1.0))

    def read(self) -> float:
        return self.g

    def close(self):
        pass


class GloveHuman:
    """Glove -> armed two-point-calibrated GripControl (glove/grip.py) -> grip request g = filtered closure.
    PAUSED / DISARMED: g is HELD (existing stop/hold behaviour). Ingestion runs in a child process."""
    label = "GLOVE"

    def __init__(self, source_kw: dict, mapping, calibration=None, raw_path: str | None = None, transport: str = "",
                 confirmed: bool = False):
        from ..glove.discovery import source_label
        from ..glove.grip import GripControl
        from ..glove.procsource import ProcessTelemetry
        self.mapping = mapping
        pods = list(dict.fromkeys(list(mapping.required_pods) + list(mapping.optional_pods)))
        self.tele = ProcessTelemetry(source_kw, pods, record_path=raw_path, timeout_s=mapping.watchdog_s)
        self.control = GripControl(mapping, calibration)
        self.raw_path, self.transport, self.confirmed = raw_path, transport, confirmed
        self._source_label = source_label
        self.g = 0.35
        self.status = "DISARMED"
        self.message = "calibrate, then arm"
        self.refs: dict = {}
        self._capture = None

    @property
    def sublabel(self) -> str:
        s = self.tele.latest()
        return self._source_label(getattr(self.tele, "label", "") or ("SYNTHETIC" if self.transport == "synthetic" else
                                  "REPLAY" if self.transport in ("replay", "wire-file") else "GLOVE"),
                                  s.frame.device_id if s else None, self.transport, self.confirmed)

    def channels(self) -> dict:
        s = self.tele.latest()
        if s is None:
            return {}
        out = {}
        for i, p in enumerate(s.pods):
            out[p] = {"mag": "missing" if not s.present[i] else "fresh" if s.mag_fresh[i] else
                      "stale" if s.mag_stale[i] else "invalid",
                      "imu": "missing" if not s.present[i] else "fresh" if s.imu_fresh[i] else
                      "stale" if s.imu_stale[i] else "invalid",
                      "required": p in self.mapping.required_pods}
        return out

    def command(self, cmd: str, value=None) -> None:
        from ..glove.grip import CalibrationRejected, ReferenceCapture, build_calibration
        if cmd in ("ref_unloaded", "ref_press"):
            state = "unloaded" if cmd == "ref_unloaded" else "press"
            # non-blocking: read() feeds this capture with each new snapshot for 1.5 s
            self._capture = (state, ReferenceCapture(state, self.mapping.required_pods + self.mapping.optional_pods),
                             time.monotonic() + 1.5, None)
            self.message = f"capturing {state.upper()} reference: hold still (1.5 s)"
        elif cmd == "calibrate":
            if not {"unloaded", "press"} <= set(self.refs):
                self.message = "capture UNLOADED and PRESS references first"
                return
            try:
                cal = build_calibration(self.refs["unloaded"], self.refs["press"], self.mapping)
            except CalibrationRejected as e:
                self.message = "CALIBRATION REJECTED: " + "; ".join(e.reasons)
                return
            ok, msg = self.control.set_calibration(cal)
            self.message = f"calibration {cal.calibration_id} accepted" if ok else msg
        elif cmd == "arm":
            ok, msg = self.control.arm(self.tele.latest(), self.tele.source_stale())
            self.message = msg
        elif cmd == "disarm":
            self.control.disarm()
            self.message = "disarmed by operator"

    def _feed_capture(self, snap) -> None:
        state, cap, t_end, last = self._capture
        if snap is not None and snap is not last:
            cap.add(snap)
            self._capture = (state, cap, t_end, snap)
        if time.monotonic() >= t_end:
            self.refs[state] = cap
            self.message = f"{state} reference captured: " + ", ".join(f"{p} {v['n']} samples" for p, v in cap.stats().items())
            self._capture = None

    def read(self) -> float:
        snap = self.tele.latest()
        if self._capture is not None:
            self._feed_capture(snap)
        cmd = self.control.tick(snap)
        self.status = cmd.state
        if cmd.state == "ARMED" and cmd.closure is not None:
            self.g = float(cmd.closure)
        return self.g                                   # held while PAUSED / DISARMED

    def close(self):
        self.tele.stop_recording()


class DemoApp:
    def __init__(self, human, session_dir: str = None, seed: int = 4_100_000, viewer=None,
                 demo_dir: str = str(L.DEMO_DIR)):
        cfg = default_config()
        self.env = AssistHandEnv(cfg)
        self.ctrl = ModeController(cfg.dt)
        self.human = human
        self.viewer = viewer
        self.seed = seed
        self.session = Path(session_dir or f"data/assist_sessions/{time.strftime('%Y%m%dT%H%M%S')}")
        self.session.mkdir(parents=True, exist_ok=True)
        self.store = EpisodeStore(demo_dir)
        self.cmds: queue.Queue = queue.Queue(maxsize=100)
        self.lock = threading.Lock()
        self.events = collections.deque(maxlen=60)
        self.hist = collections.deque(maxlen=300)           # ~10 s of traces for the page
        self.recording = False
        self.rec = None
        self.last_episode = None
        self.saved: list[str] = []
        self.paired: list[dict] = []
        self._paired_plan: list[str] = []
        self.policy_note = ""
        self.perf = {"control_hz": 0.0, "sim_speed": 0.0, "tick_ms_median": 0.0, "viewer_fps": None}
        self._tick_times = collections.deque(maxlen=90)
        self._assist_was_on = False
        self.state: dict = {}
        self.load_active_policy()
        self.reset_scenario()

    # -- logging -------------------------------------------------------------------------
    def log(self, kind: str, **kw) -> None:
        e = {"t_unix": round(time.time(), 3), "event": kind, "mode": self.ctrl.mode, **kw}
        self.events.append(e)
        with open(self.session / "events.jsonl", "a") as fh:
            fh.write(json.dumps(e, default=str) + "\n")

    # -- learned policy ----------------------------------------------------------------------
    def load_active_policy(self, path: str | None = None) -> None:
        from ..cup.train import CupPolicy
        path = path or L.active_checkpoint()
        if not path:
            self.ctrl.policy, self.policy_note = None, "no active assist_hand checkpoint: AUTONOMOUS disabled"
            return
        ok, why, meta = checkpoint_compatible(path)
        if not ok:
            self.ctrl.policy, self.policy_note = None, f"AUTONOMOUS disabled: {why}"
            return
        prov = meta["provenance"]
        act = json.loads(L.ACTIVE.read_text()) if L.ACTIVE.exists() else {}
        info = {"checkpoint": path, "name": meta["name"], "sources": prov["sources"],
                "train_episodes": len(prov["train_episode_ids"]), "snapshot": act.get("snapshot"),
                "snapshot_counts": act.get("snapshot_counts"), "pretrained_from": prov.get("pretrained_from"),
                "trained_unix": prov.get("trained_unix"), "dev_eval": act.get("dev_eval")}
        self.ctrl.set_policy(CupPolicy.load(path), info)
        self.policy_note = (f"{meta['name']}: trained on {info['train_episodes']} "
                            f"{'/'.join(s.upper() for s in prov['sources'])} episodes"
                            + (" (FINE-TUNED from " + prov['pretrained_from']['name'] + ")" if prov.get("pretrained_from") else ""))

    # -- commands (thread-safe: queued, applied by the control loop) ------------------------------
    def submit(self, cmd: str, value=None) -> None:
        try:
            self.cmds.put_nowait((cmd, value))
        except queue.Full:
            pass

    def apply(self, cmd: str, value=None):
        h = self.human
        if cmd in ("g_up", "g_down", "open", "close", "set_g", "ref_unloaded", "ref_press", "calibrate", "arm", "disarm"):
            if self.ctrl.mode == AUTONOMOUS and cmd in ("g_up", "g_down", "open", "close", "set_g"):
                self.log("human_command_ignored", command=cmd, reason="AUTONOMOUS: human motor commands disconnected")
                return False
            if hasattr(h, "command"):
                h.command(cmd, value)
                self.log("human_input", command=cmd, message=getattr(h, "message", None))
            return True
        if cmd == "mode":
            ok, msg = self.ctrl.switch(value)
            self.log("mode_switch", ok=ok, message=msg, to=value)
            return ok
        if cmd == "reset":
            self.reset_scenario()
            return True
        if cmd == "next":
            self.seed += 1
            self.reset_scenario()
            return True
        if cmd == "add_load":
            ok = self.env.pour()
            self.log("add_load", ok=ok, schedule_peak_n=round(self.scenario.schedule.peak, 2))
            return ok
        if cmd == "record":
            self.recording = bool(value) if value is not None else not self.recording
            if self.recording:
                self.reset_scenario()
            self.log("recording", on=self.recording)
            return True
        if cmd == "paired":
            modes = [m for m in (value or [MANUAL, ASSISTED]) if m in MODES]
            self._paired_plan = list(modes)
            self.log("paired_start", modes=modes, seed=self.seed)
            self._next_paired()
            return True
        if cmd == "reload_policy":
            self.load_active_policy()
            self.log("policy_reloaded", note=self.policy_note)
            return True
        return False

    def _next_paired(self) -> None:
        if not self._paired_plan:
            return
        mode = self._paired_plan.pop(0)
        ok, msg = self.ctrl.switch(mode)
        self.log("mode_switch", ok=ok, message=msg, to=mode, paired=True)
        if not ok:
            self.paired.append({"mode": mode, "seed": self.seed, "termination": "not run: " + msg})
            return self._next_paired()
        self.reset_scenario(pour_at_s=1.0, paired_mode=mode)

    # -- episode ---------------------------------------------------------------------------
    def reset_scenario(self, pour_at_s: float | None = None, paired_mode: str | None = None) -> None:
        self.scenario = CupScenario.sample(self.seed, self.env.cfg)
        self.obs = self.env.reset(self.scenario, pour_at_s=pour_at_s)
        self.ctrl.reset()
        self.hist.clear()
        self._paired_mode = paired_mode
        src = self.human.label.lower() if self.ctrl.mode != AUTONOMOUS else "autonomous"
        self.rec = (AssistRecorder(src, self.ctrl.mode, self.env, self.ctrl.mode,
                                   extra={"session": str(self.session), "paired": paired_mode is not None,
                                          "raw_recording": getattr(self.human, "raw_path", None),
                                          "glove_calibration_id": getattr(getattr(getattr(self.human, "control", None), "cal", None),
                                                                          "calibration_id", None),
                                          "policy": self.ctrl.policy_info if self.ctrl.mode == AUTONOMOUS else None})
                    if (self.recording or paired_mode) else None)
        self.log("reset", seed=self.seed, setup=self.env.stats.termination or "valid grasp", paired_mode=paired_mode)

    def tick(self) -> None:
        t0 = time.monotonic()
        while not self.cmds.empty():
            self.apply(*self.cmds.get_nowait())
        if self.env.done:
            self._publish(None)
            return
        g = None if self.ctrl.mode == AUTONOMOUS else self.human.read()
        a, trace = self.ctrl.act(self.obs, g)
        if self.ctrl.mode == ASSISTED and trace["assist_changed"] != self._assist_was_on:
            self.log("assist_adjusting" if trace["assist_changed"] else "assist_idle",
                     human_effort=trace["human_action"][1] if trace["human_action"] else None, final_effort=float(a[1]))
            self._assist_was_on = trace["assist_changed"]
        if self.rec:
            self.rec.record(self.obs, a, trace, self.ctrl.mode)
        self.obs, done, info = self.env.step(a)
        if self.rec:
            self.rec.log(info)
        self.hist.append({"t": round(self.env.t, 3), "human_g": trace["human_g"],
                          "human_effort": trace["human_action"][1] if trace["human_action"] else None,
                          "final_closure": float(a[0]), "final_effort": float(a[1]),
                          "assist_effort": trace["assist_effort"], "closure_meas": info["closure_meas"],
                          "normal_index": info["normal_index"], "normal_middle": info["normal_middle"],
                          "load_n": info["load_n"], "disp_mm": info["disp_m"] * 1000})
        if done:
            self._finish(info)
        self._tick_times.append(time.monotonic() - t0)
        self._publish(info)

    def _finish(self, info) -> None:
        out = {**self.env.stats.to_dict(), "pour_t_s": self.env.pour_t0}
        self.log("episode_end", termination=out["termination"], seed=self.seed, paired_mode=self._paired_mode)
        if self.rec:
            ep = self.rec.finish(self.obs, out)
            self.store.save(ep)
            self.saved.append(ep.id)
            self.last_episode = ep.id
            self.log("episode_saved", id=ep.id, source=ep.meta["source"], mode=ep.meta["mode"])
            self.rec = None
        if self._paired_mode:
            row = {"mode": self._paired_mode, "seed": self.seed, "termination": out["termination"],
                   "success": out["success"], "peak_normal_n": round(out["peak_normal_n"], 2),
                   "max_disp_mm": round(out["max_disp_m"] * 1000, 1), "mean_effort": round(out["mean_effort"], 3),
                   "human_source": self.human.label if self._paired_mode != AUTONOMOUS else None,
                   "label": "INTERACTIVE ILLUSTRATION (not a controlled study)" if self._paired_mode != AUTONOMOUS else "AUTONOMOUS"}
            self.paired.append(row)
            with open(self.session / "paired.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            self._paired_mode = None
            self._next_paired()

    def _publish(self, info) -> None:
        if self.viewer is not None:
            self.viewer.send(self.env, self.banner(), self.env.hold_state if not self.env.done else self.outcome_text())
            vf = self.viewer.fps()
            if vf is not None:
                self.perf["viewer_fps"] = vf
        with self.lock:
            self.state = self._state(info)

    def banner(self) -> str:
        src = "no human input" if self.ctrl.mode == AUTONOMOUS else self.source_text()
        return f"{self.ctrl.mode}   |   {src}"

    def source_text(self) -> str:
        h = self.human
        sub = getattr(h, "sublabel", None)
        return f"{h.label}" + (f" ({sub}, {h.status})" if sub else "")

    def outcome_text(self) -> str:
        t = self.env.stats.termination
        return {"success": "COMPLETED: held through the load schedule", "drop": "DROPPED",
                "released": "RELEASED by the operator", "damage_break": "EXCESSIVE SIMULATED LOAD (threshold crossed)",
                "excess_slip": "COMPLETED with excessive movement", "setup_failed": "SETUP FAILED",
                "timeout": "TIMED OUT (no load applied)"}.get(t, t or "")

    def _state(self, info) -> dict:
        e = self.env
        st = e.stats
        tr = self.ctrl.last or {}
        return {
            "mode": self.ctrl.mode, "modes": list(MODES), "banner": self.banner(), "source": self.source_text(),
            "human_label": self.human.label, "human_status": getattr(self.human, "status", "ACTIVE"),
            "human_message": getattr(self.human, "message", ""), "channels": self.human.channels() if hasattr(self.human, "channels") else {},
            "autonomous_available": self.ctrl.policy is not None, "policy": self.ctrl.policy_info, "policy_note": self.policy_note,
            "seed": self.seed, "t": round(e.t, 2), "phase": e.phase(), "loading": e.pour_t0 is not None,
            "hold_state": e.hold_state, "outcome": st.termination, "outcome_text": self.outcome_text() if e.done else "",
            "human_g": tr.get("human_g"), "human_action": tr.get("human_action"), "assist_effort": tr.get("assist_effort"),
            "final_action": tr.get("final"), "assist_changed": tr.get("assist_changed"), "assist_limited": tr.get("limited"),
            "closure_meas": None if info is None else info["closure_meas"],
            "contact_n": None if info is None else [info["normal_index"], info["normal_middle"], info["normal_palm"]],
            "support_n": None if info is None else info["support_z"], "load_n": e.load_n,
            "disp_mm": None if info is None else round(info["disp_m"] * 1000, 2),
            "threshold_n": e.cfg.break_force_n, "recording": self.recording or self.rec is not None,
            "saved": list(self.saved[-20:]), "last_episode": self.last_episode, "paired": self.paired[-12:],
            "events": list(self.events)[-14:], "hist": list(self.hist)[-240:], "perf": self.perf,
            "labels": {"simulated": "Contact loads, applied load and displacement are SIMULATED (PyBullet).",
                       "glove": "Glove values are physical fingertip-pad deformation (magnetic change), not force.",
                       "prototype": "Simulated assistive-grasp research prototype - not a prosthesis."}}

    def get_state(self) -> dict:
        with self.lock:
            return dict(self.state)

    def run(self, stop: threading.Event) -> None:
        dt = self.env.cfg.dt
        t_next = time.monotonic()
        n, t_win = 0, time.monotonic()
        while not stop.is_set():
            self.tick()
            n += 1
            now = time.monotonic()
            if now - t_win >= 2.0:
                ticks = sorted(self._tick_times)
                self.perf.update(control_hz=round(n / (now - t_win), 1), sim_speed=round(n / (now - t_win) / 30.0, 2),
                                 tick_ms_median=round(1000 * ticks[len(ticks) // 2], 1) if ticks else 0.0)
                n, t_win = 0, now
            t_next += dt
            time.sleep(max(0.0, t_next - time.monotonic()))
            if time.monotonic() - t_next > 0.5:
                t_next = time.monotonic()          # fell far behind: do not try to catch up (no physics rate change)

    def close(self) -> None:
        self.env.close()
        self.human.close()
        if self.viewer is not None:
            self.viewer.close()
