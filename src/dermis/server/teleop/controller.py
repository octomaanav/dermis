"""TELEOPERATION controller: latest telemetry -> bounded, smoothed gripper commands, with a watchdog.

Pure Python/numpy; it never touches PyBullet or sockets. The sim loop calls tick() once per
control period with the newest (seq, Snapshot) from the LatestSlot and the server status.

States
    DISARMED  default. No Cartesian motion, gripper opening held. Leaves only via arm().
    ARMED     mapping drives the gripper.
    PAUSED    armed, but input is unusable: no usable control frame for watchdog_s of host
              time, or a required channel is invalid / null / STALE beyond the hold.
              Motion stops at once, opening is held. With resume_policy="auto" it re-arms after
              resume_frames consecutive usable frames; with "manual" the operator must arm().
    Connection loss, device/config change or invalid calibration -> DISARMED; arm() required.

Limits (applied every tick, whatever the mapping says)
    |vx|, |vy|, |vz| <= max_speed; velocity low-passed (vel_tau_s) and accel-limited (max_accel)
    opening in [0, max_opening], low-passed (open_tau_s) and rate-limited (max_open_rate)
    workspace limits are enforced again by the env on the integrated position target.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .calibration import Calibration, CalibrationCollector, CalibrationError

DISARMED, ARMED, PAUSED = "DISARMED", "ARMED", "PAUSED"


@dataclass
class ControllerConfig:
    control_hz: float = 30.0
    watchdog_s: float = 0.25
    max_speed: float = 0.05             # m/s per axis
    max_accel: float = 0.5              # m/s^2
    z_speed: float = 0.03               # m/s while a Z key is held
    max_open_rate: float = 0.10         # m/s of total opening
    open_tau_s: float = 0.10
    vel_tau_s: float = 0.15
    resume_frames: int = 5
    resume_policy: str = "auto"         # auto | manual (after watchdog/channel pauses only)
    calib_duration_s: float = 1.0
    calib_max_mag_std_ut: float = 80.0


@dataclass
class Command:
    v: np.ndarray                       # (3,) m/s
    opening: float                      # m
    state: str
    reason: str | None
    extra: dict = field(default_factory=dict)   # robot-specific targets (e.g. HandController finger flex)


@dataclass
class ControllerStatus:
    state: str = DISARMED
    reason: str | None = "disarmed (default): calibrate, then arm"
    calibrated: bool = False
    calibrating: bool = False
    control_valid: bool = False
    usable_age_ms: float | None = None
    last_frame: int | None = None
    signals: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)


class TeleopController:
    def __init__(self, mapping, cfg: ControllerConfig | None = None, initial_opening: float = 0.08,
                 calibration: Calibration | None = None, calib_dir: str | None = None, log=print):
        self.mapping = mapping
        self.cfg = cfg or ControllerConfig()
        self.log = log
        self.calib = calibration
        self.calib_dir = calib_dir
        self.collector: CalibrationCollector | None = None
        self.state = DISARMED
        self.reason: str | None = "disarmed (default): calibrate, then arm"
        self.opening = float(initial_opening)
        self.v = np.zeros(3)
        self._seen_seq = 0
        self._last_snap = None
        self._last_out = None
        self._last_usable_ns: int | None = None
        self._consecutive_usable = 0
        self._input_reason: str | None = "no telemetry yet"
        self._armed_epoch: int | None = None
        self.messages: list[str] = []

    # -- operator actions --------------------------------------------------

    def _say(self, msg: str) -> str:
        self.messages = (self.messages + [msg])[-6:]
        self.log(f"[teleop] {msg}")
        return msg

    def start_calibration(self) -> tuple[bool, str]:
        if self.state != DISARMED:
            return False, self._say("calibration refused: disarm first (never recalibrate during a grasp)")
        self.collector = CalibrationCollector(self.mapping, self.cfg.calib_duration_s,
                                              min_samples=max(5, int(0.5 * self.cfg.calib_duration_s * self.cfg.control_hz)),
                                              max_mag_std_ut=self.cfg.calib_max_mag_std_ut)
        return True, self._say(f"calibrating: keep the hand still and unloaded for {self.cfg.calib_duration_s:.1f} s")

    def arm(self, link: dict, now_ns: int | None = None) -> tuple[bool, str]:
        if self.collector is not None:
            return False, self._say("arm refused: calibration in progress")
        if self.calib is None:
            return False, self._say("arm refused: not calibrated (press C with the hand still and unloaded)")
        if link.get("state") != "connected":
            return False, self._say("arm refused: no controller connected")
        if self._last_snap is not None:
            why = self.calib.mismatch(self._last_snap)
            if why:
                self.calib = None
                return False, self._say(f"arm refused: calibration invalid ({why}); recalibrate")
        if not self._input_ok(time.monotonic_ns() if now_ns is None else now_ns):
            return False, self._say(f"arm refused: input not usable ({self._input_reason})")
        self.state, self.reason = ARMED, None
        self._armed_epoch = link.get("epoch")
        self.v[:] = 0.0
        return True, self._say("ARMED (EXPERIMENTAL mapping)")

    def disarm(self, reason: str = "operator disarm") -> None:
        if self.state != DISARMED:
            self._say(f"DISARMED: {reason}")
        self.state, self.reason = DISARMED, reason
        self.v[:] = 0.0

    # -- per tick ------------------------------------------------------------

    def _input_ok(self, now_ns: int) -> bool:
        if self._last_usable_ns is None:
            return False
        if (now_ns - self._last_usable_ns) > self.cfg.watchdog_s * 1e9:
            return False
        return self._last_out is not None and self._last_out.usable

    def tick(self, seq: int, snap, link: dict, z_input: float = 0.0, now_ns: int | None = None) -> Command:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        dt = 1.0 / self.cfg.control_hz

        if snap is not None and seq != self._seen_seq:
            self._seen_seq = seq
            self._on_snapshot(snap)

        # connection loss and re-connection both require an explicit re-arm
        if self.state != DISARMED and (link.get("state") != "connected" or link.get("epoch") != self._armed_epoch):
            self.disarm("connection lost: re-arm required")

        if self.collector is not None:
            self._advance_calibration()

        ok = self._input_ok(now_ns)
        if not ok:
            age = None if self._last_usable_ns is None else (now_ns - self._last_usable_ns) / 1e6
            if age is not None and age > self.cfg.watchdog_s * 1e3:
                self._input_reason = f"watchdog: no usable control frame for {age:.0f} ms"
        if self.state == ARMED and not ok:
            self.state, self.reason = PAUSED, self._input_reason
            self._say(f"PAUSED: {self.reason}")
        elif self.state == PAUSED:
            if ok and self.cfg.resume_policy == "auto" and self._consecutive_usable >= self.cfg.resume_frames:
                self.state, self.reason = ARMED, None
                self._say("resumed (input usable again)")
            elif not ok:
                self.reason = self._input_reason

        if self.state == ARMED:
            self._drive(self._last_out, dt, z_input)
        else:
            self._hold()
        return self._command()

    # -- per-robot command shaping (override for other robots; see hand.py) --------

    def _drive(self, out, dt: float, z_input: float) -> None:
        """ARMED: move the bounded, smoothed command toward the mapping output."""
        v_target = np.array([out.v_xy[0], out.v_xy[1], out.vz + z_input * self.cfg.z_speed])
        v_target = np.clip(v_target, -self.cfg.max_speed, self.cfg.max_speed)
        a = min(1.0, dt / max(self.cfg.vel_tau_s, dt))
        v_new = self.v + a * (v_target - self.v)
        dv = np.clip(v_new - self.v, -self.cfg.max_accel * dt, self.cfg.max_accel * dt)
        self.v = np.clip(self.v + dv, -self.cfg.max_speed, self.cfg.max_speed)
        b = min(1.0, dt / max(self.cfg.open_tau_s, dt))
        want = self.opening + b * (out.opening - self.opening)
        step = float(np.clip(want - self.opening, -self.cfg.max_open_rate * dt, self.cfg.max_open_rate * dt))
        self.opening = float(np.clip(self.opening + step, 0.0, self.mapping.cfg.max_opening))

    def _hold(self) -> None:
        """Not ARMED: stop Cartesian motion now; hold the opening."""
        self.v = np.zeros(3)

    def _command(self) -> Command:
        return Command(self.v.copy(), self.opening, self.state, self.reason)

    def _on_snapshot(self, snap) -> None:
        self._last_snap = snap
        if self.calib is not None:
            why = self.calib.mismatch(snap)
            if why:
                self.calib = None
                self.disarm(f"calibration invalidated: {why}")
                self._say("recalibrate before arming")
        self._evaluate(snap)
        if self.collector is not None:
            try:
                self.collector.add(snap)
            except CalibrationError as e:
                self.collector = None
                self._say(f"calibration failed: {e}")

    def _evaluate(self, snap) -> None:
        out = self.mapping.evaluate(snap, self.calib)
        self._last_out = out
        if out.usable:
            self._last_usable_ns = snap.frame.t_recv_ns     # host monotonic receive time
            self._consecutive_usable += 1
            self._input_reason = None
        else:
            self._consecutive_usable = 0
            self._input_reason = out.reason

    def _advance_calibration(self) -> None:
        c = self.collector
        if c.ready():
            self.collector = None
            try:
                cal = c.result()
            except CalibrationError as e:
                self._say(f"calibration failed: {e}")
                return
            self.calib = cal
            if self._last_snap is not None:
                self._evaluate(self._last_snap)          # the pre-calibration result said "not calibrated"
            if self.calib_dir:
                p = cal.save(f"{self.calib_dir}/calibration-{time.strftime('%Y%m%d-%H%M%S')}.json")
                self._say(f"calibrated from {cal.samples} frames; saved {p}")
            else:
                self._say(f"calibrated from {cal.samples} frames")
        elif c.timed_out():
            self.collector = None
            self._say(f"calibration failed: only {len(c.snaps)} usable frames in {c.timeout_s:.0f} s "
                      f"(skipped: {c.skipped or 'none'})")

    def status(self, now_ns: int | None = None) -> ControllerStatus:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        age = None if self._last_usable_ns is None else round((now_ns - self._last_usable_ns) / 1e6, 1)
        return ControllerStatus(self.state, self.reason, self.calib is not None, self.collector is not None,
                                self._input_ok(now_ns), age,
                                None if self._last_snap is None else self._last_snap.frame.frame,
                                {} if self._last_out is None else self._last_out.signals, list(self.messages))
