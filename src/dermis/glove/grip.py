"""EXPERIMENTAL glove grip control: magnetic pad change -> grip command, with two-point calibration,
explicit arming, watchdog and bounded output.

What this is: a *calibrated, configurable mapping* from the change in fingertip magnetic field
between two operator-recorded reference states (UNLOADED and a COMFORTABLE PRESS) to a unitless
closure 0..1, then to a bounded robot command (grip effort in N for fragile_cup, or opening).
What it is not: a force measurement (no reference load calibration exists), a slip detector, a
finger-joint estimate, or a position estimate (no IMU integration happens anywhere here).

Config (GripMappingConfig, YAML in configs/glove/grip.yaml):
    required_pods   pods whose usable magnetometer is needed; any unusable -> PAUSED (output held)
    optional_pods   used when usable, ignored when missing/invalid (never pause control)
    axis            "auto" (direction unloaded->press from the calibration), "x"/"y"/"z" (one chip axis),
                    or "norm" (|dB|)
    sign            +1 / -1, applied after projection (flip if pressing *reduces* the signal)
    gain, deadband  on the normalised closure (0 = unloaded reference, 1 = press reference)
    filter_tau_s    exponential smoothing of the closure
    spike_ut        a sample jumping more than this from the last accepted value is rejected
    out_min/out_max output limits; max_rate (output units / s) slew limit
    min_separation  calibration is rejected unless |press - unloaded| >= min_separation x
                    the larger stationary std of either reference, on every required pod
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..telemetry.pipeline import Snapshot

DISARMED, ARMED, PAUSED = "DISARMED", "ARMED", "PAUSED"
CAL_VERSION = 1


@dataclass
class GripMappingConfig:
    required_pods: tuple = ("index_tip", "thumb_tip")
    optional_pods: tuple = ()
    axis: str = "auto"
    sign: float = 1.0
    gain: float = 1.0
    deadband: float = 0.05
    filter_tau_s: float = 0.08
    spike_ut: float = 4000.0
    out_min: float = 2.0
    out_max: float = 20.0
    max_rate: float = 40.0
    combine: str = "mean"                   # mean | max over usable pods
    min_separation: float = 5.0
    max_reference_std_ut: float = 60.0      # a reference capture noisier than this is not stationary
    min_reference_samples: int = 20
    watchdog_s: float = 0.25                # no accepted frame for this long -> PAUSED
    disarm_after_s: float = 2.0             # still nothing after this -> DISARMED (connection lost)
    label: str = "EXPERIMENTAL: magnetic change -> grip; not force"

    @classmethod
    def from_dict(cls, d: dict | None) -> "GripMappingConfig":
        d = dict(d or {})
        for k in ("required_pods", "optional_pods"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path) -> "GripMappingConfig":
        from ..config import load_config
        return cls.from_dict(load_config(path))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["required_pods"], d["optional_pods"] = list(self.required_pods), list(self.optional_pods)
        return d


class CalibrationRejected(ValueError):
    def __init__(self, reasons: list[str], report: dict):
        super().__init__("; ".join(reasons))
        self.reasons, self.report = reasons, report


@dataclass
class TwoPointCalibration:
    device_id: str
    config_id: int
    pod_ids: dict
    pods: dict                  # loc -> {unloaded_mean, unloaded_std, press_mean, press_std, n_unloaded, n_press, separation}
    created_unix: float
    mapping: dict
    version: int = CAL_VERSION
    label: str = "EXPERIMENTAL two-point magnetic reference (unloaded / comfortable press); not force"
    calibration_id: str = ""

    def __post_init__(self):
        if not self.calibration_id:
            import hashlib
            body = json.dumps([self.device_id, self.config_id, self.pods, self.created_unix], sort_keys=True)
            self.calibration_id = "grip-" + hashlib.sha256(body.encode()).hexdigest()[:10]

    def mismatch(self, snap: Snapshot) -> str | None:
        f = snap.frame
        if f.device_id != self.device_id:
            return f"device changed ({self.device_id} -> {f.device_id})"
        if f.config_id != self.config_id:
            return f"config_id changed ({self.config_id} -> {f.config_id})"
        for loc, pid in self.pod_ids.items():
            pod = f.pods.get(loc)
            if pod is not None and pid and pod.pod_id != pid:
                return f"{loc} pod_id changed ({pid} -> {pod.pod_id})"
        return None

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))
        return path

    @classmethod
    def load(cls, path) -> "TwoPointCalibration":
        d = json.loads(Path(path).read_text())
        if d.get("version") != CAL_VERSION:
            raise ValueError(f"unsupported grip calibration version {d.get('version')}")
        return cls(**d)


class ReferenceCapture:
    """Collect FRESH magnetometer samples for one reference state. STALE / invalid / null are skipped."""

    def __init__(self, state: str, pods):
        self.state, self.pods = state, tuple(pods)
        self.samples: dict = {p: [] for p in self.pods}
        self.device_id = self.config_id = None
        self.pod_ids: dict = {}
        self.mixed = False

    def add(self, snap: Snapshot) -> None:
        f = snap.frame
        if self.device_id is not None and (f.device_id, f.config_id) != (self.device_id, self.config_id):
            self.mixed = True
        self.device_id, self.config_id = f.device_id, f.config_id
        for p in self.pods:
            if p in snap.pods:
                i = snap.pods.index(p)
                if snap.mag_fresh[i]:
                    self.samples[p].append(snap.mag_raw[i].copy())
                    if p in f.pods:
                        self.pod_ids[p] = f.pods[p].pod_id

    def stats(self) -> dict:
        out = {}
        for p, s in self.samples.items():
            a = np.array(s).reshape(-1, 3)
            out[p] = {"n": len(a), "mean": a.mean(0).tolist() if len(a) else None,
                      "std": a.std(0).tolist() if len(a) else None}
        return out


def build_calibration(unloaded: ReferenceCapture, press: ReferenceCapture, cfg: GripMappingConfig) -> TwoPointCalibration:
    """Accept or reject. Rejection lists every reason and returns the per-pod numbers."""
    reasons, report, pods = [], {}, {}
    if unloaded.mixed or press.mixed:
        reasons.append("device/config changed during a reference capture")
    if (unloaded.device_id, unloaded.config_id) != (press.device_id, press.config_id):
        reasons.append(f"captures come from different device/config: {unloaded.device_id}/{unloaded.config_id} "
                       f"vs {press.device_id}/{press.config_id}")
    su, sp = unloaded.stats(), press.stats()
    for p in cfg.required_pods + cfg.optional_pods:
        required = p in cfg.required_pods
        u, q = su.get(p, {"n": 0}), sp.get(p, {"n": 0})
        r = {"required": required, "n_unloaded": u["n"], "n_press": q["n"]}
        problems = []
        if u["n"] < cfg.min_reference_samples or q["n"] < cfg.min_reference_samples:
            problems.append(f"{p}: too few fresh samples (unloaded {u['n']}, press {q['n']}, "
                            f"need {cfg.min_reference_samples})")
        else:
            um, us, pm, ps = (np.array(u["mean"]), np.array(u["std"]), np.array(q["mean"]), np.array(q["std"]))
            noise = float(max(us.max(), ps.max()))
            dist = float(np.linalg.norm(pm - um))
            sep = dist / max(noise, 1e-6)
            r.update(unloaded_mean=um.round(3).tolist(), unloaded_std=us.round(3).tolist(), press_mean=pm.round(3).tolist(),
                     press_std=ps.round(3).tolist(), change_ut=round(dist, 2), stationary_std_ut=round(noise, 3),
                     separation=round(sep, 2))
            if us.max() > cfg.max_reference_std_ut:
                problems.append(f"{p}: unloaded reference not stationary (std {us.max():.1f} uT > {cfg.max_reference_std_ut})")
            if ps.max() > cfg.max_reference_std_ut:
                problems.append(f"{p}: press reference not stationary (std {ps.max():.1f} uT > {cfg.max_reference_std_ut})")
            if sep < cfg.min_separation:
                problems.append(f"{p}: insufficient separation {sep:.1f} < {cfg.min_separation} "
                                f"(change {dist:.1f} uT vs stationary std {noise:.1f} uT)")
            if cfg.axis in ("x", "y", "z"):
                k = "xyz".index(cfg.axis)
                ax_sep = abs(pm[k] - um[k]) / max(noise, 1e-6)
                r["axis_separation"] = round(float(ax_sep), 2)
                if ax_sep < cfg.min_separation:
                    problems.append(f"{p}: selected axis {cfg.axis} separates only {ax_sep:.1f}")
            if not problems:
                pods[p] = {"unloaded_mean": um.tolist(), "unloaded_std": us.tolist(), "press_mean": pm.tolist(),
                           "press_std": ps.tolist(), "n_unloaded": u["n"], "n_press": q["n"], "separation": sep}
        r["accepted"] = not problems
        report[p] = r
        if problems and required:
            reasons += problems
        elif problems:
            r["note"] = "optional pod left out of the calibration"
    if reasons:
        raise CalibrationRejected(reasons, report)
    return TwoPointCalibration(unloaded.device_id, unloaded.config_id,
                               {p: unloaded.pod_ids.get(p, "") for p in pods}, pods, time.time(), cfg.to_dict())


@dataclass
class GripCommand:
    value: float                 # bounded output (effort N or opening m)
    closure: float | None        # filtered closure 0..1 (None if no usable input)
    state: str
    reason: str
    pods_used: list = field(default_factory=list)
    spikes_rejected: int = 0


class GripControl:
    """Arming state machine + mapping. tick() is called once per control tick with the latest snapshot."""

    def __init__(self, cfg: GripMappingConfig, calibration: TwoPointCalibration | None = None, initial: float | None = None):
        self.cfg, self.cal = cfg, calibration
        self.state, self.reason = DISARMED, "not armed"
        self.value = float(initial if initial is not None else cfg.out_min)
        self.closure_f: float | None = None
        self.last_good: dict = {}
        self.spikes = 0

    # -- operator actions --------------------------------------------------------------
    def set_calibration(self, cal: TwoPointCalibration) -> tuple[bool, str]:
        if self.state != DISARMED:
            return False, "recalibration refused while armed; disarm first"
        self.cal, self.closure_f, self.last_good = cal, None, {}
        return True, f"calibration {cal.calibration_id} loaded"

    def arm(self, snap: Snapshot | None, source_stale: bool) -> tuple[bool, str]:
        if self.cal is None:
            return False, "no accepted calibration"
        if snap is None or source_stale:
            return False, "no live telemetry"
        mm = self.cal.mismatch(snap)
        if mm:
            return False, f"calibration does not match the stream: {mm}"
        r = self._required_unusable(snap)
        if r:
            return False, f"required channel unusable: {r}"
        self.state, self.reason = ARMED, "armed by operator"
        return True, "ARMED"

    def disarm(self, reason: str = "operator disarm") -> None:
        self.state, self.reason = DISARMED, reason

    # -- per tick -----------------------------------------------------------------------
    def _required_unusable(self, snap: Snapshot) -> str | None:
        for p in self.cfg.required_pods:
            if p not in snap.pods:
                return f"{p} not in layout"
            i = snap.pods.index(p)
            if not snap.present[i]:
                return f"{p} missing"
            if not snap.mag_held_mask[i] or not np.isfinite(snap.mag_held[i]).all():
                return f"{p} " + ("STALE beyond hold" if snap.mag_stale[i] else "invalid/null")
        return None

    def _pod_closure(self, snap: Snapshot, p: str) -> float | None:
        if p not in snap.pods or p not in self.cal.pods:
            return None
        i = snap.pods.index(p)
        if not snap.mag_held_mask[i] or not np.isfinite(snap.mag_held[i]).all():
            return None
        b = snap.mag_held[i]
        prev = self.last_good.get(p)
        if prev is not None and np.linalg.norm(b - prev) > self.cfg.spike_ut:
            self.spikes += 1
            return None                           # spike: rejected, not averaged in
        self.last_good[p] = b.copy()
        c = self.cal.pods[p]
        u0, p1 = np.array(c["unloaded_mean"]), np.array(c["press_mean"])
        ax = self.cfg.axis
        if ax == "auto":
            d = p1 - u0
            s = float((b - u0) @ d / (d @ d))
        elif ax == "norm":
            s = float(np.linalg.norm(b - u0) / np.linalg.norm(p1 - u0))
        else:
            k = "xyz".index(ax)
            s = float((b[k] - u0[k]) / (p1[k] - u0[k]))
        return self.cfg.sign * s

    def tick(self, snap: Snapshot | None, now_ns: int | None = None, dt: float = 1 / 30) -> GripCommand:
        cfg = self.cfg
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        age = None if snap is None else (now_ns - snap.frame.t_recv_ns) / 1e9
        # watchdog
        if snap is None or age > cfg.watchdog_s:
            if self.state == ARMED:
                self.state, self.reason = PAUSED, "no telemetry (watchdog)"
            if self.state == PAUSED and (snap is None or age > cfg.disarm_after_s):
                self.state, self.reason = DISARMED, "connection lost: re-arm required"
            return GripCommand(self.value, self.closure_f, self.state, self.reason, [], self.spikes)
        if self.cal is not None and self.state != DISARMED:
            mm = self.cal.mismatch(snap)
            if mm:
                self.state, self.reason = DISARMED, f"calibration invalid: {mm}"
        if self.state == DISARMED:
            return GripCommand(self.value, self.closure_f, self.state, self.reason, [], self.spikes)
        req = self._required_unusable(snap)
        if req:
            self.state, self.reason = PAUSED, f"required channel unusable: {req}"
            return GripCommand(self.value, self.closure_f, self.state, self.reason, [], self.spikes)
        vals, used = [], []
        for p in cfg.required_pods + cfg.optional_pods:
            v = self._pod_closure(snap, p)       # None: optional pod unusable, or a rejected spike
            if v is None:
                continue
            vals.append(v)
            used.append(p)
        if not any(p in used for p in cfg.required_pods):
            if self.state == ARMED:
                self.reason = "holding (spike rejected)"
            return GripCommand(self.value, self.closure_f, self.state, self.reason, used, self.spikes)
        if self.state == PAUSED:                  # input usable again: resume (re-arm not needed for a brief pause)
            self.state, self.reason = ARMED, "resumed"
        raw = max(vals) if cfg.combine == "max" else float(np.mean(vals))
        x = raw - math.copysign(min(abs(raw), cfg.deadband), raw)
        closure = float(np.clip(cfg.gain * x / max(1e-9, 1 - cfg.deadband), 0.0, 1.0))
        a = 1.0 - math.exp(-dt / max(cfg.filter_tau_s, 1e-6))
        self.closure_f = closure if self.closure_f is None else self.closure_f + a * (closure - self.closure_f)
        target = cfg.out_min + self.closure_f * (cfg.out_max - cfg.out_min)
        step = cfg.max_rate * dt
        self.value = float(np.clip(np.clip(target, self.value - step, self.value + step), cfg.out_min, cfg.out_max))
        return GripCommand(self.value, self.closure_f, self.state, self.reason, used, self.spikes)
