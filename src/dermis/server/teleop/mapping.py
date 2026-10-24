"""EXPERIMENTAL telemetry -> gripper command mappings for TELEOPERATION tests.

A mapping turns one pipeline Snapshot plus a Calibration into a MappingOutput. It is the only
place that decides what the sensors mean; replace it (implement the Mapping protocol) without
touching the TCP server, the pipeline or the simulator.

MagGripTiltMapping (provisional, development only):
    grip    baseline-subtracted magnetic change on the selected pods along one chip axis,
            averaged, times sign and gain, saturated -> closure 0..1 -> opening.
            This is a change in microtesla, NOT a calibrated force.
    tilt    optional: the selected pod's gravity direction (accelerometer, quasi-static only)
            relative to its calibrated rest direction -> bounded XY *velocity*.
            Acceleration is never integrated into position, and samples whose |a| is far from
            1 g (the hand is accelerating) produce zero velocity rather than a guess.
    z       not mapped by default (fixed, or keyboard E/Q). With pitch_axis="z" (--tilt-z) the
            pod's pitch drives a bounded Z *velocity* instead of X: tip the finger to move the arm
            down / up. Same deadband, saturation and quasi-static check as XY; never integrated
            from acceleration.
    The tactile magnetometers are never used as a heading reference.

Channel usability uses the pipeline's bounded hold: *_held is the last fresh value if it is at
most max_hold_us old on the device clock (so a briefly STALE reading is accepted), else NaN.
TEMP_STALE does not affect field or IMU usability.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from ...telemetry.pipeline import Snapshot

G = 9.80665
MAX_OPENING = 0.08
LABEL = "EXPERIMENTAL"


@dataclass
class MappingConfig:
    grip_pods: tuple = ("index_tip", "thumb_tip")
    grip_axis: int = 2                    # chip axis of b_ut: 0=x, 1=y, 2=z
    grip_sign: float = 1.0                # +1: field increase along the axis closes the gripper
    grip_gain_per_ut: float = 1.0 / 6000.0
    grip_deadband_ut: float = 150.0
    grip_saturation_ut: float = 8000.0    # |change| beyond this is clamped before the gain
    max_opening: float = MAX_OPENING
    tilt_xy: bool = False
    tilt_pod: str = "index_tip"
    tilt_gain_mps_per_rad: float = 0.12
    tilt_deadband_rad: float = 0.05
    tilt_max_rad: float = 0.45
    tilt_sign_x: float = 1.0
    tilt_sign_y: float = 1.0
    pitch_axis: str = "x"                 # "x": pitch -> vx (default); "z": pitch -> vz (arm down/up)
    tilt_sign_z: float = 1.0
    gravity_tolerance: float = 0.15       # accept |a| within (1 +- tol) g as quasi-static

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grip_pods"] = list(self.grip_pods)
        d["label"] = LABEL
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MappingConfig":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "grip_pods" in d:
            d["grip_pods"] = tuple(d["grip_pods"])
        return cls(**d)


@dataclass
class MappingOutput:
    usable: bool
    reason: str | None                    # why not usable
    opening: float | None                 # desired total finger opening, m
    v_xy: tuple                           # desired XY velocity, m/s (unbounded here; the controller clamps)
    signals: dict                         # inspection values (closure, delta_ut, tilt_rad, ...)
    vz: float = 0.0                       # desired Z velocity, m/s (only with pitch_axis="z")


class Mapping(Protocol):
    label: str

    def required_channels(self) -> list[tuple[str, str]]: ...
    def check_channels(self, snap: Snapshot) -> str | None: ...
    def evaluate(self, snap: Snapshot, calib) -> MappingOutput: ...


def _pod_index(snap: Snapshot, pod: str) -> int | None:
    try:
        return snap.pods.index(pod)
    except ValueError:
        return None


def _shape(x: float, deadband: float, limit: float) -> float:
    """Deadband then saturate, preserving sign."""
    m = min(max(abs(x) - deadband, 0.0), max(limit - deadband, 0.0))
    return math.copysign(m, x)


def gravity_tilt(acc: np.ndarray, rest: np.ndarray) -> tuple[float, float]:
    """(pitch, roll) change in rad of the gravity direction `acc` relative to `rest` (chip axes).
    pitch: rotation that moves gravity into chip x; roll: rotation of gravity about chip x."""
    a = acc / np.linalg.norm(acc)
    r = rest / np.linalg.norm(rest)
    pitch = math.asin(max(-1.0, min(1.0, -a[0]))) - math.asin(max(-1.0, min(1.0, -r[0])))
    roll = math.atan2(a[1], a[2]) - math.atan2(r[1], r[2])
    roll = (roll + math.pi) % (2 * math.pi) - math.pi
    return pitch, roll


class MagGripTiltMapping:
    label = f"{LABEL}: magnetic-change grip"

    def __init__(self, cfg: MappingConfig | None = None):
        self.cfg = cfg or MappingConfig()
        if self.cfg.tilt_xy:
            self.label = (f"{LABEL}: magnetic-change grip + IMU-tilt "
                          + ("Z (pitch) / Y (roll) velocity" if self.cfg.pitch_axis == "z" else "XY velocity"))

    def required_channels(self) -> list[tuple[str, str]]:
        ch = [("mag", p) for p in self.cfg.grip_pods]
        if self.cfg.tilt_xy:
            ch.append(("imu", self.cfg.tilt_pod))
        return ch

    def check_channels(self, snap: Snapshot, require_fresh: bool = False) -> str | None:
        """None if every required channel is usable in this snapshot, else the first reason."""
        for kind, pod in self.required_channels():
            i = _pod_index(snap, pod)
            if i is None or not snap.present[i]:
                return f"{pod} missing"
            if kind == "mag":
                if require_fresh and not snap.mag_fresh[i]:
                    return f"{pod} mag not fresh" + (" (STALE)" if snap.mag_stale[i] else " (invalid/null)")
                if not snap.mag_held_mask[i] or not np.all(np.isfinite(snap.mag_held[i])):
                    why = "STALE too long" if snap.mag_stale[i] else "invalid/null"
                    return f"{pod} mag unusable ({why})"
            else:
                if require_fresh and not snap.imu_fresh[i]:
                    return f"{pod} imu not fresh" + (" (STALE)" if snap.imu_stale[i] else " (invalid/null)")
                if not snap.imu_held_mask[i] or not np.all(np.isfinite(snap.acc_held[i])):
                    why = "STALE too long" if snap.imu_stale[i] else "invalid/null"
                    return f"{pod} imu unusable ({why})"
        return None

    def evaluate(self, snap: Snapshot, calib) -> MappingOutput:
        c = self.cfg
        why = self.check_channels(snap)
        if why:
            return MappingOutput(False, why, None, (0.0, 0.0), {})
        if calib is None:
            return MappingOutput(False, "not calibrated", None, (0.0, 0.0), {})
        deltas = []
        for pod in c.grip_pods:
            i = _pod_index(snap, pod)
            deltas.append(float(snap.mag_held[i][c.grip_axis] - calib.mag_baseline_ut[pod][c.grip_axis]))
        delta = float(np.mean(deltas))
        drive = max(0.0, min(c.grip_sign * delta - c.grip_deadband_ut, c.grip_saturation_ut))
        closure = max(0.0, min(1.0, c.grip_gain_per_ut * drive))
        opening = c.max_opening * (1.0 - closure)
        signals = {"delta_ut": round(delta, 1), "closure": round(closure, 3)}

        v, vz = (0.0, 0.0), 0.0
        if c.tilt_xy:
            i = _pod_index(snap, c.tilt_pod)
            acc = snap.acc_held[i]
            g = float(np.linalg.norm(acc))
            if abs(g - G) <= c.gravity_tolerance * G:
                pitch, roll = gravity_tilt(acc, np.asarray(calib.acc_baseline_mps2[c.tilt_pod]))
                lim = c.tilt_max_rad
                vp = c.tilt_gain_mps_per_rad * _shape(pitch, c.tilt_deadband_rad, lim)
                vy = c.tilt_sign_y * c.tilt_gain_mps_per_rad * _shape(roll, c.tilt_deadband_rad, lim)
                if c.pitch_axis == "z":
                    v, vz = (0.0, vy), c.tilt_sign_z * vp
                else:
                    v = (c.tilt_sign_x * vp, vy)
                signals.update(pitch_rad=round(pitch, 3), roll_rad=round(roll, 3))
            else:
                signals["tilt"] = f"|a|={g:.2f} m/s2 not quasi-static: XY velocity 0"
        return MappingOutput(True, None, opening, v, signals, vz)
