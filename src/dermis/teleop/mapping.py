"""Teleoperation mappings for `demo --source glove`: telemetry Snapshot -> robot action [vx, vy, vz, opening, effort].

The interface (TeleopMapping) is stable; the only concrete glove mapping here is EXPERIMENTAL
and refuses to construct unless explicitly enabled. It is a placeholder to exercise the
integration path, not a validated controller. (The TCP teleop path has its own, separate
mapping and armed/disarmed controller in server/teleop/{mapping,controller}.py.)

Safety behaviour shared by all mappings (TeleopMapping.__call__):
    * no accepted frame for source_timeout_s (wall clock)       -> PAUSED_NO_DATA
    * a required channel neither fresh nor within max hold     -> PAUSED_STALE
    * calibration not valid for the frame's device/config      -> PAUSED_CONFIG
    In every paused state the action is NEUTRAL: zero velocity, opening/effort held at their
    last commanded values. Paused ticks are recorded with their status.

Position: this module never integrates IMU acceleration into position (double integration
drifts within seconds). Absolute hand/object position must come from an external tracker
through a PoseProvider.
"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from ..sim.env import MAX_OPENING
from ..telemetry.calibration import Calibration
from ..telemetry.pipeline import Snapshot

ACTIVE, PAUSED_NO_DATA, PAUSED_STALE, PAUSED_CONFIG = "ACTIVE", "PAUSED_NO_DATA", "PAUSED_STALE", "PAUSED_CONFIG"


class PoseProvider(Protocol):
    """Hook for external hand/object tracking (camera, mocap). Returns a 4x4 world pose or None."""

    def hand_pose(self, t_host_ns: int) -> Optional[np.ndarray]: ...

    def object_pose(self, t_host_ns: int) -> Optional[np.ndarray]: ...


class NullPoseProvider:
    def hand_pose(self, t_host_ns):
        return None

    def object_pose(self, t_host_ns):
        return None


class TeleopMapping:
    required_mag: tuple = ()
    required_imu: tuple = ()

    def __init__(self, calibration: Calibration | None = None, default_effort: float = 15.0):
        self.calibration = calibration
        self.last_opening, self.last_effort = MAX_OPENING, default_effort
        self.status = PAUSED_NO_DATA

    def neutral(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, self.last_opening, self.last_effort], np.float32)

    def check(self, snap: Snapshot | None, source_stale: bool) -> str:
        if snap is None or source_stale:
            return PAUSED_NO_DATA
        idx = {p: i for i, p in enumerate(snap.pods)}
        for p in self.required_mag:
            if p not in idx or not snap.mag_held_mask[idx[p]]:
                return PAUSED_STALE
        for p in self.required_imu:
            if p not in idx or not snap.imu_held_mask[idx[p]]:
                return PAUSED_STALE
        if self.required_mag and self.calibration is not None:
            if (snap.frame.device_id, snap.frame.config_id) != (self.calibration.device_id, self.calibration.config_id):
                return PAUSED_CONFIG
        return ACTIVE

    def __call__(self, snap: Snapshot | None, source_stale: bool) -> tuple[np.ndarray, str]:
        self.status = self.check(snap, source_stale)
        if self.status != ACTIVE:
            return self.neutral(), self.status
        a = np.asarray(self.map(snap), np.float32)
        self.last_opening, self.last_effort = float(a[3]), float(a[4])
        return a, ACTIVE

    def map(self, snap: Snapshot) -> np.ndarray:
        raise NotImplementedError


class ExperimentalTiltGripMapping(TeleopMapping):
    """EXPERIMENTAL. Hand tilt (thenar IMU gravity direction vs. a captured rest pose) -> x/y velocity;
    mean normalized magnetic deformation of index+thumb pads -> open/close. z velocity is not
    mapped (keyboard E/Q in the GUI). Gains, deadbands and the threshold are guesses."""

    EXPERIMENTAL = True
    required_imu = ("thenar_apb",)
    required_mag = ("index_tip", "thumb_tip")

    def __init__(self, calibration: Calibration, enabled: bool = False, gain_mps_per_rad: float = 0.25,
                 deadband_rad: float = 0.08, max_speed: float = 0.12, grip_threshold: float = 0.3,
                 effort_n: float = 15.0):
        if not enabled:
            raise PermissionError("ExperimentalTiltGripMapping is experimental; enable it deliberately "
                                  "(--experimental-glove-teleop)")
        if calibration is None:
            raise ValueError("the grip mapping needs a magnetic baseline calibration (python -m dermis calibrate)")
        super().__init__(calibration, effort_n)
        self.gain, self.deadband, self.max_speed = gain_mps_per_rad, deadband_rad, max_speed
        self.grip_threshold, self.effort = grip_threshold, effort_n
        self.rest: np.ndarray | None = None

    def capture_rest(self, snap: Snapshot) -> bool:
        i = snap.pods.index("thenar_apb")
        if not snap.imu_held_mask[i]:
            return False
        self.rest = snap.acc_held[i] / np.linalg.norm(snap.acc_held[i])
        return True

    def map(self, snap: Snapshot) -> np.ndarray:
        i = snap.pods.index("thenar_apb")
        g = snap.acc_held[i] / max(1e-6, np.linalg.norm(snap.acc_held[i]))
        if self.rest is None:
            self.rest = g
        d = g - self.rest                         # ~ small-angle tilt about chip x / y
        tilt = np.where(np.abs(d[:2]) > self.deadband, d[:2] - np.sign(d[:2]) * self.deadband, 0.0)
        vxy = np.clip(self.gain * tilt, -self.max_speed, self.max_speed)
        cal = self.calibration.apply(snap)
        k = [snap.pods.index(p) for p in self.required_mag]
        deform = float(np.mean(cal["deformation"][k]))
        opening = 0.0 if deform >= self.grip_threshold else MAX_OPENING
        return np.array([vxy[0], vxy[1], 0.0, opening, self.effort], np.float32)
