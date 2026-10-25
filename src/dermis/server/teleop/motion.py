"""EXPERIMENTAL: short hand motions (e.g. a punch) from one IMU, with spring-back. Not tracking.

An IMU measures acceleration and rotation rate, never position. Integrating acceleration twice
turns a tiny bias into metres of drift within seconds, so this estimator deliberately only
reproduces *short, quick* motions and always pulls the hand back to its home pose:

    orientation   gyro-integrated (body rates), gently corrected toward the measured gravity
                  direction when the hand is quasi-static. Yaw is relative to calibration and
                  slowly drifts; nothing here is a compass.
    linear acc    a_world = R @ specific_force - g   (deadband removes noise and residual tilt error).
                  Quasi-static samples (|specific force| within 5 % of 1 g, gyro quiet) count as zero
                  linear acceleration, so slow tilting or a small orientation error cannot drift.
    velocity      trapezoid-integrated, leaky (time constant leak_v_s), and set to zero after
                  still_s of near-zero acceleration and rotation (zero-velocity update: a punch
                  momentarily stops at full extension and at home). At low frame rates a fast
                  motion is only a few samples, so send >= 100 Hz for motion; 30 Hz works roughly.
    position      integrated velocity x gain while moving; once the hand is still it springs
                  back to home (time constant spring_s). Clamped to `limits` around home.

Runs on EVERY accepted frame (TelemetryServer observer, network thread) using the device clock
for dt, because a 30 Hz sim sample of a ~250 ms punch would miss most of it. offset() is read
by the sim thread. Frame gaps longer than max_dt, clock resets or an unusable IMU zero the velocity,
and after a gap the orientation is re-levelled from gravity at once (the hand may have turned).

``free`` mode assumes the IMU chip's +x points along the fingers (forward) and its +z out of
the back of the hand. ``forward`` mode is intended for a repeatable demo: after stationary
calibration, press F and make one short forward jab. The strongest horizontal acceleration
defines forward; later translation is projected onto that direction and exposed only as the
simulator's X axis. It suppresses sideways / vertical drift, but is not 3-D tracking.
"""
from __future__ import annotations

import math
import threading

import numpy as np

G = 9.80665


def _rot(axis_angle: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation matrix for the rotation vector `axis_angle` (rad)."""
    th = float(np.linalg.norm(axis_angle))
    if th < 1e-12:
        return np.eye(3)
    k = axis_angle / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation vector that turns direction a onto direction b (minimal rotation)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    c = np.cross(a, b)
    s, d = float(np.linalg.norm(c)), float(np.clip(np.dot(a, b), -1, 1))
    if s < 1e-9:
        return np.zeros(3) if d > 0 else np.array([math.pi, 0, 0])
    return c / s * math.atan2(s, d)


class MotionEstimator:
    label = "EXPERIMENTAL IMU motion bursts (spring-back, not position tracking)"

    def __init__(self, pods=("wrist", "index_tip"), gain: float = 1.0, leak_v_s: float = 2.0,
                 spring_s: float = 0.35, acc_deadband: float = 0.8, tilt_gain: float = 0.06,
                 limits=((-0.05, 0.30), (-0.12, 0.12), (-0.10, 0.12)), max_dt: float = 0.25,
                 still_s: float = 0.20, mode: str = "free", rotation_gate_rads: float = 1.2,
                 axis_capture_s: float = 1.5):
        if mode not in {"free", "forward"}:
            raise ValueError("motion mode must be 'free' or 'forward'")
        self.pods = tuple(pods)
        self.gain, self.leak_v_s, self.spring_s = gain, leak_v_s, spring_s
        self.acc_deadband, self.tilt_gain, self.max_dt, self.still_s = acc_deadband, tilt_gain, max_dt, still_s
        self.limits = np.array(limits, dtype=float)
        self.mode, self.rotation_gate_rads, self.axis_capture_s = mode, rotation_gate_rads, axis_capture_s
        self._lock = threading.Lock()
        self.pod: str | None = None
        self._ref = None
        self.R = np.eye(3)                   # chip -> world
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self._last_t: int | None = None
        self._last_a = np.zeros(3)
        self._acc_bias = np.zeros(3)
        self._still_t = 0.0
        self.gyr_bias = np.zeros(3)
        self.forward_axis: np.ndarray | None = None
        self._capture_left = 0.0
        self._capture_peak = np.zeros(3)
        self._capture_peak_norm = 0.0
        self._notices: list[str] = []
        self.peak = np.zeros(3)              # largest |offset| per axis (inspection)

    def set_reference(self, calib) -> None:
        """Use the calibration's rest gravity to define 'up'. Resets the motion state."""
        with self._lock:
            self._ref = calib
            self.pod = next((p for p in self.pods if p in calib.acc_baseline_mps2), None)
            if self.pod is None:
                return
            g0 = np.asarray(calib.acc_baseline_mps2[self.pod], dtype=float)
            self.gyr_bias = np.asarray(getattr(calib, "gyr_bias_rads", {}).get(self.pod, (0.0, 0.0, 0.0)), dtype=float)
            if self.gyr_bias.shape != (3,) or not np.all(np.isfinite(self.gyr_bias)):
                self.gyr_bias = np.zeros(3)
            self.R = _rot(_align(g0, np.array([0.0, 0.0, 1.0])))
            self.v[:] = 0.0
            self.p[:] = 0.0
            self._last_t = None
            self._last_a[:] = 0.0
            self.forward_axis = None
            self._acc_bias[:] = 0.0
            self._capture_left = 0.0

    def start_forward_axis_capture(self) -> tuple[bool, str]:
        """Begin a one-shot direction capture for the constrained demo mode."""
        with self._lock:
            if self.mode != "forward":
                return False, "forward-axis capture requires --motion-mode forward"
            if self.pod is None:
                return False, "forward-axis capture requires a completed stationary calibration"
            self.forward_axis = None
            self._capture_left = self.axis_capture_s
            self._capture_peak[:] = 0.0
            self._capture_peak_norm = 0.0
            self.v[:] = self.p[:] = self._last_a[:] = 0.0
            return True, (f"forward-axis capture started: make one short, straight forward jab within "
                          f"{self.axis_capture_s:.1f} s, then hold still")

    def pop_notices(self) -> list[str]:
        with self._lock:
            notices, self._notices = self._notices, []
            return notices

    def forward_capture_active(self) -> bool:
        with self._lock:
            return self._capture_left > 0.0

    def forward_axis_ready(self) -> bool:
        with self._lock:
            return self.forward_axis is not None

    def reference(self):
        return self._ref

    def update(self, snap) -> None:
        """Network thread: integrate one accepted frame."""
        with self._lock:
            if self.pod is None or self.pod not in snap.pods:
                return
            i = snap.pods.index(self.pod)
            if not snap.imu_fresh[i]:
                if not snap.imu_valid[i]:
                    self.v[:] = 0.0                      # unusable IMU: stop, let the spring bring it home
                return                                   # STALE repeats would double-count acceleration
            imu = snap.frame.pods[self.pod].imu
            t = imu.t_us
            dt = None if self._last_t is None else (t - self._last_t) / 1e6
            self._last_t = t
            # The telemetry layer already applies a short median window and low-pass filter.
            # Use those values here so isolated electrical spikes cannot become metres of
            # double-integrated position error.
            acc, gyr = snap.acc_filtered[i], snap.gyr_filtered[i]
            if not (np.all(np.isfinite(acc)) and np.all(np.isfinite(gyr))):
                self.v[:] = 0.0
                return
            bias = np.asarray(getattr(self._ref, "gyr_bias_rads", {}).get(self.pod, [0.0, 0.0, 0.0]))
            gyr = gyr - bias
            if dt is None or dt <= 0 or dt > self.max_dt:
                self.v[:] = 0.0                          # gap, restart or clock reset
                self._last_a[:] = 0.0
                if abs(float(np.linalg.norm(acc)) - G) < 0.1 * G:      # re-level: orientation may have changed
                    self.R = _rot(_align(self.R @ acc, np.array([0.0, 0.0, 1.0]))) @ self.R
                return
            self.R = self.R @ _rot(gyr * dt)
            na = float(np.linalg.norm(acc))
            if abs(na - G) < 0.1 * G and float(np.linalg.norm(gyr)) < 0.5:      # quasi-static: fix tilt drift
                up = self.R @ acc
                self.R = _rot(self.tilt_gain * _align(up, np.array([0.0, 0.0, 1.0]))) @ self.R
            residual = self.R @ acc - np.array([0.0, 0.0, G])
            quasi_static = abs(na - G) < 0.05 * G and float(np.linalg.norm(gyr)) < 0.5
            if quasi_static:
                # Learn only the small world-frame residual while stationary. This cancels
                # temperature/bias drift without absorbing deliberate motion.
                self._acc_bias += min(1.0, dt / 3.0) * (residual - self._acc_bias)
            a = residual - self._acc_bias
            # While rotating rapidly, accelerometer translation is ambiguous.
            # Suppress it instead of creating an unexpected diagonal trajectory.
            rotating = float(np.linalg.norm(gyr)) > self.rotation_gate_rads
            if quasi_static or rotating or float(np.linalg.norm(a)) < self.acc_deadband:
                a = np.zeros(3)
            if self._capture_left > 0.0:
                horizontal = a.copy()
                horizontal[2] = 0.0
                strength = float(np.linalg.norm(horizontal))
                if strength > self._capture_peak_norm:
                    self._capture_peak_norm, self._capture_peak = strength, horizontal
                self._capture_left -= dt
                if self._capture_left <= 0.0:
                    if self._capture_peak_norm >= max(1.2, self.acc_deadband):
                        self.forward_axis = self._capture_peak / self._capture_peak_norm
                        self._notices.append("forward axis calibrated; translation is locked to simulator X")
                    else:
                        self._notices.append("forward-axis capture failed: no clear acceleration; press F and jab once")
                    self.v[:] = self.p[:] = self._last_a[:] = 0.0
                a = np.zeros(3)              # calibration gesture must not move the simulation
            elif self.mode == "forward":
                if self.forward_axis is None:
                    a = np.zeros(3)
                else:
                    # The learned physical direction becomes exactly simulator +X.
                    a = np.array([float(np.dot(a, self.forward_axis)), 0.0, 0.0])
            still = not a.any() and float(np.linalg.norm(gyr)) < 0.3
            self._still_t = self._still_t + dt if still else 0.0
            self.v = (self.v + 0.5 * (a + self._last_a) * dt) * math.exp(-dt / self.leak_v_s)
            self._last_a = a
            if self._still_t >= self.still_s:
                self.v[:] = 0.0                          # zero-velocity update
            self.p = self.p + self.gain * self.v * dt
            if self._still_t >= self.still_s:
                self.p *= math.exp(-dt / self.spring_s)  # still: return home
            self.p = np.clip(self.p, self.limits[:, 0], self.limits[:, 1])
            self.peak = np.maximum(self.peak, np.abs(self.p))

    def offset(self) -> np.ndarray:
        with self._lock:
            return self.p.copy()

    def orientation_rpy(self) -> np.ndarray:
        """Filtered full 3-D orientation as roll/pitch/yaw relative to calibration.

        Roll/pitch are gravity-corrected. Yaw comes from the gyro and is necessarily relative
        and slowly drifting because the glove has no magnetometer heading reference.
        """
        with self._lock:
            R = self.R.copy()
        pitch = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
        if abs(math.cos(pitch)) < 1e-6:
            roll = math.atan2(-R[0, 1], R[1, 1])
            yaw = 0.0
        else:
            roll = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(R[1, 0], R[0, 0])
        return np.array([roll, pitch, yaw], dtype=float)
