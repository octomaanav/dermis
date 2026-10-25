"""Explicit, operator-triggered baseline capture for the EXPERIMENTAL teleop mapping.

The operator holds the hand still and unloaded (nothing touching the pads) and triggers
calibration. The collector accepts only snapshots whose required channels are FRESH (STALE,
null and invalid readings are skipped, never averaged in), then checks the capture was
stationary. The result is tied to device_id, config_id and the pod IDs; any change to those
invalidates it. Calibration is never re-run automatically, and is refused while armed.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ...telemetry.pipeline import Snapshot


class CalibrationError(RuntimeError):
    pass


@dataclass
class Calibration:
    device_id: str
    config_id: int
    pod_ids: dict                       # location -> pod_id
    mag_baseline_ut: dict               # location -> [3] mean field, uT (chip axes)
    mag_std_ut: dict
    acc_baseline_mps2: dict             # location -> [3] mean specific force (gravity at rest)
    gyr_rms_rads: dict
    samples: int
    duration_s: float
    created_unix: float
    source: str
    mapping: dict
    gyr_bias_rads: dict = field(default_factory=dict)  # location -> stationary gyro median, rad/s
    label: str = "EXPERIMENTAL baseline (unloaded, stationary); magnetic change is not force"
    # Default keeps pre-bias-calibration JSON files loadable.
    gyr_bias_rads: dict = field(default_factory=dict)
    path: str | None = field(default=None, compare=False)

    def mismatch(self, snap: Snapshot) -> str | None:
        """Why this calibration no longer applies to the stream, or None."""
        f = snap.frame
        if f.device_id != self.device_id:
            return f"device changed ({self.device_id} -> {f.device_id})"
        if f.config_id != self.config_id:
            return f"config_id changed ({self.config_id} -> {f.config_id})"
        for loc, pid in self.pod_ids.items():
            pod = f.pods.get(loc)
            if pod is not None and pod.pod_id != pid:
                return f"{loc} pod_id changed ({pid} -> {pod.pod_id})"
        return None

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d.pop("path")
        path.write_text(json.dumps(d, indent=1))
        self.path = str(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        d = json.loads(Path(path).read_text())
        d.pop("path", None)
        c = cls(**d)
        c.path = str(path)
        return c


class CalibrationCollector:
    def __init__(self, mapping, duration_s: float = 1.0, min_samples: int = 15, max_mag_std_ut: float = 80.0,
                 max_gyro_rms_rads: float = 0.35, timeout_s: float = 5.0):
        self.mapping = mapping
        self.duration_s, self.min_samples, self.timeout_s = duration_s, min_samples, timeout_s
        self.max_mag_std_ut, self.max_gyro_rms_rads = max_mag_std_ut, max_gyro_rms_rads
        self.t0 = time.monotonic()
        self.snaps: list[Snapshot] = []
        self.skipped: dict[str, int] = {}
        self._first: Snapshot | None = None

    def add(self, snap: Snapshot) -> None:
        why = self.mapping.check_channels(snap, require_fresh=True)
        if why:
            self.skipped[why] = self.skipped.get(why, 0) + 1
            return
        if self._first is None:
            self._first = snap
        elif self._first.frame.device_id != snap.frame.device_id or self._first.frame.config_id != snap.frame.config_id:
            raise CalibrationError("device or configuration changed during calibration")
        self.snaps.append(snap)

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def ready(self) -> bool:
        return self.elapsed() >= self.duration_s and len(self.snaps) >= self.min_samples

    def timed_out(self) -> bool:
        return self.elapsed() >= self.timeout_s

    def result(self) -> Calibration:
        if len(self.snaps) < self.min_samples:
            raise CalibrationError(f"only {len(self.snaps)} usable frames (need {self.min_samples}); "
                                   f"skipped: {self.skipped or 'none'}")
        pods = self.snaps[0].pods
        f0 = self.snaps[0].frame
        mag_mean, mag_std, acc_mean, gyr_rms, gyr_bias = {}, {}, {}, {}, {}

        def robust_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Median and Gaussian-equivalent MAD, insensitive to isolated sensor spikes."""
            center = np.median(values, axis=0)
            sigma = 1.4826 * np.median(np.abs(values - center), axis=0)
            # Quantized quiet channels can have MAD=0. Keep a small floor without allowing
            # one outlier to inflate the accepted calibration noise.
            sigma = np.maximum(sigma, 1e-6)
            return center, sigma

        for kind, pod in self.mapping.required_channels():
            i = pods.index(pod)
            if kind == "mag":
                b = np.array([s.mag_raw[i] for s in self.snaps])
                center, sigma = robust_stats(b)
                mag_mean[pod], mag_std[pod] = center.tolist(), sigma.tolist()
                if max(mag_std[pod]) > self.max_mag_std_ut:
                    raise CalibrationError(f"{pod} field not steady (robust sigma {max(mag_std[pod]):.0f} uT > "
                                           f"{self.max_mag_std_ut:.0f}); keep the hand still and unloaded")
            else:
                a = np.array([s.acc_raw[i] for s in self.snaps])
                w = np.array([s.gyr_raw[i] for s in self.snaps])
                acc_center, _ = robust_stats(a)
                gyro_center, _ = robust_stats(w)
                acc_mean[pod] = acc_center.tolist()
                gyr_bias[pod] = gyro_center.tolist()
                residual = w - gyro_center
                gyr_rms[pod] = float(np.sqrt((residual ** 2).sum(1).mean()))
                if gyr_rms[pod] > self.max_gyro_rms_rads:
                    raise CalibrationError(f"{pod} moving (gyro residual rms {gyr_rms[pod]:.2f} rad/s); hold still")
        # tilt needs a rest gravity direction even when only the grip channels are required
        cfg = self.mapping.cfg
        for pod in (*getattr(cfg, "grip_pods", ()), *getattr(cfg, "optional_imu_pods", ())):
            if pod not in pods or pod in acc_mean:
                continue
            i = pods.index(pod)
            a = np.array([s.acc_raw[i] for s in self.snaps])
            if np.all(np.isfinite(a)):                     # only pods whose IMU was usable throughout
                acc_mean[pod] = robust_stats(a)[0].tolist()
                w = np.array([s.gyr_raw[i] for s in self.snaps])
                if np.all(np.isfinite(w)):
                    gyro_center = robust_stats(w)[0]
                    gyr_bias[pod] = gyro_center.tolist()
                    residual = w - gyro_center
                    gyr_rms[pod] = float(np.sqrt((residual ** 2).sum(1).mean()))
        pod_ids = {loc: r.pod_id for loc, r in f0.pods.items() if loc in {p for _, p in self.mapping.required_channels()}}
        return Calibration(f0.device_id, f0.config_id, pod_ids, mag_mean, mag_std, acc_mean, gyr_rms,
                           len(self.snaps), round(self.elapsed(), 3), time.time(), f0.source,
                           self.mapping.cfg.to_dict(), gyr_bias)
