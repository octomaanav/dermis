"""Magnetic baseline calibration. Outputs magnetic CHANGE (uT) and unitless normalized deformation.

Nothing here is force. Converting delta-B to newtons needs a load-cell calibration per pod
(see docs/hardware_checklist.md); until that exists, never label these values as force.

    delta_ut[pod]   = R_pod @ (b_ut[pod] - baseline_ut[pod])   R_pod: chip axes -> pod frame (d, l, n)
    deformation     = |delta_ut| / scale_ut                     0 = unloaded, 1 = "scale_ut of change"

A calibration is tied to (device_id, config_id): with any other config the scale factors or
mounts may differ, so apply() reports valid_for_frame = False and the values must not be used.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .pipeline import Snapshot

CALIBRATION_VERSION = 1


def mount_matrix(mount: str) -> np.ndarray:
    """'+X+Y+Z'-style mount (SPEC section 8): which signed chip axis points along d, l, n.
    Returns R with pod = R @ chip. Rejects non-rotations."""
    if len(mount) != 6:
        raise ValueError(f"bad mount {mount!r}")
    R = np.zeros((3, 3))
    used = set()
    for row in range(3):
        sign, axis = mount[2 * row], mount[2 * row + 1].upper()
        if sign not in "+-" or axis not in "XYZ" or axis in used:
            raise ValueError(f"bad mount {mount!r}")
        used.add(axis)
        R[row, "XYZ".index(axis)] = 1.0 if sign == "+" else -1.0
    if not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError(f"mount {mount!r} is a mirror image, not a rotation")
    return R


@dataclass
class Calibration:
    device_id: str
    config_id: int
    pods: dict                                  # loc -> {baseline_ut, noise_std_ut, n_samples}
    scale_ut: float = 500.0
    transforms: dict = field(default_factory=dict)   # loc -> 3x3 list (chip -> pod frame)
    created_unix: float = field(default_factory=time.time)
    version: int = CALIBRATION_VERSION
    kind: str = "unloaded_magnetic_baseline"
    note: str = "Baseline subtraction only. Outputs magnetic change and normalized deformation, not force."
    calibration_id: str = ""

    def __post_init__(self):
        if not self.calibration_id:
            body = json.dumps([self.device_id, self.config_id, self.pods, self.scale_ut, self.created_unix], sort_keys=True)
            self.calibration_id = f"cal-{hashlib.sha256(body.encode()).hexdigest()[:10]}"

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        d = json.loads(Path(path).read_text())
        if d.get("version") != CALIBRATION_VERSION:
            raise ValueError(f"calibration version {d.get('version')} not supported")
        return cls(**d)

    def apply(self, snap: Snapshot, use_held: bool = True) -> dict:
        P = len(snap.pods)
        delta = np.full((P, 3), np.nan)
        b = snap.mag_held if use_held else np.where(snap.mag_fresh[:, None], snap.mag_raw, np.nan)
        for i, loc in enumerate(snap.pods):
            c = self.pods.get(loc)
            if c is None:
                continue
            R = np.array(self.transforms.get(loc, np.eye(3).tolist()))
            delta[i] = R @ (b[i] - np.array(c["baseline_ut"]))
        mag = np.linalg.norm(delta, axis=1)
        ok = np.isfinite(mag)
        valid = snap.frame.device_id == self.device_id and snap.frame.config_id == self.config_id
        return {"delta_ut": delta, "delta_norm_ut": mag, "deformation": mag / self.scale_ut,
                "mask": ok & valid, "valid_for_frame": valid, "calibration_id": self.calibration_id}


class BaselineCapture:
    """Feed snapshots while the glove is unloaded; then finish(). Uses fresh samples only."""

    def __init__(self, min_samples: int = 50, max_noise_ut: float = 50.0):
        self.min_samples, self.max_noise_ut = min_samples, max_noise_ut
        self._samples: dict[str, list] = {}
        self.device_id = None
        self.config_id = None

    def add(self, snap: Snapshot) -> None:
        if self.config_id is not None and (snap.frame.config_id != self.config_id or
                                           snap.frame.device_id != self.device_id):
            self._samples.clear()                 # restart: mixed configs cannot share a baseline
        self.device_id, self.config_id = snap.frame.device_id, snap.frame.config_id
        for i, loc in enumerate(snap.pods):
            if snap.mag_fresh[i]:
                self._samples.setdefault(loc, []).append(snap.mag_raw[i].copy())

    def finish(self, scale_ut: float = 500.0, mounts: dict | None = None) -> tuple[Calibration, dict]:
        pods, report = {}, {}
        for loc, s in self._samples.items():
            arr = np.array(s)
            std = arr.std(0)
            ok = len(arr) >= self.min_samples and bool(np.all(std < self.max_noise_ut))
            report[loc] = {"n_samples": len(arr), "noise_std_ut": std.round(3).tolist(), "accepted": ok}
            if ok:
                pods[loc] = {"baseline_ut": arr.mean(0).round(4).tolist(), "noise_std_ut": std.round(4).tolist(),
                             "n_samples": len(arr)}
        transforms = {loc: mount_matrix(m).tolist() for loc, m in (mounts or {}).items()}
        if self.device_id is None:
            raise ValueError("no telemetry received")
        return Calibration(self.device_id, self.config_id, pods, scale_ut, transforms), report


def mounts_from_device_info(info: dict) -> dict:
    """Pod location -> mag mount string from a DEVICE_INFO dict."""
    by_ch = {m["ch"]: m.get("mount") for m in info.get("mag", [])}
    return {p["location"]: by_ch[p["mag_ch"]] for p in info.get("pods", [])
            if p.get("mag_ch") is not None and by_ch.get(p["mag_ch"])}
