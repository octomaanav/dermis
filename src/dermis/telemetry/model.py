"""Internal normalized telemetry model.

Everything downstream of the adapter (pipeline, calibration, teleop mapping, dashboard) works
with these types and never with source-specific formats. Raw readings and quality flags are
kept as delivered; `raw` holds the original decoded-frame dict for lossless recording.

Flag semantics follow firmware/v1/SPEC.md section 4.4:
    mag invalid   = BAD_LENGTH | CHIP_ERROR | COMM_FAIL     (values unusable)
    imu invalid   = COMM_FAIL
    STALE         = values repeated from an earlier sample: usable but old, never "fresh"
    TEMP_STALE    = only the temperature is old. It does NOT make b_ut / acc / gyr stale.
    RAIL          = saturated axis: a bound, not a measurement
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MAG_INVALID_FLAGS = frozenset({"BAD_LENGTH", "CHIP_ERROR", "COMM_FAIL"})
IMU_INVALID_FLAGS = frozenset({"COMM_FAIL"})
DEFAULT_POD_ORDER = ("index_tip", "thumb_tip", "middle_tip", "thenar_apb")


@dataclass(frozen=True)
class MagReading:
    ch: int
    b_ut: Optional[np.ndarray]          # (3,) microtesla, chip axes; None if absent/invalid
    temp_c: Optional[float]
    t_us: int                           # device clock, middle of the conversion window
    status: int
    flags: frozenset
    valid: bool                         # as reported by the decoder

    @property
    def usable(self) -> bool:
        """Has a value we may use (possibly old). Requires valid, non-null, finite, no invalid flag."""
        return (self.valid and self.b_ut is not None and not (self.flags & MAG_INVALID_FLAGS)
                and bool(np.all(np.isfinite(self.b_ut))))

    @property
    def fresh(self) -> bool:
        return self.usable and "STALE" not in self.flags

    @property
    def temp_fresh(self) -> bool:
        return self.temp_c is not None and "TEMP_STALE" not in self.flags and self.valid

    @property
    def saturated(self) -> bool:
        return "RAIL" in self.flags


@dataclass(frozen=True)
class ImuReading:
    ch: int
    acc_mps2: Optional[np.ndarray]      # (3,) specific force, chip axes (+g on the up axis at rest)
    gyr_rads: Optional[np.ndarray]      # (3,) rad/s, chip axes
    temp_c: Optional[float]
    t_us: int
    sensortime: int
    flags: frozenset
    valid: bool

    @property
    def usable(self) -> bool:
        return (self.valid and self.acc_mps2 is not None and self.gyr_rads is not None
                and not (self.flags & IMU_INVALID_FLAGS)
                and bool(np.all(np.isfinite(self.acc_mps2)) and np.all(np.isfinite(self.gyr_rads))))

    @property
    def fresh(self) -> bool:
        return self.usable and "STALE" not in self.flags

    @property
    def temp_fresh(self) -> bool:
        return self.temp_c is not None and "TEMP_STALE" not in self.flags and self.valid

    @property
    def saturated(self) -> bool:
        return bool(self.flags & {"ACC_RAIL", "GYR_RAIL"})


@dataclass(frozen=True)
class PodReading:
    location: str
    pod_id: str
    mag: Optional[MagReading]
    imu: Optional[ImuReading]


@dataclass(frozen=True)
class TelemetryFrame:
    device_id: str
    frame: int                          # device frame counter (u32, wraps)
    t_device_us: int
    t_host_ns: Optional[int]            # device time mapped to host clock; None until clock sync
    t_recv_ns: int                      # host monotonic receive time (always present)
    config_id: int
    flags: frozenset
    pods: dict                          # location -> PodReading
    source: str                         # SYNTHETIC | REPLAY | GLOVE
    raw: dict = field(repr=False, compare=False, default_factory=dict)
