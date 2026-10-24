"""Adapter: decoded-frame dict (firmware/v1/schemas/decoded_frame.schema.json) -> TelemetryFrame.

This is the only module that knows the decoded-frame layout. If the final glove stream uses a
different shape, write another adapter that returns TelemetryFrame and pass it to the pipeline.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .model import ImuReading, MagReading, PodReading, TelemetryFrame

FRAME_KEYS = ("device_id", "frame", "t_device_us", "t_host_ns", "config_id", "flags", "pods")
MAG_KEYS = ("ch", "b_ut", "temp_c", "t_us", "status", "flags", "valid")
IMU_KEYS = ("ch", "acc_mps2", "gyr_rads", "temp_c", "t_us", "sensortime", "flags", "valid")


class AdapterError(ValueError):
    pass


def _vec3(v, where: str):
    if v is None:
        return None
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise AdapterError(f"{where}: expected 3-vector or null, got {v!r}")
    out = np.array([math.nan if x is None else float(x) for x in v], dtype=np.float64)
    return out


def _num(v, where: str):
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AdapterError(f"{where}: expected number or null")
    return float(v)


def _require(d: dict, keys, where: str) -> None:
    if not isinstance(d, dict):
        raise AdapterError(f"{where}: expected object")
    missing = [k for k in keys if k not in d]
    if missing:
        raise AdapterError(f"{where}: missing {missing}")


def _mag(d, where) -> MagReading | None:
    if d is None:
        return None
    _require(d, MAG_KEYS, where)
    return MagReading(int(d["ch"]), _vec3(d["b_ut"], where + ".b_ut"), _num(d["temp_c"], where + ".temp_c"),
                      int(d["t_us"]), int(d["status"]), frozenset(d["flags"]), bool(d["valid"]))


def _imu(d, where) -> ImuReading | None:
    if d is None:
        return None
    _require(d, IMU_KEYS, where)
    return ImuReading(int(d["ch"]), _vec3(d["acc_mps2"], where + ".acc_mps2"), _vec3(d["gyr_rads"], where + ".gyr_rads"),
                      _num(d["temp_c"], where + ".temp_c"), int(d["t_us"]), int(d["sensortime"]),
                      frozenset(d["flags"]), bool(d["valid"]))


def adapt_decoded_frame(d: dict, t_recv_ns: int, source: str) -> TelemetryFrame:
    _require(d, FRAME_KEYS, "frame")
    if not isinstance(d["pods"], dict):
        raise AdapterError("frame.pods: expected object keyed by location")
    pods = {}
    for loc, pod in d["pods"].items():
        _require(pod, ("pod_id", "mag", "imu"), f"pods.{loc}")
        pods[loc] = PodReading(loc, str(pod["pod_id"]), _mag(pod["mag"], f"pods.{loc}.mag"),
                               _imu(pod["imu"], f"pods.{loc}.imu"))
    return TelemetryFrame(str(d["device_id"]), int(d["frame"]), int(d["t_device_us"]),
                          None if d["t_host_ns"] is None else int(d["t_host_ns"]), int(t_recv_ns),
                          int(d["config_id"]), frozenset(d["flags"]), pods, source, raw=d)


Adapter = Callable[[dict, int, str], TelemetryFrame]
