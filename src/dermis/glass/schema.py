"""Glove frame schema with provisional on-device force (schema_version 1, config
'dermis-4mlx-1imu-force-provisional-v1'): four magnetic pods (index_tip, thumb_tip, middle_tip, thenar_apb),
each with one `mag` block that carries a firmware force estimate, plus ONE top-level IMU.

    {"schema_version":1,"device_id":"DERMIS-...","frame":1761,"t_device_us":184921209,"t_host_ns":null,
     "config_id":"dermis-4mlx-1imu-force-provisional-v1","flags":[],
     "pods":{"index_tip":{"pod_id":"P-0001","mag":{"ch":0,"b_ut":[x,y,z],"temp_c":26.57,"force_n":7.68,
             "force_valid":true,"t_us":...,"status":35,"flags":["FORCE_PROVISIONAL"],"valid":true}}, ...},
     "imu":{"imu_id":"IMU-0001","ch":0,"acc_mps2":[..],"gyr_rads":[..],"temp_c":27.0,"t_us":...,
            "sensortime":4688984,"flags":[],"valid":true}}

This is a different layout from the decoded-frame v1 handled by dermis.telemetry.adapter (integer
config_id, an IMU per pod, no force); that adapter is unchanged. `force_n` is the firmware's PROVISIONAL
estimate (FORCE_PROVISIONAL), clamped to [0, FORCE_MAX_N] (FORCE_CLAMPED_LOW / _HIGH); it is not a
calibrated measurement. Frames produced by dermis.glass.sensors carry the top-level flag 'SYNTHETIC'.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCHEMA_VERSION = 1
CONFIG_ID = "dermis-4mlx-1imu-force-provisional-v1"
POD_ORDER = ("thumb_tip", "index_tip", "middle_tip", "thenar_apb")
POD_IDS = {"index_tip": "P-0001", "thumb_tip": "P-0002", "middle_tip": "P-0003", "thenar_apb": "P-0004"}
POD_CH = {"index_tip": 0, "thumb_tip": 1, "middle_tip": 2, "thenar_apb": 3}
FORCE_MAX_N = 9.80665                       # firmware clamp seen in device frames (1 kgf)
FORCE_FLAGS = ("FORCE_PROVISIONAL", "FORCE_CLAMPED_LOW", "FORCE_CLAMPED_HIGH")

_TOP = {"schema_version": int, "device_id": str, "frame": int, "t_device_us": int, "config_id": str,
        "flags": list, "pods": dict, "imu": dict}
_MAG = {"ch": int, "b_ut": list, "temp_c": (int, float), "force_n": (int, float), "force_valid": bool,
        "t_us": int, "status": int, "flags": list, "valid": bool}
_IMU = {"imu_id": str, "ch": int, "acc_mps2": list, "gyr_rads": list, "temp_c": (int, float), "t_us": int,
        "sensortime": int, "flags": list, "valid": bool}


class SchemaError(ValueError):
    pass


def _check(d: dict, spec: dict, where: str) -> None:
    for k, t in spec.items():
        if k not in d:
            raise SchemaError(f"{where}: missing '{k}'")
        v = d[k]
        if isinstance(v, bool) and t is not bool and bool not in (t if isinstance(t, tuple) else (t,)):
            raise SchemaError(f"{where}.{k}: bool where {t} expected")
        if not isinstance(v, t):
            raise SchemaError(f"{where}.{k}: expected {t}, got {type(v).__name__}")


def validate(d: dict) -> None:
    """Raise SchemaError unless `d` matches the frame layout above (all four pods required)."""
    _check(d, _TOP, "frame")
    if "t_host_ns" not in d or not (d["t_host_ns"] is None or isinstance(d["t_host_ns"], int)):
        raise SchemaError("frame.t_host_ns: expected int or null")
    if d["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(f"unsupported schema_version {d['schema_version']}")
    missing = set(POD_ORDER) - set(d["pods"])
    if missing:
        raise SchemaError(f"missing pods {sorted(missing)}")
    for name, pod in d["pods"].items():
        if not isinstance(pod, dict) or "pod_id" not in pod or "mag" not in pod:
            raise SchemaError(f"pods.{name}: needs pod_id and mag")
        _check(pod["mag"], _MAG, f"pods.{name}.mag")
        if len(pod["mag"]["b_ut"]) != 3:
            raise SchemaError(f"pods.{name}.mag.b_ut: expected 3 values")
    _check(d["imu"], _IMU, "imu")
    for k in ("acc_mps2", "gyr_rads"):
        if len(d["imu"][k]) != 3:
            raise SchemaError(f"imu.{k}: expected 3 values")


@dataclass
class ParsedFrame:
    frame: int
    t_device_us: int
    synthetic: bool
    b_ut: np.ndarray          # (4, 3) in POD_ORDER
    force_n: np.ndarray       # (4,) provisional firmware force; NaN where force_valid/valid is false
    clamped_high: np.ndarray  # (4,) bool
    clamped_low: np.ndarray   # (4,) bool
    valid: np.ndarray         # (4,) bool
    acc_mps2: np.ndarray
    gyr_rads: np.ndarray


def parse(d: dict) -> ParsedFrame:
    validate(d)
    mags = [d["pods"][n]["mag"] for n in POD_ORDER]
    ok = np.array([m["valid"] for m in mags])
    fv = np.array([m["valid"] and m["force_valid"] for m in mags])
    return ParsedFrame(
        frame=d["frame"], t_device_us=d["t_device_us"], synthetic="SYNTHETIC" in d["flags"],
        b_ut=np.array([m["b_ut"] for m in mags], float),
        force_n=np.where(fv, [float(m["force_n"]) for m in mags], np.nan),
        clamped_high=np.array(["FORCE_CLAMPED_HIGH" in m["flags"] for m in mags]),
        clamped_low=np.array(["FORCE_CLAMPED_LOW" in m["flags"] for m in mags]),
        valid=ok, acc_mps2=np.array(d["imu"]["acc_mps2"], float), gyr_rads=np.array(d["imu"]["gyr_rads"], float))
