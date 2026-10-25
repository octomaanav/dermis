"""Bring known *draft* decoded-frame shapes into the v1 shape before schema validation.

The v1 schema (firmware/v1/schemas/decoded_frame.schema.json) stays the contract and is never
edited here. This step only handles shapes we have actually been shown, each one noted:

    pod without an "imu" (or "mag") key        -> that key set to null: the channel is absent
    bare sensor record directly under "pods"   -> wrapped into a pod. "wrist_imu" becomes location
      (e.g. "wrist_imu": {"ch":3,"acc_mps2":...})  "wrist" if that is a schema location, else
                                                   "imu<ch>" / "mag<ch>" (the schema's channel form)
    record without "flags"                      -> flags [] (no flags reported)
    record without "valid"                      -> valid FALSE (unusable), unless assume_valid=True,
                                                   then valid = every value present and finite.
                                                   Quality is never invented by default.
    record still missing required keys          -> record dropped (null) so the rest of the frame survives

Anything else is left alone, so the schema check still rejects it with a clear message. The
input object is not modified (the raw line is what gets recorded).
"""
from __future__ import annotations

import copy
import math
import zlib

from .validation import frame_validator

MAG_KEYS = ("ch", "b_ut", "temp_c", "t_us", "status", "flags", "valid")
IMU_KEYS = ("ch", "acc_mps2", "gyr_rads", "temp_c", "t_us", "sensortime", "flags", "valid")
OPTIONAL_NULL = {"temp_c"}
ROOT_FLAGS = {"CONFIG_CHANGED", "OVERRUN", "TX_DROP", "MARKER"}
MAG_FLAGS = {"STALE", "BAD_LENGTH", "CHIP_ERROR", "RAIL", "COMM_FAIL", "TEMP_STALE", "RECOVERED"}
IMU_FLAGS = {"STALE", "COMM_FAIL", "ACC_RAIL", "GYR_RAIL", "TEMP_STALE", "RECOVERED"}


def _locations() -> set:
    return set(frame_validator().schema["$defs"]["location"]["enum"])


def _finite(v) -> bool:
    if v is None:
        return False
    if isinstance(v, list):
        return all(_finite(x) for x in v)
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _fix_record(rec: dict, kind: str, where: str, assume_valid: bool, notes: list):
    keys = MAG_KEYS if kind == "mag" else IMU_KEYS
    if "flags" not in rec:
        rec["flags"] = []
        notes.append(f"{where}: no 'flags'; using []")
    else:
        allowed = MAG_FLAGS if kind == "mag" else IMU_FLAGS
        mapped: list[str] = []
        for flag in rec["flags"]:
            replacements = {
                "NOT_FOUND": ("COMM_FAIL",),
                "SPI_ERROR": ("COMM_FAIL",),
                "SATURATED": (("RAIL",) if kind == "mag" else ("ACC_RAIL", "GYR_RAIL")),
                "SED": ("RECOVERED",),
            }.get(flag, (flag,))
            for replacement in replacements:
                if replacement in allowed and replacement not in mapped:
                    mapped.append(replacement)
        if mapped != rec["flags"]:
            notes.append(f"{where}: normalized draft flags {rec['flags']} -> {mapped}")
        rec["flags"] = mapped
    if "valid" not in rec:
        values = [rec.get("b_ut")] if kind == "mag" else [rec.get("acc_mps2"), rec.get("gyr_rads")]
        rec["valid"] = bool(assume_valid and all(_finite(v) for v in values))
        notes.append(f"{where}: no 'valid'; treated as {'valid (assume_valid)' if rec['valid'] else 'INVALID'}")
    for k in OPTIONAL_NULL:
        rec.setdefault(k, None)
    if kind == "mag":
        rec.setdefault("status", 0)
    missing = [k for k in keys if k not in rec]
    if missing:
        notes.append(f"{where}: missing {missing}; record dropped")
        return None
    return rec


def normalize_frame(obj: dict, assume_valid: bool = False) -> tuple[dict, list[str]]:
    """Returns (frame in v1 shape where possible, notes). Empty notes = already v1-shaped."""
    pods = obj.get("pods") if isinstance(obj, dict) else None
    if not isinstance(pods, dict):
        return obj, []
    notes: list[str] = []
    out = copy.deepcopy(obj)
    if isinstance(out.get("config_id"), str):
        original = out["config_id"]
        out["config_id"] = zlib.crc32(original.encode("utf-8"))
        notes.append(f"config_id: draft string {original!r} mapped to stable u32 {out['config_id']}")
    if isinstance(out.get("flags"), list):
        kept = [flag for flag in out["flags"] if flag in ROOT_FLAGS]
        if kept != out["flags"]:
            notes.append(f"frame.flags: ignored non-v1 draft flags {out['flags']} -> {kept}")
        out["flags"] = kept
    new_pods: dict = {}
    locs = _locations()
    for key, val in out["pods"].items():
        if not isinstance(val, dict):
            new_pods[key] = val
            continue
        if any(k in val for k in ("pod_id", "mag", "imu")):                 # a pod
            for kind in ("mag", "imu"):
                if kind not in val:
                    val[kind] = None
                    notes.append(f"pods.{key}: no '{kind}' key; channel absent")
                elif isinstance(val[kind], dict):
                    val[kind] = _fix_record(val[kind], kind, f"pods.{key}.{kind}", assume_valid, notes)
            if "pod_id" not in val:
                val["pod_id"] = ""
                notes.append(f"pods.{key}: no 'pod_id'")
            new_pods[key] = val
            continue
        kind = "imu" if ("acc_mps2" in val or "gyr_rads" in val) else "mag" if "b_ut" in val else None
        if kind is None:
            new_pods[key] = val                                              # unknown: let the schema say so
            continue
        base = key[: -len(f"_{kind}")] if key.endswith(f"_{kind}") else key
        loc = base if base in locs else f"{kind}{val.get('ch', '')}"
        rec = _fix_record(val, kind, f"pods.{key}", assume_valid, notes)
        target = new_pods.get(loc)
        if target is None:
            new_pods[loc] = {"pod_id": "", "mag": rec if kind == "mag" else None, "imu": rec if kind == "imu" else None}
        elif isinstance(target, dict) and target.get(kind) is None:
            target[kind] = rec
        else:
            loc = f"{kind}{val.get('ch', '')}"
            new_pods[loc] = {"pod_id": "", "mag": rec if kind == "mag" else None, "imu": rec if kind == "imu" else None}
        notes.append(f"pods.{key}: bare {kind} record moved to pods.{loc}")
    # v0 has exactly one palm IMU at the frame top level. Give it the canonical wrist
    # location so orientation and translation use the actual glove layout.
    top_imu = out.pop("imu", None)
    if isinstance(top_imu, dict):
        rec = _fix_record(top_imu, "imu", "imu", assume_valid, notes)
        wrist = new_pods.get("wrist")
        if wrist is None:
            new_pods["wrist"] = {"pod_id": str(top_imu.get("imu_id", "")), "mag": None, "imu": rec}
        elif isinstance(wrist, dict) and wrist.get("imu") is None:
            wrist["imu"] = rec
        notes.append("imu: top-level v0 palm IMU moved to pods.wrist")
    out["pods"] = new_pods
    return out, notes
