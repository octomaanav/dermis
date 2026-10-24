"""JSON Schema validation of decoded frames against firmware/v1/schemas/decoded_frame.schema.json.

Schema validation answers "is this shaped like a decoded frame"; runtime checks (validity flags,
STALE, ordering, finiteness, required channels) happen afterwards in the adapter, the pipeline
and the teleop mapping.

Speed: the generic jsonschema validator takes ~3 ms per frame (measured, M2), which would cost a
full CPU core at the glove's 350 Hz. fast_valid() checks exactly the same rules for this schema in
~tens of microseconds, reading every enum and bound from the schema file itself. Frames it accepts
are valid; anything it rejects goes through jsonschema, which produces the error messages.
tests/test_tcp_validation_fast.py checks the two agree on the fixture and thousands of mutations.
"""
from __future__ import annotations

import json
from functools import lru_cache

from ..telemetry.kit import find_kit_file


@lru_cache(maxsize=None)
def frame_validator():
    from jsonschema import Draft202012Validator
    schema = json.loads(find_kit_file("decoded_frame.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@lru_cache(maxsize=None)
def _rules() -> dict:
    s = frame_validator().schema
    d = s["$defs"]
    pn = s["properties"]["pods"]["propertyNames"]["anyOf"]
    return {
        "frame_req": tuple(s["required"]), "frame_flags": frozenset(s["properties"]["flags"]["items"]["enum"]),
        "locations": frozenset(d["location"]["enum"]), "pod_pattern": pn[1]["pattern"],
        "pod_req": tuple(d["pod"]["required"]),
        "mag_req": tuple(d["mag"]["required"]), "mag_flags": frozenset(d["mag"]["properties"]["flags"]["items"]["enum"]),
        "status": (d["mag"]["properties"]["status"]["minimum"], d["mag"]["properties"]["status"]["maximum"]),
        "imu_req": tuple(d["imu"]["required"]), "imu_flags": frozenset(d["imu"]["properties"]["flags"]["items"]["enum"]),
        "sensortime": (d["imu"]["properties"]["sensortime"]["minimum"], d["imu"]["properties"]["sensortime"]["maximum"]),
        "min0": ("frame", "t_device_us"),
    }


def _int(v) -> bool:
    # JSON Schema "integer": any number with zero fractional part; booleans are not numbers
    return (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer())


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _vec3(v) -> bool:
    return v is None or (isinstance(v, list) and len(v) == 3 and all(_num(x) for x in v))


def _flags(v, allowed) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) and x in allowed for x in v)


def fast_valid(obj) -> bool:
    """True only if obj satisfies decoded_frame.schema.json (same rules, much faster)."""
    import re
    r = _rules()
    if not isinstance(obj, dict) or any(k not in obj for k in r["frame_req"]):
        return False
    if not isinstance(obj["device_id"], str) or not _int(obj["config_id"]):
        return False
    for k in r["min0"]:
        if not _int(obj[k]) or obj[k] < 0:
            return False
    if not (obj["t_host_ns"] is None or _int(obj["t_host_ns"])):
        return False
    if not _flags(obj["flags"], r["frame_flags"]):
        return False
    pods = obj["pods"]
    if not isinstance(pods, dict):
        return False
    for name, pod in pods.items():
        if name not in r["locations"] and not re.search(r["pod_pattern"], name):
            return False
        if not isinstance(pod, dict) or any(k not in pod for k in r["pod_req"]) or not isinstance(pod["pod_id"], str):
            return False
        m, i = pod["mag"], pod["imu"]
        if m is not None:
            if not isinstance(m, dict) or any(k not in m for k in r["mag_req"]):
                return False
            if not (_int(m["ch"]) and _vec3(m["b_ut"]) and (m["temp_c"] is None or _num(m["temp_c"]))
                    and _int(m["t_us"]) and _int(m["status"]) and r["status"][0] <= m["status"] <= r["status"][1]
                    and _flags(m["flags"], r["mag_flags"]) and isinstance(m["valid"], bool)):
                return False
        if i is not None:
            if not isinstance(i, dict) or any(k not in i for k in r["imu_req"]):
                return False
            if not (_int(i["ch"]) and _vec3(i["acc_mps2"]) and _vec3(i["gyr_rads"])
                    and (i["temp_c"] is None or _num(i["temp_c"])) and _int(i["t_us"])
                    and _int(i["sensortime"]) and r["sensortime"][0] <= i["sensortime"] <= r["sensortime"][1]
                    and _flags(i["flags"], r["imu_flags"]) and isinstance(i["valid"], bool)):
                return False
    return True


def schema_errors(obj: dict, limit: int = 3) -> list[str]:
    """Empty list if obj satisfies the decoded-frame schema, else up to `limit` short messages."""
    if fast_valid(obj):
        return []
    out = []
    for e in frame_validator().iter_errors(obj):
        where = "/".join(str(x) for x in e.absolute_path) or "<root>"
        out.append(f"{where}: {e.message[:160]}")
        if len(out) >= limit:
            break
    return out
