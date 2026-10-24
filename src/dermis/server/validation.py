"""JSON Schema validation of decoded frames against firmware/v1/schemas/decoded_frame.schema.json.

Schema validation answers "is this shaped like a decoded frame"; runtime checks (validity flags,
STALE, ordering, finiteness, required channels) happen afterwards in the adapter, the pipeline
and the teleop mapping.
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


def schema_errors(obj: dict, limit: int = 3) -> list[str]:
    """Empty list if obj satisfies the decoded-frame schema, else up to `limit` short messages."""
    out = []
    for e in frame_validator().iter_errors(obj):
        where = "/".join(str(x) for x in e.absolute_path) or "<root>"
        out.append(f"{where}: {e.message[:160]}")
        if len(out) >= limit:
            break
    return out
