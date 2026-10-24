"""YAML/JSON config loading."""
from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    return yaml.safe_load(text) or {}
