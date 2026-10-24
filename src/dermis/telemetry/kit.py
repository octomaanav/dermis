"""Locate the firmware/v1 kit files (schemas, example fixtures) without hardcoding a workspace path.

Search order:
    1. $DERMIS_KIT_DIR (a directory holding schemas/ and examples/, i.e. firmware/v1)
    2. firmware/v1 under the current directory or any parent
    3. firmware/v1 under the directory holding this package, or any parent (editable installs)
    4. a bounded recursive search under the current directory (uploaded fixtures in other folders)
The fixtures are only ever read; nothing here writes to them.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

SUBDIRS = ("schemas", "examples", "host/python", "")


def _candidate_roots():
    env = os.environ.get("DERMIS_KIT_DIR")
    if env:
        yield Path(env)
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        for d in (base, *base.parents):
            yield d / "firmware" / "v1"


@lru_cache(maxsize=None)
def find_kit_file(name: str) -> Path:
    for root in _candidate_roots():
        for sub in SUBDIRS:
            p = root / sub / name if sub else root / name
            if p.is_file():
                return p
    for depth, p in _bounded_walk(Path.cwd(), max_depth=5):
        if p.name == name:
            return p
    raise FileNotFoundError(f"{name} not found; set DERMIS_KIT_DIR to the directory holding schemas/ and examples/")


def _bounded_walk(root: Path, max_depth: int):
    skip = {".git", ".venv", "node_modules", "__pycache__", "runs", "data", "results"}
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.is_dir():
                if depth < max_depth and e.name not in skip:
                    stack.append((e, depth + 1))
            else:
                yield depth, e
