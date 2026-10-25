"""Where recordings come from: SpacetimeDB first, the local Dermis dataset as fallback.

Both sources return the same shapes, so analysis code never knows which one it read:
  list_episodes() -> list[dict]          episode metadata (see EPISODE_FIELDS)
  load_frames(id) -> (columns, ndarray)  frames x columns, column names as in schema.json

SpacetimeDB contract (what the agents expect Manav's module to expose):
  table `episode`: episode_id (pk), application, source ("sim" | "hardware"), object, mass_g, mu,
                   scenario, safe_grip, seed, frames, seconds, outcome, faults (JSON text or array),
                   first_fault_frame
  table `frame`:   episode_id, frame, then the 68 dermis-sim-v1 columns by the same names
                   (t_device_us, index_tip_bx_ut, ... label_fault). Missing columns are fine.
Table names are configurable with SPACETIME_EPISODE_TABLE / SPACETIME_FRAME_TABLE.
"""

from __future__ import annotations

import json
import logging
import re
import time
from functools import lru_cache

import httpx
import numpy as np

from . import config

log = logging.getLogger("dermis.data")

EPISODE_FIELDS = [
    "episode", "application", "source", "object", "mass_g", "mu", "scenario", "safe_grip",
    "seed", "frames", "seconds", "outcome", "faults", "first_fault_frame",
]


class DataUnavailable(RuntimeError):
    pass


class LocalDataset:
    name = "Dermis dataset"

    def __init__(self, root=config.DATA_DIR):
        self.root = root
        if not (root / "episodes.jsonl").exists():
            raise DataUnavailable(f"no dataset at {root}; set DERMIS_DATA_DIR")
        self.schema = json.loads((root / "schema.json").read_text())

    @lru_cache(maxsize=1)
    def list_episodes(self) -> list[dict]:
        rows = []
        for line in (self.root / "episodes.jsonl").read_text().splitlines():
            if line.strip():
                ep = json.loads(line)
                ep.setdefault("application", "hand")
                ep.setdefault("source", "sim")
                rows.append(ep)
        return rows

    @lru_cache(maxsize=16)
    def load_frames(self, episode_id: str) -> tuple[list[str], np.ndarray]:
        path = self.root / "episodes" / f"{episode_id}.npy"
        if not path.exists():
            raise DataUnavailable(f"episode {episode_id} not found")
        return self.schema["columns"], np.load(path)


class SpacetimeDataset:
    def __init__(self):
        if not config.SPACETIME_DB:
            raise DataUnavailable("SPACETIME_DB not set")
        self.url = f"{config.SPACETIME_HOST}/v1/database/{config.SPACETIME_DB}/sql"
        self.name = f"SpacetimeDB {config.SPACETIME_DB}"
        self._episodes: tuple[float, list[dict]] | None = None

    def sql(self, query: str) -> list[dict]:
        headers = {"Content-Type": "text/plain"}
        if config.SPACETIME_TOKEN:
            headers["Authorization"] = f"Bearer {config.SPACETIME_TOKEN}"
        try:
            r = httpx.post(self.url, content=query, headers=headers, timeout=20)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise DataUnavailable(f"SpacetimeDB query failed: {e}") from e
        out = []
        for stmt in r.json():
            names = [_field_name(el) for el in stmt["schema"]["elements"]]
            out.extend(dict(zip(names, row)) for row in stmt["rows"])
        return out

    def list_episodes(self) -> list[dict]:
        # Short cache so a burst of chat requests doesn't hammer the database.
        if self._episodes and time.time() - self._episodes[0] < 30:
            return self._episodes[1]
        rows = []
        for r in self.sql(f"SELECT * FROM {config.SPACETIME_EPISODE_TABLE}"):
            ep = {k: r.get(k) for k in EPISODE_FIELDS}
            ep["episode"] = r.get("episode_id") or r.get("episode")
            ep["faults"] = _as_list(ep.get("faults"))
            ep["application"] = ep.get("application") or "hand"
            ep["source"] = ep.get("source") or "hardware"
            ep["first_fault_frame"] = ep.get("first_fault_frame", -1) if ep.get("first_fault_frame") is not None else -1
            ep["safe_grip"] = bool(ep.get("safe_grip"))
            rows.append(ep)
        self._episodes = (time.time(), rows)
        return rows

    def load_frames(self, episode_id: str) -> tuple[list[str], np.ndarray]:
        if not re.fullmatch(r"[\w\-.:]+", episode_id):
            raise DataUnavailable(f"bad episode id {episode_id!r}")
        rows = self.sql(
            f"SELECT * FROM {config.SPACETIME_FRAME_TABLE} WHERE episode_id = '{episode_id}'"
        )
        if not rows:
            raise DataUnavailable(f"episode {episode_id} has no frames in SpacetimeDB")
        cols = [k for k in rows[0] if k != "episode_id" and _numeric(rows[0][k])]
        X = np.array([[float(r.get(c) or 0.0) for c in cols] for r in rows], dtype=np.float32)
        if "frame" in cols:
            X = X[np.argsort(X[:, cols.index("frame")])]
        return cols, X


def _field_name(el) -> str:
    name = el.get("name")
    if isinstance(name, dict):  # SATS option encoding: {"some": "col"}
        name = name.get("some")
    return str(name)


def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return [s for s in v.split(",") if s]
    return list(v)


def _numeric(v) -> bool:
    return isinstance(v, (int, float, bool))


class Source:
    """SpacetimeDB when it answers, otherwise the local dataset. Remembers why it fell back."""

    def __init__(self):
        self.note = ""
        self._remote = None
        self._local = None
        try:
            self._remote = SpacetimeDataset()
        except DataUnavailable as e:
            self.note = str(e)
        try:
            self._local = LocalDataset()
        except DataUnavailable as e:
            log.warning(str(e))

    def _try(self, method: str, *args):
        if self._remote is not None:
            try:
                result = getattr(self._remote, method)(*args)
                self.note = ""
                return result, self._remote.name
            except DataUnavailable as e:
                self.note = str(e)
                log.warning("falling back to local data: %s", e)
        if self._local is None:
            raise DataUnavailable(self.note or "no data source available")
        return getattr(self._local, method)(*args), self._local.name

    def list_episodes(self) -> tuple[list[dict], str]:
        return self._try("list_episodes")

    def load_frames(self, episode_id: str) -> tuple[tuple[list[str], np.ndarray], str]:
        return self._try("load_frames", episode_id)


_source: Source | None = None


def source() -> Source:
    global _source
    if _source is None:
        _source = Source()
    return _source
