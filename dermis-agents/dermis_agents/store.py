"""The snapshot every agent reads at runtime: local file first, then DERMIS_SNAPSHOT_URL.

Pure Python + httpx, so it runs inside Agentverse-hosted agents. Hosted agents reset globals between
messages, so there the cache only lives for one request; locally it's reloaded when the file changes.
"""

from __future__ import annotations

import json

import httpx

from . import config


class DataUnavailable(RuntimeError):
    pass


class Snapshot:
    def __init__(self, raw: dict, origin: str):
        if raw.get("format") != "dermis-snapshot-v1":
            raise DataUnavailable(f"unexpected snapshot format {raw.get('format')!r}")
        self.episodes: list[dict] = raw["episodes"]
        self._facts: dict = raw["facts"]
        self.baseline: dict | None = raw.get("baseline")
        self.built_at: str = raw.get("built_at", "")
        self.source: str = raw.get("source", "")
        self.origin = origin

    def meta(self, episode_id: str) -> dict | None:
        return next((e for e in self.episodes if e["episode"] == episode_id), None)

    def facts(self, episode_id: str) -> dict:
        f = self._facts.get(episode_id)
        if f is None:
            raise DataUnavailable(f"episode {episode_id} not in snapshot")
        return f

    def all_facts(self) -> list[dict]:
        return [self._facts[e["episode"]] for e in self.episodes if e["episode"] in self._facts]

    @property
    def label(self) -> str:
        return f"{self.source} · snapshot {self.built_at}"


_cache: tuple[object, Snapshot] | None = None


def snapshot() -> Snapshot:
    global _cache
    path = config.SNAPSHOT_PATH
    if path.exists():
        key = ("file", path.stat().st_mtime)
        if _cache and _cache[0] == key:
            return _cache[1]
        snap = Snapshot(json.loads(path.read_text()), str(path))
    elif url := config.snapshot_url():
        key = ("url", url)
        if _cache and _cache[0] == key:
            return _cache[1]
        try:
            r = httpx.get(url, timeout=30, follow_redirects=True)
            r.raise_for_status()
            snap = Snapshot(r.json(), url)
        except (httpx.HTTPError, ValueError) as e:
            raise DataUnavailable(f"couldn't download the snapshot from {url}: {e}") from e
    else:
        raise DataUnavailable("no snapshot: run `python -m dermis_agents.snapshot` or set DERMIS_SNAPSHOT_URL")
    _cache = (key, snap)
    return snap
