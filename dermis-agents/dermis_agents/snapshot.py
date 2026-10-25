"""Build the snapshot: every recording analysed once, saved as one JSON file all agents read.

  python -m dermis_agents.snapshot            build from SpacetimeDB (or the local dataset)

Agents never touch raw frames at runtime, so they run anywhere, including Agentverse-hosted agents
where pandas isn't available and the 222 MB of recordings can't be shipped. Rebuild whenever the
data changes, then publish the file wherever DERMIS_SNAPSHOT_URL points.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from . import config
from .data import source
from .features import analyze_episode

FORMAT = "dermis-snapshot-v1"


def build() -> dict:
    src = source()
    episodes, src_name = src.list_episodes()
    facts = {}
    for i, e in enumerate(episodes, 1):
        (cols, X), src_name = src.load_frames(e["episode"])
        facts[e["episode"]] = analyze_episode(e, cols, X)
        if i % 32 == 0 or i == len(episodes):
            print(f"  analysed {i}/{len(episodes)}", flush=True)
    baseline_path = config.DATA_DIR / "baseline_results.json"
    return {
        "format": FORMAT,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": src_name,
        "episodes": episodes,
        "facts": facts,
        "baseline": json.loads(baseline_path.read_text()) if baseline_path.exists() else None,
    }


def main() -> None:
    t0 = time.time()
    snap = build()
    config.SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.SNAPSHOT_PATH.write_text(json.dumps(snap, separators=(",", ":"), default=str))
    size = config.SNAPSHOT_PATH.stat().st_size / 1e6
    print(f"Wrote {config.SNAPSHOT_PATH} · {len(snap['episodes'])} episodes · {size:.1f} MB · "
          f"{time.time() - t0:.0f} s · from {snap['source']}")


if __name__ == "__main__":
    main()
