import json

import numpy as np
import pytest

from dermis.cli import main
from dermis.data.episodes import EpisodeStore


@pytest.mark.slow
def test_smoke_pipeline(tmp_path):
    out = tmp_path / "smoke"
    assert main(["smoke-test", "--out", str(out), "--episodes", "3", "--eval-episodes", "1"]) == 0
    s = json.loads((out / "smoke_summary.json").read_text())
    assert s["ok"] and s["demos_succeeded"] == 3 and s["eval_attempts"] == 1
    r = json.loads((out / "results.json").read_text())
    assert r["policy"]["label"] == "AUTONOMOUS" and r["episodes"][0]["scenario"]["seed"] >= 1_000_000


@pytest.mark.slow
def test_headless_glove_demo_with_synthetic_telemetry(tmp_path):
    """GLOVE source through the shared pipeline + EXPERIMENTAL mapping; records status per tick."""
    cal = tmp_path / "cal.json"
    assert main(["calibrate", "--source", "synthetic", "--rate", "1000", "--frames", "30", "--out", str(cal)]) == 0
    assert main(["demo", "--source", "glove", "--headless", "--max-episodes", "1", "--seed-start", "7",
                 "--glove-input", "synthetic", "--calibration", str(cal), "--out", str(tmp_path / "eps")]) == 2
    rc = main(["demo", "--source", "glove", "--headless", "--max-episodes", "1", "--seed-start", "7",
               "--experimental-glove-teleop", "--glove-input", "synthetic", "--calibration", str(cal),
               "--out", str(tmp_path / "eps")])
    assert rc == 0
    store = EpisodeStore(tmp_path / "eps")
    ep = store.load(store.ids()[0])
    assert ep.meta["source"] == "glove" and ep.meta["telemetry_source"] == "SYNTHETIC"
    assert ep.meta["calibration_id"] and "EXPERIMENTAL" in ep.meta["teleop_mapping"]
    tags = set(ep.arrays["tags"])
    assert tags <= {"ACTIVE", "PAUSED_NO_DATA", "PAUSED_STALE", "PAUSED_CONFIG"} and "ACTIVE" in tags
    assert np.isfinite(ep.obs).all() and len(ep) == len(ep.actions)
