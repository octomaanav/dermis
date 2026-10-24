import argparse
import json
import urllib.request
from pathlib import Path

import pytest

from dermis.config import load_config
from dermis.evaluate import TRAIN_SEED_LIMIT, summarize, suite_scenarios

ROOT = Path(__file__).resolve().parents[1]


def test_saved_suites_are_held_out_and_disjoint():
    dev = suite_scenarios(load_config(ROOT / "configs/suites/dev_v1.json"))
    held = suite_scenarios(load_config(ROOT / "configs/suites/heldout_v1.json"))
    assert len(held) == 50 and all(s.seed >= 2_000_000 for s in held)
    assert {s.seed for s in dev}.isdisjoint({s.seed for s in held})
    assert all(s.seed >= TRAIN_SEED_LIMIT for s in dev)
    assert {round(s.friction, 2) for s in held} == {0.4, 0.6, 0.8, 1.0}
    # materialized file matches its generator spec
    spec = suite_scenarios(load_config(ROOT / "configs/suites/heldout.yaml"))
    assert [s.to_dict() for s in spec] == [s.to_dict() for s in held]


def test_suite_rejects_training_seeds():
    with pytest.raises(ValueError):
        suite_scenarios({"name": "bad", "generate": {"seed_start": 10, "count": 2}})


def test_summarize_counts():
    sc = {"friction": 0.4}
    eps = [
        dict(scenario=sc, success=True, termination="success", drops=0, excess_force_ticks=0, peak_force_n=15.0,
             completion_time_s=6.0, was_lifted=True),
        dict(scenario=sc, success=False, termination="timeout", drops=2, excess_force_ticks=3, peak_force_n=35.0,
             completion_time_s=None, was_lifted=True),
        dict(scenario={"friction": 1.0}, success=False, termination="object_out_of_bounds", drops=0,
             excess_force_ticks=0, peak_force_n=0.0, completion_time_s=None, was_lifted=False),
    ]
    s = summarize(eps, 30.0)
    assert (s["successes"], s["attempts"], s["episodes_with_drop"], s["total_drops"]) == (1, 3, 1, 2)
    assert s["episodes_with_excess_force"] == 1 and s["peak_contact_force_n_max"] == 35.0
    assert s["timeouts"] == 1 and s["terminations"]["object_out_of_bounds"] == 1 and s["never_lifted"] == 1
    assert s["completion_time_s_median"] == 6.0 and s["success_by_friction"] == {"0.40": "1/2", "1.00": "0/1"}
    assert "not a clinical" in s["excess_force_threshold_note"]


def test_dashboard_endpoints(tmp_path, fixture_path):
    from dermis.dashboard.server import serve_in_thread
    a = argparse.Namespace(source="replay", path=str(fixture_path("decoded_frame.json")), rate=50.0, loop=True,
                           host="127.0.0.1", port=0, serial_port=None, wire=False, faults="",
                           pods="index_tip,thumb_tip,middle_tip,thenar_apb", calibration=None, http_port=0,
                           data_root=str(tmp_path / "eps"), runs_root=str(tmp_path / "runs"),
                           results_root=str(tmp_path / "res"), recordings=str(tmp_path / "tel"))
    srv, dash = serve_in_thread(a)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    get = lambda p: json.loads(urllib.request.urlopen(base + p, timeout=5).read())
    try:
        dash.runner.wait_for_frame(3)
        st = get("/api/status")
        assert st["source"] == "REPLAY" and st["pipeline"]["accepted"] >= 1
        rows = get("/api/telemetry?since=0")["rows"]
        p = rows[0]["pods"]
        assert p[2]["imu"] == "stale" and p[3]["mag"] == "invalid" and p[3]["b_norm"] is None
        html = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        for label in ("SYNTHETIC", "SCRIPTED", "KEYBOARD", "GLOVE", "REPLAY", "AUTONOMOUS"):
            assert label in html
        assert get("/api/episodes") == [] and get("/api/results") == {"results": [], "comparisons": []}
        req = urllib.request.Request(base + "/api/episode?path=/etc/passwd")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=5)
        assert e.value.code == 403
    finally:
        srv.shutdown()
