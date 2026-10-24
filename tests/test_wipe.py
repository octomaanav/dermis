"""wipe_v1: metric exploit guards, observation access, frozen suite integrity, glove tooling (software only)."""
import json

import numpy as np
import pybullet as p
import pytest

import dermis.wipe.env as wenv
from dermis.wipe.controllers import ConstantHeight, ContactPI, GeometryFollow, ObsFilter, make
from dermis.wipe.env import WipeConfig, WipeEnv, WipeScenario
from dermis.wipe.evaluate import load_eval_suite, make_eval_suite, run_episode


def test_flat_sanity_case_is_solvable():
    r = run_episode({"kind": "D", "k_z": 600.0, "kp": 0.0003, "ki": 0.005}, WipeScenario.flat(), WipeConfig())
    assert r["success"] and r["coverage"] == 1.0 and r["overload_s"] == 0.0 and r["termination"] == "completed"
    assert r["approach_time_s"] > 1.9 and r["wipe_time_s"] > 20 and r["retract_time_s"] > 0.5


def test_hovering_cleans_nothing():
    r = run_episode({"kind": "A", "k_z": 3000.0, "target_n": -30.0}, WipeScenario.flat(), WipeConfig())
    assert r["coverage"] == 0.0 and r["attempted_coverage"] == 0.0 and r["contact_loss_s"] > 20


def test_pressing_through_is_overload_not_coverage():
    r = run_episode({"kind": "A", "k_z": 3000.0, "target_n": 60.0}, WipeScenario.flat(), WipeConfig())
    assert r["coverage"] < 0.05 and r["overload_s"] > 1.0 and r["peak_load_n"] > 10 and not r["success"]


def test_moving_too_fast_cannot_clean(monkeypatch):
    monkeypatch.setattr(wenv, "V_XY_MAX", 2.0)
    import dermis.wipe.controllers as wc
    monkeypatch.setattr(wc, "V_XY_MAX", 2.0)
    fast = WipeConfig(wipe_speed=0.6)                 # cell traversal 33 ms << 150 ms dwell
    r = run_episode({"kind": "D", "k_z": 600.0, "kp": 0.0003, "ki": 0.005}, WipeScenario.flat(), fast)
    assert r["coverage"] < 0.2


def test_cells_count_once():
    env = WipeEnv()
    try:
        env.reset(WipeScenario.flat())
        c = env.cfg
        xy = (c.region_x[0] + 0.01, c.region_y[0] + 0.01)
        for _ in range(2000):                          # 8 s of in-band contact in one cell
            env._account(4.0, xy, "wipe", 1 / 240)
        assert env.wiped.sum() == 1 and len(env.stats.events) == 1
        env._account(4.0, (5.0, 5.0), "wipe", 1 / 240)   # contact outside the grid is ignored
        assert env.wiped.sum() == 1
    finally:
        env.close()


def test_workspace_escape_detected():
    env = WipeEnv()
    try:
        env.reset(WipeScenario.flat())
        done = False
        for _ in range(200):
            _, done, _ = env.step([0.12, 0.12, 0.05])
            if done:
                break
        assert env.stats.termination == "workspace_escape" and not env.stats.success
    finally:
        env.close()


def test_observation_access_is_enforced():
    env = WipeEnv()
    try:
        full = env.reset(WipeScenario.sample(1_000_007))
        for ctrl, has_load in ((ConstantHeight(300.0), False), (GeometryFollow(150.0), False),
                               (ContactPI(600.0, 0.0, 0.002), True)):
            v = ObsFilter(env, ctrl.SEES).view(full)
            assert ("normal_load" in v) == has_load
            assert "geometry" in v or isinstance(ctrl, ConstantHeight)
            for banned in ("pad_stiffness", "geom_offset", "scenario", "wiped", "dwell", "success", "coverage"):
                assert banned not in v
        # observed geometry carries the scenario's calibration offset, i.e. it is not the truth
        v = ObsFilter(env, GeometryFollow.SEES).view(full)
        x, y = 0.5, 0.0
        assert abs(v["geometry"](x, y) - float(env.surface_height(x, y)) - env.scenario.geom_offset) < 1e-12
    finally:
        env.close()


def test_frozen_suite_integrity(tmp_path):
    path = tmp_path / "s.json"
    make_eval_suite(str(path), count=3)
    assert len(load_eval_suite(str(path))) == 4
    d = json.loads(path.read_text())
    d["scenarios"][1]["bump_amp"] = 0.0
    path.write_text(json.dumps(d))
    with pytest.raises(AssertionError):
        load_eval_suite(str(path))


def test_repo_eval_suite_is_frozen_and_disjoint_from_dev():
    from dermis.wipe.evaluate import dev_suite
    ev = load_eval_suite("configs/wipe/eval_v1.json")
    assert {s.seed for s in ev}.isdisjoint({s.seed for s in dev_suite()})


@pytest.mark.slow
def test_glove_tools_software_path(tmp_path):
    """SYNTHETIC input exercises the tooling only; the report must say PENDING HARDWARE."""
    from dermis.cli import main
    out = tmp_path / "sess"
    assert main(["wipe", "glove-record", "--source", "synthetic", "--pod", "thenar_apb", "--out", str(out), "--auto",
                 "--repeats", "1", "--trial-s", "0.5", "--baseline-frames", "80"]) == 0
    assert main(["wipe", "glove-report", "--session", str(out)]) == 0
    nums = json.loads((out / "report" / "numbers.json").read_text())
    assert nums["status"] == "PENDING HARDWARE" and nums["source"] == "SYNTHETIC"
    assert "NOT force" in nums["units"] and (out / "report" / "repeats.png").exists()
    frames = (out / "frames.jsonl").read_text().splitlines()
    assert "_t_recv_ns" in json.loads(frames[0])
