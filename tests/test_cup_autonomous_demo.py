"""Autonomous fragile_cup demo: status classifier, worker trace, live snapshot, human disconnection with the
real demo checkpoint (skipped if it has not been trained), and checkpoint metadata completeness."""
import json
from pathlib import Path

import numpy as np
import pytest

from dermis.cup.controllers import PolicyController
from dermis.cup.env import CupConfig, CupHoldEnv, CupScenario
from dermis.cup.live import AUTONOMOUS, LiveSession
from dermis.cup.schedule import LoadSchedule
from dermis.cup.status import ENTER_TICKS, EXIT_TICKS, HoldStatus

CKPT = Path("runs/cup/autonomous_demo/cup_autonomous_demo_v1")
needs_ckpt = pytest.mark.skipif(not (CKPT / "model.pt").exists(), reason="demo checkpoint not trained")


class FakeEnv:
    """Minimal stand-in exposing what HoldStatus reads."""
    def __init__(self):
        self.cfg = CupConfig(damage_model="instant+sustained")
        self.stats = type("S", (), {"final_slip_m": 0.0, "termination": None, "damage_ns": 0.0, "success": False})()
        self._contact = {"normal": np.array([5.0, 5.0])}
        self.t, self.end_t, self.vz = 0.0, 10.0, 0.0

    def observe(self):
        from dermis.cup.env import OBS_NAMES
        o = np.zeros(len(OBS_NAMES), np.float32)
        o[OBS_NAMES.index("obj_vz")] = self.vz
        return o


def test_status_hysteresis_and_separate_load_status():
    env, hs = FakeEnv(), HoldStatus()
    env.vz = -0.01                                   # 10 mm/s down
    states = [hs.update(env)["movement"] for _ in range(ENTER_TICKS)]
    assert states[:-1] == ["HOLDING"] * (ENTER_TICKS - 1) and states[-1] == "SLIPPING"
    env.vz = -0.002                                  # between thresholds: stays SLIPPING (no flicker)
    assert all(hs.update(env)["movement"] == "SLIPPING" for _ in range(20))
    env.vz = 0.0
    states = [hs.update(env)["movement"] for _ in range(EXIT_TICKS)]
    assert states[-1] == "HOLDING" and states[-2] == "SLIPPING"
    env.stats.final_slip_m = 0.031                   # benchmark drop threshold
    assert hs.update(env)["movement"] == "DROPPED"
    env2, hs2 = FakeEnv(), HoldStatus()              # stationary cup, excessive squeeze
    env2._contact = {"normal": np.array([26.0, 25.5])}
    s = hs2.update(env2)
    assert s["movement"] == "HOLDING" and s["load_status"] == "EXCESSIVE LOAD"
    env2._contact = {"normal": np.array([9.0, 9.0])}
    env2.stats.damage_ns = 9.0                       # > half of the 15 N*s sustained budget
    assert hs2.update(env2)["load_status"] == "SUSTAINED-SQUEEZE WARNING"


def test_insufficient_grip_really_drops_and_is_reported(tmp_path):
    from dermis.cup.worker import run_trace
    r = run_trace({"kind": "scripted", "name": "fixed_2N"}, 2_000_000, str(tmp_path / "t.jsonl"))
    rows = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    head, last = rows[0], rows[-1]
    assert head["header"] and head["schedule_id"] == r["schedule_id"] and "schedule" in head["scenario"]
    assert r["termination"] == "drop" and last["status"]["movement"] == "DROPPED" and last["status"]["slip_cm"] > 3.0
    for k in ("robot_joints", "gripper_pos", "cup_pos", "cup_quat", "contact", "applied_load_n", "events", "status"):
        assert k in last
    assert set(last["robot_joints"]) == {f"j{j}" for j in list(range(7)) + [9, 10]}
    assert last["contact"]["normal_total"] == pytest.approx(last["contact"]["normal_l"] + last["contact"]["normal_r"], abs=2e-3)
    evs = [e for r_ in rows[1:] for e in r_["events"]]
    assert [e["event"] for e in evs] == ["load_start", "termination"] and evs[-1]["termination"] == "drop"


@needs_ckpt
def test_demo_checkpoint_metadata_is_complete():
    m = json.loads((CKPT / "metadata.json").read_text())
    for k in ("feature_names", "normalizer", "history", "action_names", "action_low", "action_high", "task",
              "task_version", "task_config", "definitions", "provenance"):
        assert k in m, k
    assert m["task"] == "fragile_cup" and m["task_version"].startswith("fragile_cup/instant+sustained/")
    assert m["provenance"]["sources"] == ["scripted"] and m["normalizer_fitted_on"] == "training episodes only"
    assert set(m["provenance"]["train_episode_ids"]).isdisjoint(m["provenance"]["val_episode_ids"])


@needs_ckpt
def test_real_checkpoint_ignores_human_commands_and_exposes_state():
    from dermis.cup.train import CupPolicy
    env = CupHoldEnv(CupConfig.from_dict({"damage_model": "instant+sustained"}))
    try:
        pol = PolicyController(CupPolicy.load(CKPT))
        sess = LiveSession(env, AUTONOMOUS, 2_000_000, policies={"demo": pol})
        assert sess.human is None
        sess.do("pour")
        ref = CupPolicy.load(CKPT)
        ref.reset()
        efforts, expected = [], []
        obs = sess.obs
        for _ in range(60):
            expected.append(float(ref.act(obs)[0]))
            sess.human_input({"up", "w"}, {"9", "w"})          # mashed grip keys
            sess.tick()
            efforts.append(env.effort)
            obs = sess.obs
        assert sess.ignored_human_events == 60
        np.testing.assert_allclose(efforts, np.clip(expected, 0.5, 40.0), atol=1e-5)   # the policy, not the keys
        snap = sess.snapshot()
        for k in ("robot_joints", "cup_pose", "finger_normal_n", "finger_normal_total_n", "applied_load_n", "hold_status",
                  "schedule_id", "ramp_remaining_s", "checkpoint", "outcome"):
            assert k in snap, k
        assert snap["checkpoint"] == "cup_autonomous_demo_v1" and snap["hold_status"]["movement"] in ("HOLDING", "SLIPPING", "DROPPED")
        json.dumps(snap)
    finally:
        env.close()


def test_no_attachment_during_live_session():
    import pybullet as p
    env = CupHoldEnv()
    try:
        class P:
            meta = {"name": "p", "feature_groups": ["gripper"], "provenance": {"train_episode_ids": [], "sources": []}}
            def reset(self): pass
            def act(self, o): return np.array([0.5], np.float32)
        sess = LiveSession(env, AUTONOMOUS, 2_000_001, policies={"p": PolicyController(P())})
        sess.do("pour")
        for _ in range(200):
            sess.tick()
        assert p.getNumConstraints(physicsClientId=env.client) == 0
        assert env.stats.termination == "drop"                    # 0.5 N cannot hold the cup: it really falls
    finally:
        env.close()
