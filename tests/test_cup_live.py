"""fragile_cup: policy inference drives the joints; human input cannot reach the robot in AUTONOMOUS;
identical scenarios across policies; recorder; tiny training round trip."""
import json

import numpy as np
import pybullet as p
import pytest

from dermis.cup.controllers import KeyboardGrip, PolicyController
from dermis.cup.env import FINGER_JOINTS, OBS_NAMES, CupHoldEnv, CupScenario
from dermis.cup.live import AUTONOMOUS, TEACH, LiveSession
from dermis.cup.rollout import CupRecorder, collect, run_episode
from dermis.data.episodes import EpisodeStore


class FakePolicy:
    def __init__(self, effort, name="fake"):
        self.effort, self.calls = effort, 0
        self.meta = {"name": name, "feature_groups": ["gripper", "object"],
                     "provenance": {"train_episode_ids": ["a"], "sources": ["scripted"]}}

    def reset(self):
        pass

    def act(self, obs):
        self.calls += 1
        assert obs.shape == (len(OBS_NAMES),)
        return np.array([self.effort], np.float32)


@pytest.fixture(scope="module")
def env():
    e = CupHoldEnv()
    yield e
    e.close()


@pytest.mark.parametrize("effort", [3.0, 18.0])
def test_policy_inference_controls_finger_joints(env, effort):
    pol = FakePolicy(effort)
    env.reset(CupScenario.sample(1_000_001, env.cfg), pour_at_s=None)
    for _ in range(30):
        env.step(PolicyController(pol).act(env.observe())[0])
    assert pol.calls == 30
    # the policy's number is what the finger motors apply: measured squeeze follows it
    assert np.allclose(env._contact["normal"], effort, atol=0.3)
    forces = [p.getJointState(env.robot, j, physicsClientId=env.client)[3] for j in FINGER_JOINTS]
    assert all(abs(abs(f) - effort) < 0.5 for f in forces)


def test_human_commands_cannot_reach_robot_in_autonomous(env, tmp_path):
    sess = LiveSession(env, AUTONOMOUS, 1_000_002, policies={"fake": PolicyController(FakePolicy(10.0))})
    assert sess.human is None
    sess.do("pour")
    for _ in range(40):
        sess.human_input({"up", "w"}, {"9", "w"})              # effort keys mashed by a "judge"
        sess.tick()
    assert env.effort == 10.0 and sess.ignored_human_events == 40
    assert np.allclose(env._contact["normal"], 10.0, atol=0.3)

    kb = KeyboardGrip()
    teach = LiveSession(env, TEACH, 1_000_002, teach_input=kb, teach_dir=str(tmp_path / "teach"))
    teach.human_input(set(), {"9"})
    teach.tick()
    assert env.effort == 36.0                                   # same key reaches the robot in TEACH


def test_teach_records_issued_commands_with_source(env, tmp_path):
    kb = KeyboardGrip()
    sess = LiveSession(env, TEACH, 1_000_003, teach_input=kb, teach_dir=str(tmp_path / "teach"))
    sess.do("record")
    sess.do("pour")
    issued = []
    while not env.done and len(issued) < 2000:
        sess.human_input({"up"} if len(issued) % 3 == 0 else set(), set())
        sess.tick()
        issued.append(env.effort)
    store = EpisodeStore(tmp_path / "teach")
    ep = store.load(store.ids()[0])
    assert ep.meta["source"] == "keyboard" and ep.meta["label"] == "KEYBOARD" and ep.meta["task"] == "fragile_cup"
    np.testing.assert_allclose(ep.actions[:, 0], issued[:len(ep)], atol=1e-5)
    assert "expert_actions" not in ep.arrays and set(ep.arrays["tags"]) == {"KEYBOARD"}


def test_reset_and_paired_comparison_use_identical_conditions(env):
    pols = {"a": PolicyController(FakePolicy(6.0, "a")), "b": PolicyController(FakePolicy(20.0, "b"))}
    sess = LiveSession(env, AUTONOMOUS, 1_000_004, policies=pols)
    o1 = sess.reset()
    o2 = sess.reset()
    np.testing.assert_array_equal(o1, o2)
    res = sess.paired_compare()
    assert [r["policy"] for r in res] == ["a", "b"] and {r["seed"] for r in res} == {1_000_004}
    assert res[0]["mean_effort_n"] != res[1]["mean_effort_n"]


def test_evaluation_scenarios_identical_across_controllers(tmp_path):
    from dermis.cup.evaluate import evaluate
    suite = tmp_path / "s.json"
    suite.write_text(json.dumps({"name": "t", "generate": {"seed_start": 1_700_000, "count": 3}}))
    r1 = evaluate({"kind": "scripted", "name": "fixed_10N"}, str(suite), None, 1, log=lambda *_: None)
    r2 = evaluate({"kind": "scripted", "name": "grip_load_expert"}, str(suite), None, 1, log=lambda *_: None)
    assert [e["scenario_hash"] for e in r1["episodes"]] == [e["scenario_hash"] for e in r2["episodes"]]
    assert [e["scenario"] for e in r1["episodes"]] == [e["scenario"] for e in r2["episodes"]]


def test_suite_rejects_training_seeds():
    from dermis.cup.evaluate import suite_scenarios
    with pytest.raises(ValueError):
        suite_scenarios({"generate": {"seed_start": 5, "count": 2}})


@pytest.mark.slow
def test_scripted_collection_and_training_round_trip(tmp_path):
    from dermis.cup.train import CupPolicy, train
    collect(str(tmp_path / "eps"), [0, 1, 2, 3], "grip_load_expert", 1.0, {})
    store = EpisodeStore(tmp_path / "eps")
    ep = store.load(store.ids()[0])
    assert ep.meta["source"] == "scripted" and ep.meta["controller"] == "grip_load_expert"
    assert "expert_actions" in ep.arrays and not np.allclose(ep.arrays["expert_actions"], ep.actions)  # noise on issued
    ck = train({"name": "t", "data": str(tmp_path / "eps"), "out": str(tmp_path / "run"),
                "feature_groups": ["gripper", "object"], "val_fraction": 0.25,
                "model": {"hidden": [8]}, "train": {"epochs": 2}}, log=lambda *_: None)
    meta = json.loads((ck / "metadata.json").read_text())
    assert set(meta["provenance"]["train_episode_ids"]).isdisjoint(meta["provenance"]["val_episode_ids"])
    assert not any(n.startswith(("normal", "shear", "contact")) for n in meta["feature_names"])
    pol = CupPolicy.load(ck)
    a = pol.act(ep.obs[0])
    assert a.shape == (1,) and 0.5 <= a[0] <= 40.0
