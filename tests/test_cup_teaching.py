"""Teaching workflow end to end (no hardware): TEACH recording -> snapshot -> candidate training ->
activation gate -> dev evaluation -> activation -> AUTONOMOUS with human input ignored; glove grip
through the same action interface."""
import json

import numpy as np
import pytest

import dermis.cup.teaching as teaching
from dermis.cup.controllers import GloveGrip, KeyboardGrip, PolicyController
from dermis.cup.env import CupHoldEnv
from dermis.cup.live import AUTONOMOUS, TEACH, LiveSession
from dermis.cup.train import CupPolicy
from dermis.data.episodes import EpisodeStore


@pytest.fixture()
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(teaching, "CANDIDATES", tmp_path / "candidates")
    monkeypatch.setattr(teaching, "ACTIVE", tmp_path / "active.json")
    monkeypatch.setattr(teaching, "DATASETS", tmp_path / "datasets")
    return tmp_path


def record_keyboard_episode(env, seed, out):
    sess = LiveSession(env, TEACH, seed, teach_input=KeyboardGrip(), teach_dir=str(out))
    sess.do("record")
    sess.human_input(set(), {"3"})                     # 12 N preset
    for _ in range(15):
        sess.tick()
    sess.do("pour")
    k = 0
    while not env.done and k < 3000:
        sess.human_input({"up"} if k % 25 == 0 else set(), set())
        sess.tick()
        k += 1
    return sess.saved


@pytest.mark.slow
def test_teach_snapshot_train_activate_autonomous(paths):
    env = CupHoldEnv()
    teach_dir = paths / "teach"
    try:
        ids = record_keyboard_episode(env, 3_100_000, teach_dir) + record_keyboard_episode(env, 3_100_001, teach_dir)
    finally:
        env.close()
    assert len(ids) == 2
    ep = EpisodeStore(teach_dir).load(ids[0])
    assert ep.meta["source"] == "keyboard" and "host_ns" in ep.arrays and ep.meta["outcome"]["pour_t_s"] is not None
    assert np.all(np.diff(ep.arrays["host_ns"]) > 0)

    snap = teaching.snapshot([str(teach_dir)], "s1", out_root=paths / "datasets")
    man = json.loads((snap / "manifest.json").read_text())
    assert man["counts_by_source"] == {"keyboard": 2} and all(len(e["sha256_npz"]) == 16 for e in man["episodes"])
    with pytest.raises(SystemExit):
        teaching.snapshot([str(teach_dir)], "s1", out_root=paths / "datasets")   # immutable

    teaching.start_training(str(snap), "c1", "configs/cup/bc_contact.yaml", epochs=2, foreground=True)
    st = teaching.status("c1")
    assert st["state"] == "done" and st["n_episodes"] == 2 and st["train_seconds"] > 0
    with pytest.raises(SystemExit):
        teaching.activate("c1")                                               # no dev evaluation yet
    suite = paths / "suite.json"
    suite.write_text(json.dumps({"name": "t", "generate": {"seed_start": 1_000_000, "count": 2},
                                 "task_config": {"damage_model": "instant+sustained"}}))
    teaching.evaluate_candidate("c1", str(suite), 1)
    rec = teaching.activate("c1")
    assert rec["snapshot_counts"] == {"keyboard": 2} and not rec["forced_without_eval"]
    ck = teaching.active_checkpoint()
    assert ck and ck.endswith("c1/checkpoint")

    pol = CupPolicy.load(ck)
    assert pol.meta["provenance"]["sources"] == ["keyboard"]
    env = CupHoldEnv()
    try:
        sess = LiveSession(env, AUTONOMOUS, 3_200_000, policies={"active": PolicyController(pol)})
        assert sess.human is None
        sess.do("pour")
        efforts = []
        for _ in range(60):
            sess.human_input({"up", "w"}, {"9"})
            sess.tick()
            efforts.append(env.effort)
        assert sess.ignored_human_events == 60 and all(0.5 <= e <= 40 for e in efforts)
    finally:
        env.close()

    # fine-tuning from a pretrained checkpoint discloses its sources
    teaching.start_training(str(snap), "c2", "configs/cup/bc_contact.yaml", init_from=ck, epochs=1, foreground=True)
    meta = json.loads((paths / "candidates" / "c2" / "checkpoint" / "metadata.json").read_text())
    pre = meta["provenance"]["pretrained_from"]
    assert pre["checkpoint"] == ck and pre["sources"] == ["keyboard"] and meta["normalizer_fitted_on"].startswith("reused")


class FakeRunner:
    """Stands in for TelemetryRunner: serves snapshots from the test Stream."""
    def __init__(self, stream):
        self.stream, self.snap, self.press = stream, None, 0.0

    def latest(self):
        self.snap = self.stream.frame(press=self.press, dt_s=1 / 30)
        return self.snap

    def source_stale(self):
        return False


def test_glove_replaces_keyboard_through_same_action_interface(tmp_path):
    from test_glove_grip import Stream, calibrate
    from dermis.glove.grip import GripControl, GripMappingConfig
    import time
    cfg = GripMappingConfig()
    s = Stream()
    s.t_ns = time.monotonic_ns()
    cal = calibrate(s, cfg)
    s.t_ns = time.monotonic_ns()
    runner = FakeRunner(s)
    with pytest.raises(PermissionError):
        GloveGrip(runner, GripControl(cfg, cal))
    glove = GloveGrip(runner, GripControl(cfg, cal), enabled=True)
    env = CupHoldEnv()
    try:
        sess = LiveSession(env, TEACH, 3_300_000, teach_input=glove, teach_dir=str(tmp_path / "t"))
        sess.do("record")
        for _ in range(5):                                   # not armed yet: effort held at the setup value
            s.t_ns = time.monotonic_ns()
            sess.tick()
        assert glove.status == "DISARMED" and env.effort == 8.0
        s.t_ns = time.monotonic_ns()
        sess.human_input(set(), {"a"})                       # operator arms
        assert glove.control.state == "ARMED", glove.last_message
        runner.press = 1.0
        for _ in range(40):
            s.t_ns = time.monotonic_ns()
            sess.tick()
        assert env.effort > 15.0                             # press reference -> near out_max (20 N)
        sess.do("reset")
    finally:
        env.close()
    # AUTONOMOUS never constructs a human input, glove or keyboard
    env = CupHoldEnv()
    try:
        class P:
            meta = {"name": "p", "feature_groups": ["gripper"], "provenance": {"train_episode_ids": [], "sources": []}}
            def reset(self): pass
            def act(self, o): return np.array([9.0], np.float32)
        a = LiveSession(env, AUTONOMOUS, 3_300_001, teach_input=glove, policies={"p": PolicyController(P())})
        assert a.human is None
        a.human_input(set(), {"a"})
        a.tick()
        assert env.effort == 9.0
    finally:
        env.close()
