"""Recorder round trip, episode-level splits, history windows, normalization, checkpoint consistency."""
import json

import numpy as np
import pytest

from dermis.bc.dataset import (Normalizer, build_arrays, feature_names, history_windows, scale_actions,
                               split_episodes, unscale_actions)
from dermis.bc.model import BCPolicy
from dermis.bc.train import train
from dermis.data.episodes import EpisodeRecorder, EpisodeStore, load_episode
from dermis.sim.env import ACTION_HIGH, ACTION_LOW, OBS_NAMES, obs_indices

D = len(OBS_NAMES)


def fake_episode(k, T, source="scripted", success=True, expert=True):
    rec = EpisodeRecorder(source, {"control_hz": 30}, {"seed": k, "object_xy": [0.5, 0], "target_xy": [0.5, 0.1],
                                                      "friction": 0.5}, 1 / 30, episode_id=f"{source}-s{k:07d}")
    rng = np.random.default_rng(k)
    for t in range(T):
        obs = np.full(D, 1000.0 * k + t, np.float32)          # encodes (episode, tick) in every feature
        act = rng.uniform(ACTION_LOW, ACTION_HIGH).astype(np.float32)
        rec.record(obs, act, t / 30, act + 0.001 if expert else None, tag="ACTIVE" if t % 2 else "PAUSED_STALE")
        rec.log_step({"force_l": 1.0, "force_r": 2.0, "table_force": 0.0, "held": False, "obj_pos": [0, 0, 0]})
    return rec.finish(np.zeros(D), {"success": success, "termination": "success" if success else "timeout"})


def test_recorder_round_trip(tmp_path):
    ep = fake_episode(3, 25)
    store = EpisodeStore(tmp_path)
    store.save(ep)
    back = load_episode(tmp_path / f"{ep.id}.json")
    assert back.meta == json.loads(json.dumps(ep.meta))
    for k, v in ep.arrays.items():
        np.testing.assert_array_equal(back.arrays[k], v)
    assert back.meta["source"] == "scripted" and back.meta["label"] == "SCRIPTED"
    assert back.meta["obs_names"] == list(OBS_NAMES) and back.meta["seed"] == 3
    assert list(back.arrays["tags"][:2]) == ["PAUSED_STALE", "ACTIVE"]
    assert store.ids() == [ep.id]


def test_recorder_rejects_bad_source_and_misordered_calls():
    with pytest.raises(ValueError):
        EpisodeRecorder("autonomous", {}, {}, 0.1)
    rec = EpisodeRecorder("keyboard", {}, {"seed": 1}, 0.1)
    rec.record(np.zeros(D), np.zeros(5), 0.0)
    with pytest.raises(RuntimeError):
        rec.record(np.zeros(D), np.zeros(5), 0.1)          # missing log_step: obs/action pairing broken


def test_split_is_by_whole_episode():
    ids = [f"e{i}" for i in range(20)]
    tr, va = split_episodes(ids, 0.25, seed=0)
    assert set(tr).isdisjoint(va) and set(tr) | set(va) == set(ids) and len(va) == 5
    assert split_episodes(ids, 0.25, seed=0) == (tr, va)            # reproducible
    assert split_episodes(ids, 0.25, seed=1) != (tr, va)


def test_history_windows_never_cross_episodes():
    eps = [fake_episode(k, T) for k, T in ((1, 7), (2, 3), (3, 12))]
    norm = Normalizer(np.zeros(D, np.float32), np.ones(D, np.float32))
    H = 4
    X, Y, owner = build_arrays(eps, ["proprio", "object", "target", "contact"], H, norm, "actions")
    assert len(X) == 7 + 3 + 12
    for row, k in zip(X, owner):
        ep_ids = np.floor(row.reshape(H, -1)[:, 0] / 1000.0)
        assert (ep_ids == [1, 2, 3][k]).all()                          # every frame from its own episode
    w = history_windows(np.arange(5.0)[:, None], 3)
    np.testing.assert_array_equal(w, [[0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]])


def test_feature_names_and_groups():
    names = feature_names(["proprio", "contact"], 2)
    assert names[0].endswith("@t-1") and names[-1] == "contact_r@t"
    assert len(names) == 2 * len(obs_indices(["proprio", "contact"]))
    assert "force_l" not in [OBS_NAMES[i] for i in obs_indices(["proprio", "object", "target"])]


def test_action_scaling_round_trip():
    a = np.random.default_rng(0).uniform(ACTION_LOW, ACTION_HIGH, (10, 5))
    s = scale_actions(a)
    assert (np.abs(s) <= 1 + 1e-6).all()
    np.testing.assert_allclose(unscale_actions(s), a, rtol=1e-5)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("bc")
    store = EpisodeStore(root / "eps")
    for k in range(6):
        store.save(fake_episode(k, 20 + k))
    store.save(fake_episode(99, 10, success=False))                     # filtered by success_only
    ckpt = train({"name": "t", "data": str(root / "eps"), "out": str(root / "run"), "history": 3,
                  "val_fraction": 0.34, "model": {"hidden": [16]}, "train": {"epochs": 3}}, log=lambda *_: None)
    return root, ckpt


def test_normalizer_fit_on_training_episodes_only(tiny_run):
    root, ckpt = tiny_run
    meta = json.loads((ckpt / "metadata.json").read_text())
    prov = meta["provenance"]
    assert set(prov["train_episode_ids"]).isdisjoint(prov["val_episode_ids"])
    assert "scripted-s0000099" not in prov["train_episode_ids"] + prov["val_episode_ids"]
    store = EpisodeStore(root / "eps")
    cols = obs_indices(meta["feature_groups"])
    train_rows = np.concatenate([store.load(i).obs[:, cols] for i in prov["train_episode_ids"]])
    np.testing.assert_allclose(meta["normalizer"]["mean"], train_rows.mean(0), rtol=1e-5)
    for k in ("feature_names", "action_names", "train_config", "label"):
        assert k in meta
    assert prov["data_digest"] and prov["train_seeds"]


def test_checkpoint_reload_is_consistent(tiny_run):
    root, ckpt = tiny_run
    ep = EpisodeStore(root / "eps").load("scripted-s0000002")
    p1, p2 = BCPolicy.load(ckpt), BCPolicy.load(ckpt)
    batch = p1.predict_batch(ep.obs)
    p2.reset()
    online = np.array([p2.act(o) for o in ep.obs])
    np.testing.assert_allclose(online, np.clip(batch, ACTION_LOW, ACTION_HIGH), atol=1e-5)
    assert (online >= ACTION_LOW).all() and (online <= ACTION_HIGH).all()
    p2.reset()
    np.testing.assert_allclose(p2.act(ep.obs[0]), online[0], atol=1e-6)    # reset clears history


def test_training_is_reproducible(tiny_run, tmp_path):
    root, ckpt = tiny_run
    cfg = json.loads((ckpt / "metadata.json").read_text())["train_config"]
    ckpt2 = train({**cfg, "out": str(tmp_path / "run2")}, log=lambda *_: None)
    obs = EpisodeStore(root / "eps").load("scripted-s0000001").obs
    np.testing.assert_allclose(BCPolicy.load(ckpt).predict_batch(obs), BCPolicy.load(ckpt2).predict_batch(obs),
                               atol=1e-6)
