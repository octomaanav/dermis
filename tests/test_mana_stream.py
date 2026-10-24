"""The Mana stream (docs/mana/dermis_sim_stream.schema.json) is schema-valid and shows exactly what the
simulator did in each scripted check: one thing moves at a time, stale frames hold the pose."""
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from dermis.server.mana import CHECKS, export_check_stream

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stream(tmp_path_factory):
    p = tmp_path_factory.mktemp("mana") / "checks.jsonl"
    export_check_stream(str(p))
    return [json.loads(l) for l in p.read_text().splitlines()]


def seg(frames, a, b):
    return [f for f in frames if a <= f["t_sim_s"] < b]


def test_schema_valid(stream):
    v = Draft202012Validator(json.loads((ROOT / "docs/mana/dermis_sim_stream.schema.json").read_text()))
    assert stream[0]["type"] == "hello" and all(f["type"] == "frame" for f in stream[1:])
    for m in stream:
        assert not list(v.iter_errors(m))
    seqs = [f["seq"] for f in stream[1:]]
    assert seqs == list(range(len(seqs)))


def test_each_check_moves_only_what_it_says(stream):
    fr = stream[1:]
    P = lambda s: np.array([f["base"]["pos"] for f in s])
    Q = lambda s: np.array([f["base"]["quat"] for f in s])
    J = lambda s: np.array([[j["q"] for j in f["joints"]] for f in s])
    for (a, b, _), axis in zip(CHECKS[1:4], range(3)):          # palm x, y, z
        r = np.ptp(P(seg(fr, a, b)), axis=0)
        assert r[axis] > 0.07 and np.delete(r, axis).max() < 1e-6
        assert np.ptp(J(seg(fr, a, b))) < 1e-6 and np.ptp(Q(seg(fr, a, b)), axis=0).max() < 1e-6
    roll = seg(fr, 7, 9)                                          # wrist roll: orientation changes, fingers don't
    assert np.ptp(Q(roll), axis=0).max() > 0.1 and np.ptp(J(roll)) < 1e-6
    idx = J(seg(fr, 9, 11))                                       # index only: joints 3..5 move, others still
    assert np.ptp(idx[:, 3:6], axis=0).min() > 0.5 and np.ptp(idx[:, [0, 1, 2, 6, 7, 8]], axis=0).max() < 1e-6
    links = np.array([[l["pos"] for l in f["links"]] for f in seg(fr, 9, 11)])
    moved = np.ptp(links, axis=0).max(axis=1) > 1e-6              # only index links (4..7) move in world
    assert set(np.where(moved)[0]) <= {5, 6, 7} and moved[5:8].any()


def test_measured_vs_commanded_and_stale_hold(stream):
    fr = stream[1:]
    fist = seg(fr, 11, 13)
    for f in fist:                                                # fingers reach their command; the thumb may be
        for j in f["joints"]:                                     # stopped short by the index (measured < cmd)
            assert j["q"] <= j["q_cmd"] + 1e-6
            if not j["name"].startswith("thumb"):
                assert j["q"] == pytest.approx(j["q_cmd"], abs=1e-4)
    stale = seg(fr, 13, 14)
    assert stale and all(f["stale"] for f in stale)
    first = stale[0]
    for f in stale:
        assert f["base"] == first["base"] and [j["q"] for j in f["joints"]] == [j["q"] for j in first["joints"]]
    assert stale[-1]["source"]["telemetry_age_ms"] > 900
