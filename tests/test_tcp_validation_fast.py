"""fast_valid() must agree with the jsonschema validator on every frame (it only skips the slow path)."""
import copy
import json
import random

from dermis.server.dev_sender import SyntheticGlove
from dermis.server.validation import fast_valid, frame_validator

BAD = [None, True, False, -1, 0, 1.5, 2.0, 16777216, 256, "x", "", [], [1, 2], [1, 2, 3], [1, "a", 3],
       {}, {"a": 1}, ["STALE"], ["NOT_A_FLAG"], ["MARKER"], float("nan")]


def _paths(o, p=()):
    yield p
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _paths(v, p + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _paths(v, p + (i,))


def _set(o, path, v):
    for k in path[:-1]:
        o = o[k]
    o[path[-1]] = v


def _del(o, path):
    for k in path[:-1]:
        o = o[k]
    del o[path[-1]]


def test_fast_validator_agrees_with_jsonschema(decoded_frame):
    v = frame_validator()
    rng = random.Random(0)
    bases = [decoded_frame, SyntheticGlove(stationary_s=0).frame_at(1.0, 5),
             SyntheticGlove(stationary_s=0).frame_at(1.0, 6, faults={"null_index"})]
    checked = 0
    for base in bases:
        paths = [p for p in _paths(base) if p]
        assert fast_valid(base) and v.is_valid(base)
        for p in paths:
            for val in BAD:
                f = copy.deepcopy(base)
                _set(f, p, val)
                assert fast_valid(f) == v.is_valid(f) or (fast_valid(f) is False and v.is_valid(f)), (p, val)
                assert not (fast_valid(f) and not v.is_valid(f)), (p, val)     # never accept an invalid frame
                checked += 1
            if isinstance(p[-1], str):
                f = copy.deepcopy(base)
                _del(f, p)
                assert not (fast_valid(f) and not v.is_valid(f)), ("del", p)
                checked += 1
        for _ in range(300):                                 # renamed pods
            f = copy.deepcopy(base)
            name = rng.choice(["mag7", "imu12", "wrist", "elbow", "imu", "mag-1", "index_tip"])
            f["pods"][name] = copy.deepcopy(next(iter(base["pods"].values())))
            assert not (fast_valid(f) and not v.is_valid(f)), name
            checked += 1
    assert checked > 2000


def test_fast_validator_accepts_what_jsonschema_accepts(decoded_frame):
    """No needless slow path: every valid frame the sender makes passes the fast check."""
    g = SyntheticGlove(stationary_s=1.0, profile="punch")
    for i in range(200):
        f = json.loads(json.dumps(g.frame_at(i / 30, i)))
        assert fast_valid(f) and frame_validator().is_valid(f)
