"""glass_hold: provisional-force glove schema, SYNTHETIC frames, and the physics behind the three demo lanes."""
import copy
import json

import numpy as np
import pybullet as p
import pytest

from dermis.glass.controllers import scripted
from dermis.glass.data import run_episode
from dermis.glass.env import GlassEnv, GlassScenario
from dermis.glass.features import NAMES, frame_features
from dermis.glass.hand3 import PODS
from dermis.glass.schema import FORCE_MAX_N, POD_ORDER, SchemaError, parse, validate
from dermis.glass.sensors import SyntheticGlove

DEVICE_FRAME = json.loads("""{"schema_version":1,"device_id":"DERMIS-74DFAEB1E590","frame":1761,"t_device_us":184921209,"t_host_ns":null,"config_id":"dermis-4mlx-1imu-force-provisional-v1","flags":[],"pods":{"index_tip":{"pod_id":"P-0001","mag":{"ch":0,"b_ut":[619.9741,-1612.413,9672.256],"temp_c":26.5708,"force_n":7.683199,"force_valid":true,"t_us":184923516,"status":35,"flags":["FORCE_PROVISIONAL"],"valid":true}},"thumb_tip":{"pod_id":"P-0002","mag":{"ch":1,"b_ut":[-533.466,-595.944,-5172.992],"temp_c":27.0354,"force_n":0,"force_valid":true,"t_us":184923637,"status":3,"flags":["FORCE_PROVISIONAL","FORCE_CLAMPED_LOW"],"valid":true}},"middle_tip":{"pod_id":"P-0003","mag":{"ch":2,"b_ut":[2835.54,-507.033,7434.24],"temp_c":25.11062,"force_n":9.80665,"force_valid":true,"t_us":184923753,"status":3,"flags":["FORCE_PROVISIONAL","FORCE_CLAMPED_HIGH"],"valid":true}},"thenar_apb":{"pod_id":"P-0004","mag":{"ch":3,"b_ut":[389.286,264.33,-6733.408],"temp_c":26.70354,"force_n":0,"force_valid":true,"t_us":184923872,"status":3,"flags":["FORCE_PROVISIONAL","FORCE_CLAMPED_LOW"],"valid":true}}},"imu":{"imu_id":"IMU-0001","ch":0,"acc_mps2":[1.908179,0.643442,9.488221],"gyr_rads":[-0.004789,0.002128,-0.005321],"temp_c":27.01563,"t_us":184923988,"sensortime":4688984,"flags":[],"valid":true}}""")


def test_device_frame_parses():
    f = parse(DEVICE_FRAME)
    i = {k: n for n, k in enumerate(POD_ORDER)}
    assert not f.synthetic and f.frame == 1761
    assert f.force_n[i["index_tip"]] == pytest.approx(7.683199)
    assert f.clamped_high[i["middle_tip"]] and f.clamped_low[i["thumb_tip"]] and f.clamped_low[i["thenar_apb"]]
    np.testing.assert_allclose(f.b_ut[i["thenar_apb"]], [389.286, 264.33, -6733.408])
    np.testing.assert_allclose(f.acc_mps2, [1.908179, 0.643442, 9.488221])


@pytest.mark.parametrize("mutate", [
    lambda d: d["pods"].pop("thenar_apb"),
    lambda d: d.pop("imu"),
    lambda d: d["pods"]["index_tip"]["mag"].pop("force_n"),
    lambda d: d.__setitem__("config_id", 3),
    lambda d: d.__setitem__("schema_version", 2),
    lambda d: d["pods"]["thumb_tip"]["mag"].__setitem__("b_ut", [1.0, 2.0]),
    lambda d: d["pods"]["thumb_tip"]["mag"].__setitem__("force_valid", "yes"),
])
def test_malformed_frames_rejected(mutate):
    d = copy.deepcopy(DEVICE_FRAME)
    mutate(d)
    with pytest.raises(SchemaError):
        validate(d)


def test_invalid_force_becomes_nan():
    d = copy.deepcopy(DEVICE_FRAME)
    d["pods"]["index_tip"]["mag"]["force_valid"] = False
    assert np.isnan(parse(d).force_n[POD_ORDER.index("index_tip")])


def test_synthetic_frames_match_schema_and_clamp():
    g = SyntheticGlove(0)
    c = {k: {"normal": 0.0, "shear_z": 0.0, "shear_h": 0.0} for k in PODS}
    c["thumb_tip"]["normal"] = 14.0
    f = g.make(c)
    validate(f)
    assert f["flags"] == ["SYNTHETIC"] and f["device_id"].startswith("SYNTHETIC")
    thumb = f["pods"]["thumb_tip"]["mag"]
    assert thumb["force_n"] == FORCE_MAX_N and "FORCE_CLAMPED_HIGH" in thumb["flags"]
    assert all("FORCE_PROVISIONAL" in f["pods"][k]["mag"]["flags"] for k in PODS)
    assert set(f["pods"]) == {"index_tip", "thumb_tip", "middle_tip", "thenar_apb"} and "imu" not in f["pods"]["index_tip"]
    assert frame_features(f, f, 0.04).shape == (len(NAMES),)


@pytest.fixture(scope="module")
def env():
    e = GlassEnv()
    yield e
    e.close()


SEED = 30_000


def test_too_little_force_drops_during_pour(env):
    r = run_episode(env, scripted("too_weak"), GlassScenario.sample(SEED))
    assert r["termination"] == "drop"
    assert env.water > 0.0 and env.t < env.end_t          # fell while water was being added, not before
    assert p.getNumConstraints(physicsClientId=env.client) == 0


def test_too_much_force_cracks_during_pour(env):
    r = run_episode(env, scripted("over_squeeze"), GlassScenario.sample(SEED))
    assert r["termination"] == "crack" and env.water > 0.0
    assert r["max_pod_normal_n"] > env.cfg.crack_force_n
    assert env.drop_m() < env.cfg.drop_m                 # still in the hand: crack is the force threshold, not a drop


def test_expert_holds_with_all_four_pods_and_force_balance(env):
    sc = GlassScenario.sample(SEED)
    env.reset(sc)
    ctrl = scripted("load_expert")
    ctrl.reset(env)
    balance = []
    while not env.done:
        env.step(ctrl.act(env))
        st = env.state()
        if env.t > sc.pour_start_s + sc.pour_s + 0.5 and not env.done:      # quasi-static, pour finished
            balance.append(abs(st["vertical_support_n"] - st["weight_n"]))
    assert env.termination == "success"
    assert p.getNumConstraints(physicsClientId=env.client) == 0
    assert all(env.contact[k]["normal"] > 0.2 for k in PODS), env.contact   # thumb, index, middle and thenar all carry load
    assert np.percentile(balance, 95) < 0.05                                  # friction on the glass = its weight
    assert max(env.peak_normal.values()) < env.cfg.crack_force_n
