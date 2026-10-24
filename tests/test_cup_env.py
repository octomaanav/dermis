"""fragile_cup: contact physics, force accounting, outcome detection, schedules."""
import numpy as np
import pybullet as p
import pytest

from dermis.cup.env import (FINGER_JOINTS, OBS_NAMES, CupConfig, CupHoldEnv, CupScenario, clip_effort)
from dermis.cup.schedule import LoadSchedule


@pytest.fixture(scope="module")
def env():
    e = CupHoldEnv()
    yield e
    e.close()


def scen(load=0.0, mu=0.5, mass=0.1, seed=7, ramp_s=1.0, hold_s=2.0):
    segs = [("ramp", 0.0, ramp_s, 0.0, load), ("pause", ramp_s, ramp_s + hold_s, load, load)] if load else \
           [("pause", 0.0, hold_s, 0.0, 0.0)]
    return CupScenario(seed, mass, mu, LoadSchedule(segs, []))


def run(env, sc, effort, cfg_override=None, pour=0.5):
    env.reset(sc, pour_at_s=pour)
    infos = []
    while not env.done:
        _, _, info = env.step([effort])
        infos.append(info)
    return env.stats, infos


def test_no_attachment_and_setup_is_valid(env):
    env.reset(scen(), pour_at_s=0.5)
    assert p.getNumConstraints(physicsClientId=env.client) == 0
    assert p.getDynamicsInfo(env.cup, -1, physicsClientId=env.client)[0] > 0       # free dynamic body
    assert env.stats.termination is None
    o = env.observe()
    i = {n: k for k, n in enumerate(OBS_NAMES)}
    assert o[i["contact_l"]] == 1 and o[i["contact_r"]] == 1
    assert env._grasp_rel[2] < 0                                    # cup centre below the grasp point


def test_holding_depends_on_finger_friction(env):
    held, _ = run(env, scen(), 10.0)
    assert held.termination == "success"
    dropped, _ = run(env, scen(mu=0.4), 0.5)                       # 2 * 0.4 * 0.5 N < m g
    assert dropped.termination == "drop" and dropped.failure_phase == "pre_pour"


def test_force_decomposition_balances_weight_plus_load(env):
    sc = scen(load=4.0, mu=0.6, mass=0.12)
    env.reset(sc, pour_at_s=0.0)
    checked = 0
    while not env.done:
        _, _, info = env.step([14.0])
        c = env._contact
        if info["phase"] == "pause":                                # quasi-static
            total_z = c["shear_z"].sum()
            assert abs(total_z - (sc.mass * 9.81 + info["load_n"])) < 0.05
            assert np.allclose(c["normal"], 14.0, atol=0.3)         # squeeze ~ commanded effort limit
            assert c["palm"] == 0.0
            checked += 1
    assert checked > 30


def test_no_double_counting(env):
    """Per-finger vectors summed == total robot->cup force computed independently from all contact points."""
    env.reset(scen(load=3.0), pour_at_s=0.0)
    for _ in range(40):
        env.step([12.0])
    fc = env._finger_forces()
    total = np.zeros(3)
    pts = p.getContactPoints(env.robot, env.cup, physicsClientId=env.client)
    assert pts
    for c in pts:
        total += -(c[9] * np.array(c[7]) + c[10] * np.array(c[11]) + c[12] * np.array(c[13]))
    np.testing.assert_allclose(fc["vec"][0] + fc["vec"][1] + 0, total, atol=1e-6)
    assert all(c[3] in FINGER_JOINTS for c in pts)


def test_applied_load_changes_object_behaviour(env):
    calm, _ = run(env, scen(load=0.0, mu=0.5), 3.0)
    loaded, infos = run(env, scen(load=6.0, mu=0.5), 3.0)
    assert calm.termination == "success"
    assert loaded.termination == "drop" and loaded.failure_phase in ("ramp", "pause")
    assert max(i["load_n"] for i in infos) > 2.0


def test_damage_detection_instant_and_sustained():
    e = CupHoldEnv(CupConfig(damage_model="instant+sustained", sustained_safe_n=8.0, damage_budget_ns=2.0))
    try:
        st, _ = run(e, scen(load=2.0), 30.0)
        assert st.termination == "damage_break" and st.peak_normal_n > 25.0
        st, _ = run(e, scen(load=2.0), 12.0)
        assert st.termination == "damage_sustained" and st.damage_ns > 2.0
        st, _ = run(e, scen(load=0.5), 6.0)
        assert st.termination == "success" and st.damage_ns == 0.0
    finally:
        e.close()


def test_reset_reproduces_identical_conditions(env):
    sc = CupScenario.sample(1_234_567, env.cfg)
    o1 = env.reset(sc, pour_at_s=1.0)
    a = [env.step([9.0])[0] for _ in range(50)]
    o2 = env.reset(CupScenario.from_dict(sc.to_dict()), pour_at_s=1.0)
    b = [env.step([9.0])[0] for _ in range(50)]
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(np.array(a), np.array(b))


def test_observation_has_no_privileged_inputs():
    for banned in ("load", "friction", "mass", "damage", "phase", "schedule", "mu"):
        assert not any(banned in n for n in OBS_NAMES)


def test_schedule_bounded_and_deterministic():
    cfg = CupConfig()
    for seed in range(30):
        s1 = CupScenario.sample(seed, cfg).schedule
        s2 = CupScenario.sample(seed, cfg).schedule
        assert s1.to_dict() == s2.to_dict()
        ts = np.linspace(-1, s1.duration + 2, 400)
        loads = np.array([s1.load(t) for t in ts])
        assert loads.min() >= 0 and loads.max() <= s1.peak + cfg.disturbance_amp + 1e-9
        assert cfg.max_load_range[0] <= s1.peak <= cfg.max_load_range[1]
        assert s1.load(-0.5) == 0.0


def test_effort_clipping():
    assert clip_effort([1e6]) == 40.0 and clip_effort([-3]) == 0.5 and clip_effort([float("nan")]) == 0.5
