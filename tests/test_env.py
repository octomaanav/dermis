import numpy as np
import pytest

from dermis.sim.env import (ACTION_HIGH, ACTION_LOW, OBS_NAMES, GraspLiftPlaceEnv, Scenario, TaskConfig,
                            clip_action)
from dermis.sim.expert import ScriptedExpert


@pytest.fixture(scope="module")
def env():
    e = GraspLiftPlaceEnv()
    yield e
    e.close()


def test_reset_is_deterministic_and_finite(env):
    o1 = env.reset(42)
    s1 = env.scenario
    o2 = env.reset(42)
    assert env.scenario == s1
    np.testing.assert_array_equal(o1, o2)
    assert o1.shape == (len(OBS_NAMES),) and np.isfinite(o1).all()
    assert env.reset(43) is not None and env.scenario != s1


def test_scenario_within_configured_ranges():
    cfg = TaskConfig()
    for seed in range(50):
        s = Scenario.sample(seed, cfg)
        assert cfg.object_x_range[0] <= s.object_xy[0] <= cfg.object_x_range[1]
        assert cfg.target_y_range[0] <= s.target_xy[1] <= cfg.target_y_range[1]
        assert cfg.friction_range[0] <= s.friction <= cfg.friction_range[1]


def test_clip_action_bounds_and_nan():
    a = clip_action([10, -10, np.nan, 1.0, 1e6])
    assert (a >= ACTION_LOW).all() and (a <= ACTION_HIGH).all()
    assert a[2] == 0.0 and a[3] == ACTION_HIGH[3] and a[4] == ACTION_HIGH[4]


def test_bounded_motion_and_finite_obs_under_extreme_actions(env):
    env.reset(1)
    rng = np.random.default_rng(0)
    prev = env.privileged_state()["ee"]
    for _ in range(60):
        obs, done, info = env.step(rng.uniform(-100, 100, 5))
        ee = env.privileged_state()["ee"]
        assert np.isfinite(obs).all()
        assert np.abs(ee - prev).max() < env.cfg.max_speed * env.cfg.dt + 0.02     # IK tracking slack
        for i, (lo, hi) in enumerate(env.cfg.workspace):
            assert lo - 0.02 <= env.target_pos[i] <= hi + 0.02
        prev = ee
        if done:
            break


def test_timeout_termination():
    e = GraspLiftPlaceEnv(TaskConfig(timeout_s=0.5))
    try:
        e.reset(3)
        done, n = False, 0
        while not done:
            _, done, info = e.step([0, 0, 0, 0.08, 10])
            n += 1
        assert info["termination"] == "timeout" and n == 15 and not e.stats.success
        with pytest.raises(RuntimeError):
            e.step([0, 0, 0, 0.08, 10])
    finally:
        e.close()


def test_scripted_expert_completes_task_with_real_contact(env):
    """Sanity check: the task is physically achievable; object is never attached."""
    ex = ScriptedExpert()
    for seed in (5, 6):
        env.reset(seed)
        ex.reset()
        done, held_force = False, []
        while not done:
            _, done, info = env.step(ex.act(env))
            if info["held"]:
                held_force.append(min(info["force_l"], info["force_r"]))
        assert env.stats.success and env.stats.drops == 0 and env.stats.excess_force_ticks == 0
        assert held_force and min(held_force) > 0.05            # lifted only while both fingers press


def test_low_grip_effort_cannot_lift(env):
    env.reset(Scenario(1, (0.55, -0.1), (0.55, 0.12), 0.4))
    ex = ScriptedExpert()
    done = False
    while not done and env.stats.ticks < 200:
        a = ex.act(env)
        a[4] = 2.0                                # minimum effort: friction cannot carry 0.2 kg
        _, done, _ = env.step(a)
    assert not env.stats.was_lifted
