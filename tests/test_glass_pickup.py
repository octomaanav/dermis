"""Real-time pick-up: grip-triggered state machine + physics, glove-frame grip extraction, calibration."""
import copy

import numpy as np
import pybullet as p
import pytest

from dermis.glass.pickup import PickupConfig, PickupSession
from dermis.glass.pickup_live import PODS, PROFILES, calibrate, pod_forces
from tests.test_glass import DEVICE_FRAME


def run_profile(level: dict, seconds: float = 8.5, release_at: float = 5.0):
    s = PickupSession()
    states, max_h, constraint_ok = [], -1.0, True
    try:
        for i in range(int(seconds * 50)):
            t = i / 50
            a = 0.0 if (t < 0.5 or t > release_at) else min(1.0, (t - 0.5) / 0.3)
            r = s.tick({k: v * a for k, v in level.items()})
            max_h = max(max_h, r["cup_height_cm"])
            if not states or states[-1] != r["state"]:
                states.append(r["state"])
            for c in range(p.getNumConstraints(physicsClientId=s.sim.client)):     # only the hand carrier
                info = p.getConstraintInfo(p.getConstraintUniqueId(c, physicsClientId=s.sim.client), physicsClientId=s.sim.client)
                constraint_ok &= info[0] == s.sim.hand and info[2] == -1
        assert constraint_ok                                    # the cup is never constrained
        return states, max_h, s.resets, [e["event"] for e in s.trig.events]
    finally:
        s.close()


def test_normal_grip_lifts_holds_and_sets_down():
    states, max_h, resets, _ = run_profile(PROFILES["normal"])
    assert states == ["OPEN", "CLOSING", "LIFTING", "HELD", "LOWERING", "OPENING", "OPEN"]
    assert max_h > 9.0 and resets == 0                      # cup really went up ~10 cm with the hand


def test_weak_grip_drops_and_crushing_grip_cracks():
    states, max_h, resets, ev = run_profile(PROFILES["weak"])
    assert "DROPPED" in states and max_h < 5.0 and "reset" in ev
    states, _, _, ev = run_profile(PROFILES["crush"])
    assert "CRACKED" in states and "HELD" not in states and "reset" in ev


def test_no_attachment_only_the_carrier_constraint():
    s = PickupSession()
    try:
        for i in range(150):
            s.tick({k: v for k, v in PROFILES["normal"].items()})
        n = p.getNumConstraints(physicsClientId=s.sim.client)
        assert n == 1
        info = p.getConstraintInfo(p.getConstraintUniqueId(0, physicsClientId=s.sim.client), physicsClientId=s.sim.client)
        assert info[0] == s.sim.hand and s.sim.cup not in (info[0], info[2])
    finally:
        s.close()


def test_stale_stream_keeps_last_command():
    s = PickupSession()
    try:
        for _ in range(60):
            s.tick(PROFILES["normal"])
        state, eff = s.trig.state, s.trig.efforts
        for _ in range(10):
            s.tick(None)                                        # no frames: nothing invented, nothing changed
        assert s.trig.state == state and s.trig.efforts == eff
    finally:
        s.close()


def test_pod_forces_uses_force_n_and_falls_back_to_proxy():
    f, mode = pod_forces(DEVICE_FRAME, None, 160.0)
    assert mode == "force_n" and f["index_tip"] == pytest.approx(7.683199) and f["middle_tip"] == pytest.approx(9.80665)
    old = copy.deepcopy(DEVICE_FRAME)                           # current repo firmware: no force_n
    for k in PODS:
        for key in ("force_n", "force_valid"):
            old["pods"][k]["mag"].pop(key)
    base = {k: np.array(old["pods"][k]["mag"]["b_ut"]) for k in PODS}
    pressed = copy.deepcopy(old)
    pressed["pods"]["thumb_tip"]["mag"]["b_ut"][2] -= 320.0
    pressed["pods"]["index_tip"]["mag"]["valid"] = False
    f, mode = pod_forces(pressed, base, 160.0)
    assert mode == "proxy" and f["thumb_tip"] == pytest.approx(2.0) and f["index_tip"] is None and f["middle_tip"] == 0.0


def test_calibration_maps_normal_hold_to_mid_window(tmp_path):
    from dermis.glass.sensors import SyntheticGlove

    class Fake:
        def __init__(self):
            self.g, self.level = SyntheticGlove(1), 0.0

        def poll(self):
            return [self.g.make({k: {"normal": self.level * PROFILES["normal"][k], "shear_z": 0, "shear_h": 0} for k in PODS})]

    src = Fake()
    steps = iter([0.0, 1.0])
    cal = calibrate(src, str(tmp_path / "cal.json"), open_s=0.2, hold_s=0.2,
                    prompt=lambda m: setattr(src, "level", next(steps)), wait_s=0.2)
    assert cal["mode"] == "force_n"
    assert cal["grip_gain"] * 4.5 == pytest.approx(0.054, rel=0.05)    # median normal-hold finger force is 4.5 N


def test_null_force_n_from_untared_firmware_uses_proxy():
    """Firmware v3 sends "force_n": null with force_valid false until tared and spanned on the device."""
    fr = copy.deepcopy(DEVICE_FRAME)
    for k in PODS:
        fr["pods"][k]["mag"].update(force_n=None, force_valid=False, flags=["FORCE_SPAN_REQUIRED"])
    base = {k: np.array(fr["pods"][k]["mag"]["b_ut"]) for k in PODS}
    fr["pods"]["index_tip"]["mag"]["b_ut"][0] += 480.0
    f, mode = pod_forces(fr, base, 160.0)
    assert mode == "proxy" and f["index_tip"] == pytest.approx(3.0) and f["thumb_tip"] == 0.0


def test_calibration_explains_a_silent_stream(tmp_path):
    class Silent:
        status, stats = "connected ws://x", {"frames": 0, "bad_json": 7, "connects": 1, "last_bad": "Expecting value"}

        def poll(self):
            return []

    with pytest.raises(RuntimeError, match="bad_json"):
        calibrate(Silent(), str(tmp_path / "c.json"), prompt=lambda m: None, wait_s=0.1)
