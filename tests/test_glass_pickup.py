"""Real-time pick-up: grip-triggered state machine + physics, glove-frame grip extraction, calibration."""
import copy

import numpy as np
import pybullet as p
import pytest

from dermis.glass.pickup import PickupConfig, PickupSession
from dermis.glass.pickup_live import PODS, PROFILES, calibrate, pod_forces
from tests.test_glass import DEVICE_FRAME


def still_imu(t):
    return {"acc": np.array([0.0, 0.0, 9.81]) + np.random.normal(0, 0.03, 3), "gyr": np.random.normal(0, 0.005, 3)}


def moving_imu(lift_at=2.0, down_at=None):
    """Hand at rest, then a lift (push up, brake) and optionally a set-down (dip, brake), as |acc| changes."""
    def imu(t):
        r = still_imu(t)
        for t0, sign in ((lift_at, 1.0), (down_at, -1.0)):
            if t0 is not None and t0 <= t < t0 + 0.25:
                r["acc"][2] += 3.0 * sign
            elif t0 is not None and t0 + 0.25 <= t < t0 + 0.5:
                r["acc"][2] -= 3.0 * sign
        return r
    return imu


def run_profile(level: dict, seconds: float = 8.5, release_at: float = 5.0, imu=None, cfg=None):
    """imu: function of t -> reading, or None (no IMU: the lift starts on a timer after the grip settles)."""
    s = PickupSession(cfg)
    states, max_h, constraint_ok = [], -1.0, True
    try:
        for i in range(int(seconds * 50)):
            t = i / 50
            a = 0.0 if (t < 0.5 or t > release_at) else min(1.0, (t - 0.5) / 0.3)
            r = s.tick({k: v * a for k, v in level.items()}, imu(t) if imu else None)
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


def test_normal_grip_lifts_holds_and_sets_down_without_imu():
    states, max_h, resets, ev = run_profile(PROFILES["normal"])
    assert states == ["OPEN", "CLOSING", "GRIPPED", "LIFTING", "HELD", "LOWERING", "OPENING", "OPEN"]
    assert max_h > 9.0 and resets == 0 and "imu_unavailable" in ev       # cup really went up ~10 cm with the hand


def test_imu_still_keeps_cup_on_table_and_motion_lifts_it():
    states, max_h, _, ev = run_profile(PROFILES["normal"], imu=still_imu)
    assert "LIFTING" not in states and max_h < 0.5 and "imu_rest" in ev
    states, max_h, _, _ = run_profile(PROFILES["normal"], imu=moving_imu(lift_at=2.0))
    assert states[:5] == ["OPEN", "CLOSING", "GRIPPED", "LIFTING", "HELD"] and max_h > 9.0


def test_imu_set_down_dip_lowers_and_stays_down_until_new_motion():
    states, _, resets, _ = run_profile(PROFILES["normal"], seconds=9.0, release_at=8.0,
                                       imu=moving_imu(lift_at=2.0, down_at=4.5))
    assert states == ["OPEN", "CLOSING", "GRIPPED", "LIFTING", "HELD", "SETTING_DOWN", "GRIPPED", "OPENING", "OPEN"]
    assert resets == 0


def test_loose_glove_uneven_pods_still_lifts_with_combined_grip():
    loose = {"thumb_tip": 3.0, "index_tip": 1.3, "middle_tip": 0.3, "thenar_apb": 4.5}   # like the real glove, rescaled
    states, max_h, _, _ = run_profile(loose, imu=moving_imu(lift_at=2.0))
    assert "HELD" in states and "DROPPED" not in states and max_h > 9.0
    states, _, _, _ = run_profile(loose, imu=moving_imu(lift_at=2.0), cfg=PickupConfig(grip_mode="per_finger", grip_gain=0.012))
    assert "DROPPED" in states                                  # what happened before: middle/index too weak


def test_light_grip_is_not_a_grasp_and_crushing_grip_cracks():
    states, max_h, resets, ev = run_profile(PROFILES["light"])
    assert states == ["OPEN"] and max_h < 0.5
    states, _, _, ev = run_profile(PROFILES["crush"])
    assert "CRACKED" in states and "HELD" not in states and "reset" in ev


@pytest.mark.parametrize("ratio", [0.15, 0.5, 1.0, 2.0, 3.0])
def test_wide_window_light_to_three_times_normal_holds(ratio):
    """Calibrated normal hold = 1.9 N here; from a light touch to 3x that, the cup is lifted and held."""
    level = {k: v * ratio for k, v in PROFILES["normal"].items()}
    states, max_h, _, _ = run_profile(level, imu=moving_imu(lift_at=2.0))
    assert "HELD" in states and "CRACKED" not in states and "DROPPED" not in states and max_h > 9.0


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
    steps = iter([0.0, 1.0, 0.0])
    cal = calibrate(src, str(tmp_path / "cal.json"), open_s=0.2, hold_s=0.2, free_s=0.2,
                    prompt=lambda m: setattr(src, "level", next(steps)), wait_s=0.2)
    assert cal["mode"] == "force_n"
    assert cal["grip_gain"] * 1.9 == pytest.approx(0.054, rel=0.05)    # combined grip: mean of thumb 2.0 and index 1.8 N


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


class V3Glove:
    """Frames shaped like firmware v3 as seen on the real glove: palm pod named palm_center, on-device force
    'valid' but stuck at 0 (FORCE_CLAMPED_LOW) during a grip, raw field moving by `shift_ut` x level."""

    def __init__(self, shift_ut=300.0):
        self.level, self.shift = 0.0, shift_ut

    def frame(self):
        fr = copy.deepcopy(DEVICE_FRAME)
        fr["config_id"] = "dermis-4mlx-1imu-force-zero-span-v3"
        fr["pods"]["palm_center"] = fr["pods"].pop("thenar_apb")
        for name, pod in fr["pods"].items():
            pod["mag"].update(force_n=0, force_valid=True, flags=["FORCE_CAL_READY", "FORCE_CLAMPED_LOW"])
            pod["mag"]["b_ut"][2] += self.shift * self.level + np.random.normal(0, 2.0)
        return fr

    def poll(self):
        return [self.frame()]


def test_palm_center_is_read_as_thenar_apb():
    fr = V3Glove().frame()
    base = {k: np.array(fr["pods"]["palm_center" if k == "thenar_apb" else k]["mag"]["b_ut"]) for k in PODS}
    fr["pods"]["palm_center"]["mag"]["b_ut"][2] += 160.0
    f, mode = pod_forces(fr, base, 160.0, mode="proxy")
    assert mode == "proxy" and f["thenar_apb"] == pytest.approx(1.0, abs=0.05)


def test_calibration_falls_back_to_proxy_when_device_force_stays_zero(tmp_path):
    src, steps = V3Glove(), iter([0.0, 1.0, 0.0])
    cal = calibrate(src, str(tmp_path / "cal.json"), open_s=0.3, hold_s=0.3, free_s=0.3,
                    prompt=lambda m: setattr(src, "level", next(steps)), wait_s=0.1)
    assert cal["mode"] == "proxy" and cal["force_hold_n"]["index_tip"] == 0.0
    assert cal["ut_per_n"] == pytest.approx(300.0 / 4.5, rel=0.05) and "FORCE_CLAMPED_LOW" in cal["flags_during_hold"]
    from dermis.glass.pickup_live import frame_b                # the live run then follows the field, not force_n
    src.level = 0.0
    fr_open = src.frame()
    src.level = 1.0
    f, mode = pod_forces(src.frame(), frame_b(fr_open), cal["ut_per_n"], mode="proxy")
    assert mode == "proxy" and f["thumb_tip"] == pytest.approx(4.5, rel=0.1)


def test_calibration_rejects_a_hold_that_moves_nothing(tmp_path):
    src, steps = V3Glove(shift_ut=0.0), iter([0.0, 1.0, 0.0])
    with pytest.raises(RuntimeError, match="no pod's hold signal"):
        calibrate(src, str(tmp_path / "c.json"), open_s=0.3, hold_s=0.3, free_s=0.3,
                  prompt=lambda m: setattr(src, "level", next(steps)), wait_s=0.1)


# -- activation from the glove calibration export (the glove at rest must never grip) -------------------
from dermis.glass.glove_cal import GripSensor, load_export  # noqa: E402

EXPORT = load_export("tests/fixtures/glove_calibration_example.json")
CH_NAME = {0: "palm_center", 1: "middle_tip", 2: "thumb_tip", 3: "index_tip"}     # firmware v3 order


class CalGlove:
    """Frames like the real glove: rest field from the export, noise at the exported level, optional slow
    drift across each pad's load axis, presses along it, and an IMU at rest."""

    def __init__(self, drift_ut_s=0.0, seed=0):
        self.rng, self.t, self.drift, self.press = np.random.default_rng(seed), 0.0, drift_ut_s, {}
        self.dt = 0.1

    def frame(self):
        self.t += self.dt
        pods = {}
        for ch, name in CH_NAME.items():
            pod = {0: "thenar_apb", 1: "middle_tip", 2: "thumb_tip", 3: "index_tip"}[ch]
            e = EXPORT[pod]
            u = e["unit"]
            across = np.cross(u, [1.0, 0.0, 0.0])
            across /= np.linalg.norm(across)
            b = e["rest"] + across * self.drift * self.t + u * self.press.get(pod, 0.0) + self.rng.normal(0, e["noise"], 3)
            pods[name] = {"pod_id": f"P-000{ch + 1}", "mag": {"ch": ch, "b_ut": b.tolist(), "valid": True,
                                                              "force_n": 0, "force_valid": True, "flags": ["FORCE_CAL_READY"]}}
        return {"schema_version": 1, "frame": int(self.t * 10), "t_device_us": int(self.t * 1e6), "pods": pods,
                "imu": {"acc_mps2": [0.0, 0.0, 9.81 + self.rng.normal(0, 0.02)], "gyr_rads": [0.0, 0.0, 0.0], "valid": True}}

    def poll(self):                                             # 10 Hz by wall clock, like the glove
        import time
        now = time.monotonic()
        self.next = getattr(self, "next", now)
        out = []
        while now >= self.next:
            out.append(self.frame())
            self.next += self.dt
        return out

    def close(self):
        pass


def feed(sensor, glove, seconds, gripping=False):
    return [sensor.update(glove.frame(), gripping) for _ in range(int(seconds * 10))]


def test_export_maps_channels_to_pods():
    assert EXPORT["thenar_apb"]["ch"] == 0 and EXPORT["middle_tip"]["ch"] == 1
    assert EXPORT["thumb_tip"]["ch"] == 2 and EXPORT["index_tip"]["ch"] == 3
    assert EXPORT["thumb_tip"]["noise"] == pytest.approx(5.593, abs=1e-3)


def test_glove_at_rest_with_drift_never_activates():
    glove = CalGlove(drift_ut_s=40.0)                           # 40 uT/s across the load axis: 400 uT in 10 s
    s = GripSensor(EXPORT, ut_per_n=100.0)
    s.set_rest([glove.frame() for _ in range(10)])
    out = feed(s, glove, 10.0)
    assert all(v == 0.0 for f in out for v in f.values()), "drift across the load axis must not count as grip"


def test_slow_drift_along_the_axis_is_tracked_while_open():
    glove = CalGlove()
    s = GripSensor(EXPORT, ut_per_n=100.0)
    s.set_rest([glove.frame() for _ in range(10)])
    for i in range(600):                                        # 5 uT/s along the load axis: 300 uT in 60 s
        glove.press = {k: 0.5 * i for k in PODS}
        s.update(glove.frame())
    assert not any(s.active.values())


def test_one_pad_is_not_a_grasp_and_a_stuck_pad_is_rezeroed():
    glove = CalGlove()
    s = GripSensor(EXPORT, ut_per_n=100.0)
    s.set_rest([glove.frame() for _ in range(10)])
    glove.press = {"thenar_apb": 400.0}                         # palm resting on something: one pad only
    out = feed(s, glove, 3.0)
    assert out[-1]["thenar_apb"] > 0 and sum(v > 0 for v in out[-1].values()) == 1
    feed(s, glove, 4.0)                                         # still alone after 6 s: an offset, not a grip
    assert "thenar_apb" in s.rezeroed and not s.active["thenar_apb"]


def test_live_loop_at_rest_stays_open_and_two_pads_grip():
    from dermis.glass.pickup_live import run
    rest = run(CalGlove(drift_ut_s=30.0), gui=False, duration_s=4.0, ut_per_n=100.0, grip_mode="proxy", export=EXPORT)
    assert not any(e["event"] == "CLOSING" for e in rest["events"]), rest["events"]

    class Grasp(CalGlove):
        def frame(self):
            self.press = {"thumb_tip": 450.0, "thenar_apb": 600.0} if self.t > 1.5 else {}
            return super().frame()
    held = run(Grasp(), gui=False, duration_s=3.5, ut_per_n=100.0, grip_mode="proxy", export=EXPORT)
    assert [e["event"] for e in held["events"]][:2] == ["CLOSING", "GRIPPED"]


class WavingGlove(CalGlove):
    """Free hand movement: palm and thumb pods swing by up to ~170 uT along their load axis without touching
    anything (what the real glove did when moved around); 'hold' presses well beyond that."""

    def __init__(self):
        super().__init__(seed=3)
        self.mode = "open"

    def frame(self):
        if self.mode == "wave":
            w = abs(np.sin(2 * np.pi * 1.3 * self.t))
            self.press = {"thenar_apb": 170.0 * w, "thumb_tip": 120.0 * w, "index_tip": 60.0 * w}
        elif self.mode == "hold":
            self.press = {"thenar_apb": 700.0, "thumb_tip": 500.0, "index_tip": 250.0}
        else:
            self.press = {}
        return super().frame()


def test_free_movement_step_raises_activation_so_waving_is_not_a_grip(tmp_path):
    import time
    from dermis.glass.pickup_live import run
    src = WavingGlove()
    modes = iter(["open", "hold", "wave"])
    cal = calibrate(src, str(tmp_path / "c.json"), open_s=0.6, hold_s=0.6, free_s=1.5, wait_s=0.6,
                    prompt=lambda m: setattr(src, "mode", next(modes)), export=EXPORT)
    assert cal["activation_ut"]["thenar_apb"] >= 1.5 * 150 and cal["pods_registered"] == ["thumb_tip", "index_tip", "thenar_apb"]
    t0 = time.monotonic()

    class Waving(WavingGlove):
        def frame(self):
            self.mode = "open" if time.monotonic() - t0 < 1.2 else "wave"
            return super().frame()
    r = run(Waving(), gui=False, duration_s=5.0, ut_per_n=cal["ut_per_n"], grip_mode="proxy", export=EXPORT,
            activation_ut=cal["activation_ut"])
    assert not any(e["event"] == "CLOSING" for e in r["events"]), r["events"]


def test_waving_without_free_movement_calibration_would_have_gripped():
    """The failure being fixed: with noise-only activation (the previous 8x noise, 35 uT) waving starts a grasp."""
    import time
    from dermis.glass import glove_cal
    from dermis.glass.pickup_live import run
    t0 = time.monotonic()

    class Waving(WavingGlove):
        def frame(self):
            self.mode = "open" if time.monotonic() - t0 < 1.2 else "wave"
            return super().frame()
    orig = glove_cal.GripSensor.__init__.__defaults__
    try:
        d = list(orig)
        d[2], d[3], d[4] = 8.0, 4.0, 35.0                      # on_sigma, off_sigma, min_on_ut of the previous version
        glove_cal.GripSensor.__init__.__defaults__ = tuple(d)
        r = run(Waving(), gui=False, duration_s=5.0, ut_per_n=60.0, grip_mode="proxy", export=EXPORT,
                cfg=PickupConfig(on_frames=3, solo_ratio=0.25))
    finally:
        glove_cal.GripSensor.__init__.__defaults__ = orig
    assert any(e["event"] == "CLOSING" for e in r["events"])
