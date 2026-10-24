"""TCP teleop mapping, calibration and controller (no sockets, no PyBullet)."""
import time

import numpy as np
import pytest

from dermis.server.dev_sender import SyntheticGlove
from dermis.telemetry.pipeline import RawMessage, TelemetryPipeline
from dermis.server.teleop.calibration import CalibrationCollector, CalibrationError
from dermis.server.teleop.controller import ARMED, DISARMED, PAUSED, ControllerConfig, TeleopController
from dermis.server.teleop.mapping import MagGripTiltMapping, MappingConfig

LINK = {"state": "connected", "epoch": 1}
NS = 1_000_000_000


class Feed:
    """Synthetic frames -> pipeline snapshots with a controllable host clock."""

    def __init__(self, stationary_s=100.0, **kw):
        self.gen = SyntheticGlove(stationary_s=stationary_s, noise_ut=2.0, **kw)
        self.pipe = TelemetryPipeline()
        self.i = 0
        self.now = 10 * NS

    def snap(self, faults=frozenset(), t=None, mutate=None, advance=True):
        t = self.i / 30 if t is None else t
        f = self.gen.frame_at(t, self.i, faults=frozenset(faults))
        if mutate:
            mutate(f)
        self.i += 1
        if advance:
            self.now += NS // 30
        return self.pipe.ingest(RawMessage(f, self.now), "SYNTHETIC")


def calibrated(feed, mapping):
    col = CalibrationCollector(mapping, duration_s=0.0, min_samples=10)
    for _ in range(20):
        col.add(feed.snap(t=0.0))                      # still, unloaded pose
    return col.result()


def test_mapping_bounded_for_extreme_inputs():
    feed = Feed()
    m = MagGripTiltMapping(MappingConfig(tilt_xy=True))
    cal = calibrated(feed, m)
    for dz, expect in ((1e6, 0.0), (-1e6, 0.08), (0.0, 0.08)):
        def big(f):
            for pod in ("index_tip", "thumb_tip"):
                f["pods"][pod]["mag"]["b_ut"][2] += dz
        out = m.evaluate(feed.snap(mutate=big), cal)
        assert out.usable and out.opening == pytest.approx(expect, abs=1e-3)
        assert 0.0 <= out.opening <= 0.08

    def flip(f):                                       # pod upside down: tilt saturates, |v| bounded
        f["pods"]["index_tip"]["imu"]["acc_mps2"] = [0.0, 9.0, -3.0]
    out = m.evaluate(feed.snap(mutate=flip), cal)
    assert max(abs(v) for v in out.v_xy) <= m.cfg.tilt_gain_mps_per_rad * m.cfg.tilt_max_rad + 1e-9

    def shake(f):                                      # |a| = 3 g: not quasi-static -> zero velocity
        f["pods"]["index_tip"]["imu"]["acc_mps2"] = [0.0, 0.0, 29.4]
    out = m.evaluate(feed.snap(mutate=shake), cal)
    assert out.usable and out.v_xy == (0.0, 0.0)


def test_required_channel_validity_and_stale_handling():
    feed = Feed()
    m = MagGripTiltMapping()
    cal = calibrated(feed, m)
    assert m.evaluate(feed.snap(), cal).usable                     # thenar mag is invalid but not required
    assert m.evaluate(feed.snap({"null_index"}), cal).usable          # one null frame: inside the 50 ms hold
    for _ in range(2):
        out = m.evaluate(feed.snap({"null_index"}), cal)
    assert not out.usable and "invalid/null" in out.reason          # hold expired: unusable, never invented
    # STALE within the 50 ms device-time hold is still usable; beyond it is not
    feed.snap()
    assert m.evaluate(feed.snap({"stale_thumb"}), cal).usable
    for _ in range(3):
        out = m.evaluate(feed.snap({"stale_thumb"}), cal)
    assert not out.usable and "STALE too long" in out.reason

    def temp_stale_only(f):
        f["pods"]["index_tip"]["mag"]["flags"] = ["TEMP_STALE"]
        f["pods"]["thumb_tip"]["mag"]["flags"] = ["TEMP_STALE"]
    assert m.evaluate(feed.snap(mutate=temp_stale_only), cal).usable   # temp age does not stale the field

    def drop_thumb(f):
        del f["pods"]["thumb_tip"]
    assert "thumb_tip missing" in m.evaluate(feed.snap(mutate=drop_thumb), cal).reason

    def nan_field(f):
        f["pods"]["index_tip"]["mag"]["b_ut"][0] = float("nan")
    for _ in range(3):
        out = m.evaluate(feed.snap(mutate=nan_field), cal)
    assert not out.usable


def test_calibration_rejects_motion_and_bad_channels():
    feed = Feed(stationary_s=0.0)                       # grip moving from the start
    m = MagGripTiltMapping()
    col = CalibrationCollector(m, duration_s=0.0, min_samples=10)
    for _ in range(60):
        col.add(feed.snap())
    with pytest.raises(CalibrationError, match="not steady"):
        col.result()
    col = CalibrationCollector(m, duration_s=0.0, min_samples=10)
    for _ in range(20):
        col.add(Feed().snap({"null_index"}))
    with pytest.raises(CalibrationError, match="usable frames"):
        col.result()


def make_ctrl(feed, **kw):
    m = MagGripTiltMapping(MappingConfig(tilt_xy=True))
    c = TeleopController(m, ControllerConfig(**kw), log=lambda *_: None)
    c.calib = calibrated(feed, m)
    return c


def tick(c, feed, link=LINK, faults=(), send=True, mutate=None):
    snap = feed.snap(faults, mutate=mutate) if send else None
    if not send:
        feed.now += NS // 30
    seq = feed.i if send else c._seen_seq
    return c.tick(seq, snap, link, now_ns=feed.now)


def test_disarmed_by_default_and_arm_requires_calibration():
    feed = Feed()
    m = MagGripTiltMapping()
    c = TeleopController(m, log=lambda *_: None)
    cmd = c.tick(1, feed.snap(), LINK, now_ns=feed.now)
    assert cmd.state == DISARMED and not cmd.v.any()
    ok, msg = c.arm(LINK)
    assert not ok and "not calibrated" in msg


def test_watchdog_stop_and_hold_then_auto_resume():
    feed = Feed(stationary_s=0.5)
    c = make_ctrl(feed, resume_frames=3)
    tick(c, feed)
    assert c.arm(LINK, now_ns=feed.now)[0]
    for _ in range(60):                                # moving: opening and velocity respond
        cmd = tick(c, feed)
    assert cmd.state == ARMED
    held_opening = None
    for k in range(12):                                # data stops: 400 ms of silence
        cmd = tick(c, feed, send=False)
        if k >= 8:                                     # > 250 ms: paused, motion stopped, opening held
            assert cmd.state == PAUSED and "watchdog" in cmd.reason
            held_opening = cmd.opening if held_opening is None else held_opening
            assert not cmd.v.any() and cmd.opening == held_opening
    for _ in range(5):
        cmd = tick(c, feed)
    assert cmd.state == ARMED


def test_connection_loss_requires_rearm_and_config_change_invalidates():
    feed = Feed()
    c = make_ctrl(feed)
    tick(c, feed)
    assert c.arm(LINK, now_ns=feed.now)[0]
    cmd = tick(c, feed, link={"state": "disconnected", "epoch": 1})
    assert cmd.state == DISARMED and "re-arm" in cmd.reason
    cmd = tick(c, feed, link={"state": "connected", "epoch": 2})
    assert cmd.state == DISARMED                       # reconnecting does not re-arm by itself
    assert c.arm({"state": "connected", "epoch": 2}, now_ns=feed.now)[0]
    assert not c.start_calibration()[0]                # never recalibrate while armed

    def new_cfg(f):
        f["config_id"] = 4
    cmd = tick(c, feed, link={"state": "connected", "epoch": 2}, mutate=new_cfg)
    assert cmd.state == DISARMED and "calibration invalidated" in cmd.reason and c.calib is None


def test_velocity_and_opening_rate_limited():
    feed = Feed(stationary_s=0.0)
    c = make_ctrl(feed, max_speed=0.02, max_open_rate=0.05)
    tick(c, feed)
    assert c.arm(LINK, now_ns=feed.now)[0]
    prev = c.opening
    for _ in range(200):
        cmd = tick(c, feed)
        assert np.all(np.abs(cmd.v) <= 0.02 + 1e-12)
        assert abs(cmd.opening - prev) <= 0.05 / 30 + 1e-12
        prev = cmd.opening


def test_tilt_z_maps_pitch_to_bounded_z_velocity():
    feed = Feed(profile="updown", stationary_s=1.0)
    m = MagGripTiltMapping(MappingConfig(tilt_xy=True, pitch_axis="z"))
    cal = calibrated(feed, m)
    vzs = [m.evaluate(feed.snap(t=t), cal) for t in np.arange(1.0, 7.0, 0.1)]
    assert all(o.usable and o.v_xy[0] == 0.0 for o in vzs)        # pitch no longer drives X
    vz = np.array([o.vz for o in vzs])
    assert vz.max() > 0.01 and vz.min() < -0.01                    # down and up over one cycle
    assert np.abs(vz).max() <= m.cfg.tilt_gain_mps_per_rad * m.cfg.tilt_max_rad + 1e-9
    assert all(o.opening == pytest.approx(0.08, abs=2e-3) for o in vzs)   # updown keeps the gripper open


def test_write_updown_dataset_is_schema_valid(tmp_path):
    import json
    from types import SimpleNamespace
    from dermis.server.dev_sender import write_synthetic
    from dermis.server.validation import schema_errors
    out = tmp_path / "updown.jsonl"
    a = SimpleNamespace(write=str(out), profile="updown", duration=2.0, rate=30.0, stationary_s=0.5,
                        grip_period_s=4.0, no_tilt=False, seed=1)
    assert write_synthetic(a, lambda *_: None) == 0
    frames = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(frames) == 60 and [f["frame"] for f in frames] == list(range(60))
    assert all(schema_errors(f) == [] and f["t_host_ns"] is None for f in frames)
