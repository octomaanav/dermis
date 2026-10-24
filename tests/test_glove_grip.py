"""Glove bring-up logic without hardware. SYNTHETIC frames validate software integration only."""
import copy

import numpy as np
import pytest

from dermis.glove.discovery import merge, source_label
from dermis.glove.grip import (ARMED, DISARMED, PAUSED, CalibrationRejected, GripControl, GripMappingConfig,
                               ReferenceCapture, build_calibration)
from dermis.telemetry.pipeline import RawMessage, TelemetryPipeline
from dermis.telemetry.sources import SyntheticSource

PODS = ("index_tip", "thumb_tip", "middle_tip", "thenar_apb")
BASE = {"index_tip": np.array([120.0, -280.0, 27000.0]), "thumb_tip": np.array([-500.0, 150.0, 27500.0]),
        "middle_tip": np.array([40.0, -80.0, 26500.0]), "thenar_apb": np.array([10.0, 60.0, 26000.0])}
PRESS = np.array([60.0, -40.0, 400.0])


class Stream:
    """Frames with controlled field values, host receive time and per-pod faults."""

    def __init__(self, pods=PODS, device="dermis-test01", config=3):
        self.gen = SyntheticSource(realtime=False, pods=pods, device_id=device)
        self.k, self.t_ns, self.config = 0, 10**12, config
        self.pipe = TelemetryPipeline(pod_order=pods)
        self.rng = np.random.default_rng(0)

    def frame(self, press=0.0, noise=2.0, drop=(), null=(), stale=(), spike=None, dt_s=0.01, device=None, config=None):
        f = self.gen.frame(self.k, 1000 + self.k, config or self.config, [])
        if device:
            f["device_id"] = device
        for loc in list(f["pods"]):
            m = f["pods"][loc]["mag"]
            b = BASE.get(loc, np.zeros(3)) + (press * PRESS if loc.endswith("_tip") else 0) + self.rng.normal(0, noise, 3)
            if spike and loc == spike[0]:
                b = b + spike[1]
            m["b_ut"] = b.round(3).tolist()
            if loc in null:
                m["b_ut"] = None                    # valid=true but null: unusable
            if loc in stale:
                m["flags"] = ["STALE"]
        for loc in drop:
            f["pods"].pop(loc, None)
        self.k += 1
        self.t_ns += int(dt_s * 1e9)
        return self.pipe.ingest(RawMessage(f, self.t_ns), "SYNTHETIC")


def calibrate(s: Stream, cfg: GripMappingConfig, press_level=1.0, unloaded_noise=2.0, press_noise=2.0):
    u, p = ReferenceCapture("unloaded", cfg.required_pods + cfg.optional_pods), ReferenceCapture("press", cfg.required_pods + cfg.optional_pods)
    for _ in range(60):
        u.add(s.frame(0.0, unloaded_noise))
    for _ in range(60):
        p.add(s.frame(press_level, press_noise))
    return build_calibration(u, p, cfg)


def test_discovery_handles_variable_layouts():
    seen = {"index_tip": {"pod_id": "A", "mag": True, "imu": True}, "wrist": {"pod_id": "W", "mag": False, "imu": True}}
    info = {"pods": [{"location": "index_tip", "id": "A", "mag_ch": 0, "imu_ch": 0},
                     {"location": "ring_tip", "id": "R", "mag_ch": 4, "imu_ch": None}]}
    lay = merge(info, seen)
    assert lay.pods == ["index_tip", "ring_tip", "wrist"]
    assert lay.channels["wrist"] == {"mag": False, "imu": True, "pod_id": "W", "from": ["frames"]}
    assert lay.channels["ring_tip"]["imu"] is False and any("ring_tip" in n for n in lay.notes)
    s = Stream(pods=("index_tip", "wrist_x"))
    s.frame()
    assert set(s.pipe.seen_pods) == {"index_tip", "wrist_x"}


def test_source_labels():
    assert source_label("GLOVE", "synthetic-dermis-dev", "tcp") == "SYNTHETIC"
    assert source_label("GLOVE", "dermis-3c84279f2a10", "tcp") == "TCP-UNVERIFIED"     # e.g. fixture replay over TCP
    assert source_label("GLOVE", "dermis-3c84279f2a10", "tcp", operator_confirmed=True) == "REAL HARDWARE"
    assert source_label("GLOVE", "dermis-3c84279f2a10", "serial") == "REAL HARDWARE"
    assert source_label("REPLAY", "dermis-3c84279f2a10", "replay") == "REPLAY"


def test_two_point_calibration_accepts_clear_separation():
    cfg = GripMappingConfig(optional_pods=("middle_tip",))
    cal = calibrate(Stream(), cfg)
    assert set(cal.pods) == {"index_tip", "thumb_tip", "middle_tip"}
    assert all(c["separation"] > 20 for c in cal.pods.values())
    assert "not force" in cal.label


def test_calibration_rejected_for_insufficient_separation():
    cfg = GripMappingConfig()
    with pytest.raises(CalibrationRejected) as e:
        calibrate(Stream(), cfg, press_level=0.02)             # ~8 uT change vs ~2 uT noise
    assert any("insufficient separation" in r for r in e.value.reasons)
    assert e.value.report["index_tip"]["accepted"] is False


def test_calibration_rejected_when_not_stationary_or_missing_required():
    cfg = GripMappingConfig()
    with pytest.raises(CalibrationRejected) as e:
        calibrate(Stream(), cfg, unloaded_noise=150.0)
    assert any("not stationary" in r for r in e.value.reasons)
    s = Stream(pods=("index_tip", "middle_tip"))                # thumb_tip required but absent
    with pytest.raises(CalibrationRejected) as e:
        calibrate(s, cfg)
    assert any("thumb_tip" in r and "too few" in r for r in e.value.reasons)


def test_calibration_rejected_across_device_or_config_change():
    cfg = GripMappingConfig()
    s = Stream()
    u, p = ReferenceCapture("unloaded", cfg.required_pods), ReferenceCapture("press", cfg.required_pods)
    for _ in range(40):
        u.add(s.frame(0.0))
    for _ in range(40):
        p.add(s.frame(1.0, config=4))
    with pytest.raises(CalibrationRejected) as e:
        build_calibration(u, p, cfg)
    assert any("different device/config" in r for r in e.value.reasons)


def armed(cfg=None, **kw):
    cfg = cfg or GripMappingConfig(**kw)
    s = Stream()
    cal = calibrate(s, cfg)
    g = GripControl(cfg, cal)
    snap = s.frame(0.0)
    ok, msg = g.arm(snap, False)
    assert ok, msg
    return s, g


def tick(s, g, **kw):
    snap = s.frame(**kw)
    return g.tick(snap, now_ns=s.t_ns + 1_000_000)


def test_arming_required_and_output_bounded():
    cfg = GripMappingConfig()
    s = Stream()
    g = GripControl(cfg, calibrate(s, cfg))
    for _ in range(30):
        c = tick(s, g, press=1.0)
    assert c.state == DISARMED and c.value == cfg.out_min        # never moves until armed
    ok, _ = g.arm(s.frame(0.0), False)
    assert ok
    vals = [tick(s, g, press=1.0).value for _ in range(120)]
    assert max(vals) <= cfg.out_max + 1e-9 and min(vals) >= cfg.out_min - 1e-9
    assert vals[-1] > 0.9 * cfg.out_max                          # full press reaches the top
    steps = np.diff([cfg.out_min] + vals)
    assert steps.max() <= cfg.max_rate / 30 + 1e-9                # slew limited
    over = [tick(s, g, press=5.0).value for _ in range(30)]       # beyond the press reference: clamped
    assert max(over) <= cfg.out_max + 1e-9


def test_missing_optional_channel_does_not_pause():
    s, g = armed(optional_pods=("middle_tip",))
    for _ in range(10):
        c = tick(s, g, press=0.5, drop=("middle_tip",))
    assert c.state == ARMED and "middle_tip" not in c.pods_used and set(c.pods_used) == {"index_tip", "thumb_tip"}


def test_null_or_stale_required_channel_pauses_and_holds():
    s, g = armed()
    for _ in range(40):
        tick(s, g, press=0.6)
    held = g.value
    c = tick(s, g, press=0.6, null=("index_tip",))
    assert c.state == ARMED                                       # within the 50 ms bounded hold: still usable
    for _ in range(10):
        c = tick(s, g, press=0.6, null=("index_tip",))
    assert c.state == PAUSED and "index_tip" in c.reason and abs(c.value - held) < 0.5
    v = c.value
    for _ in range(10):
        c = tick(s, g, press=1.0, stale=("thumb_tip",), null=("index_tip",))
    assert c.state == PAUSED and c.value == v                    # held, not moving
    for _ in range(5):
        c = tick(s, g, press=1.0)
    assert c.state == ARMED                                       # brief pause resumes without re-arming


def test_spike_rejected_and_bounded():
    s, g = armed()
    for _ in range(40):
        tick(s, g, press=0.3)
    before = g.value
    c = tick(s, g, press=0.3, spike=("index_tip", np.array([0, 0, 20000.0])))
    assert g.spikes >= 1 and abs(c.value - before) <= GripMappingConfig().max_rate / 30 + 1e-9
    c = tick(s, g, press=0.3)
    assert c.state == ARMED


def test_disconnect_pauses_then_disarms_and_reconnect_needs_rearm():
    s, g = armed()
    snap = s.frame(press=0.4)
    g.tick(snap, now_ns=snap.frame.t_recv_ns + 10_000_000)
    c = g.tick(snap, now_ns=snap.frame.t_recv_ns + int(0.5e9))     # 0.5 s without frames
    assert c.state == PAUSED and "watchdog" in c.reason
    c = g.tick(snap, now_ns=snap.frame.t_recv_ns + int(3e9))       # 3 s: connection lost
    assert c.state == DISARMED and "re-arm" in c.reason
    c = tick(s, g, press=1.0)                                     # frames are back
    assert c.state == DISARMED                                    # but control does not resume on its own
    ok, _ = g.arm(s.frame(0.0), False)
    assert ok


def test_config_change_disarms_and_recalibration_needs_disarm():
    s, g = armed()
    c = tick(s, g, press=0.2, config=7)
    assert c.state == DISARMED and "config" in c.reason
    ok, msg = g.arm(s.frame(0.0, config=7), False)
    assert not ok and "does not match" in msg
    s2, g2 = armed()
    ok, msg = g2.set_calibration(g2.cal)
    assert not ok and "disarm" in msg


def test_arm_refused_without_input_or_calibration():
    cfg = GripMappingConfig()
    s = Stream()
    assert GripControl(cfg, None).arm(s.frame(), False)[0] is False
    cal = calibrate(s, cfg)
    assert GripControl(cfg, cal).arm(None, True)[0] is False
    for _ in range(10):                                           # thumb null for 100 ms > 50 ms hold
        snap = s.frame(null=("thumb_tip",))
    ok, msg = GripControl(cfg, cal).arm(snap, False)
    assert not ok and "thumb_tip" in msg
