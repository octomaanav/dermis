import numpy as np
import pytest

from dermis.telemetry.calibration import BaselineCapture, Calibration, mount_matrix, mounts_from_device_info
from dermis.telemetry.pipeline import RawMessage, TelemetryPipeline
from dermis.telemetry.sources import SyntheticSource
from dermis.teleop.mapping import (ACTIVE, PAUSED_CONFIG, PAUSED_NO_DATA, PAUSED_STALE,
                                   ExperimentalTiltGripMapping)


def snaps(n=200, faults=(), device="synthetic-000000"):
    pipe = TelemetryPipeline()
    src = SyntheticSource(n_frames=n, faults=faults, realtime=False, device_id=device)
    return [s for m in src for s in [pipe.ingest(m, "SYNTHETIC")] if s is not None]


def test_mount_matrix():
    np.testing.assert_array_equal(mount_matrix("+X+Y+Z"), np.eye(3))
    R = mount_matrix("-Y+X+Z")
    np.testing.assert_array_equal(R @ [0, 1, 0], [-1, 0, 0])      # chip +Y points along -distal
    for bad in ("+X+X+Z", "-X+Y+Z", "+X+Y"):
        with pytest.raises(ValueError):
            mount_matrix(bad)


def test_mounts_from_device_info(fixture_path):
    import json
    info = json.loads(fixture_path("device_info.json").read_text())
    m = mounts_from_device_info(info)
    assert set(m) == {"index_tip", "thumb_tip", "middle_tip", "thenar_apb"}


def test_baseline_capture_save_load_apply(tmp_path):
    ss = snaps(60)
    cap = BaselineCapture(min_samples=20)
    for s in ss[:25]:                         # synthetic presses start ~0 at t=0
        cap.add(s)
    cal, report = cap.finish(scale_ut=500.0)
    assert set(cal.pods) == set(report) and all(r["accepted"] for r in report.values())
    cal.save(tmp_path / "c.json")
    cal2 = Calibration.load(tmp_path / "c.json")
    assert cal2.calibration_id == cal.calibration_id and cal2.version == 1
    out = cal2.apply(ss[0])
    assert out["valid_for_frame"] and out["mask"].all()
    assert (out["delta_norm_ut"] < 50).all()                         # near baseline when unloaded
    assert "not force" in cal2.note.lower() or "not force" in cal2.note


def test_calibration_invalid_for_other_config():
    ss = snaps(120, faults=("config_change",))
    cap = BaselineCapture(min_samples=5)
    cap.add(ss[0])
    cal, _ = cap.finish()
    changed = [s for s in ss if s.frame.config_id != cal.config_id]
    assert changed
    out = cal.apply(changed[0])
    assert not out["valid_for_frame"] and not out["mask"].any()


def test_capture_skips_stale_and_invalid():
    ss = snaps(200, faults=("stale", "invalid"))
    cap = BaselineCapture(min_samples=1)
    for s in ss:
        cap.add(s)
    n_fresh_idx = sum(int(s.mag_fresh[0]) for s in ss)
    n_fresh_thumb = sum(int(s.mag_fresh[1]) for s in ss)
    _, report = cap.finish()
    assert report["index_tip"]["n_samples"] == n_fresh_idx < len(ss)
    assert report["thumb_tip"]["n_samples"] == n_fresh_thumb < len(ss)


def _cal(ss):
    cap = BaselineCapture(min_samples=5)
    for s in ss[:10]:
        cap.add(s)
    return cap.finish()[0]


def test_experimental_mapping_requires_deliberate_enable():
    cal = _cal(snaps(20))
    with pytest.raises(PermissionError):
        ExperimentalTiltGripMapping(cal)
    with pytest.raises(ValueError):
        ExperimentalTiltGripMapping(None, enabled=True)
    assert ExperimentalTiltGripMapping.EXPERIMENTAL


def test_mapping_pauses_and_neutralizes():
    ss = snaps(400)
    m = ExperimentalTiltGripMapping(_cal(ss), enabled=True)
    m.capture_rest(ss[0])
    a, st = m(None, True)
    assert st == PAUSED_NO_DATA and (a[:3] == 0).all()
    a, st = m(ss[5], False)
    assert st == ACTIVE and np.isfinite(a).all()
    closed = [m(s, False)[0][3] for s in ss]
    assert min(closed) == 0.0 and max(closed) > 0.07              # synthetic presses close and release
    last_open = m.last_opening
    a, st = m(ss[-1], True)                                       # wall-clock silence
    assert st == PAUSED_NO_DATA and (a[:3] == 0).all() and a[3] == last_open


def test_mapping_pauses_on_stale_beyond_hold():
    pipe = TelemetryPipeline(max_hold_us=20_000)
    frames = list(SyntheticSource(n_frames=40, realtime=False).frames())
    for k in range(10, 40):                   # index mag STALE from frame 10 on
        frames[k]["pods"]["index_tip"]["mag"]["flags"] = ["STALE"]
    ss = [pipe.ingest(RawMessage(f, 0), "SYNTHETIC") for f in frames]
    m = ExperimentalTiltGripMapping(_cal(ss), enabled=True)
    statuses = [m(s, False)[1] for s in ss]
    assert statuses[5] == ACTIVE and statuses[11] == ACTIVE           # brief STALE: bounded hold
    assert statuses[-1] == PAUSED_STALE                                # held beyond 20 ms


def test_mapping_pauses_on_config_mismatch():
    ss = snaps(120, faults=("config_change",))
    m = ExperimentalTiltGripMapping(_cal(ss), enabled=True)
    other = [s for s in ss if s.frame.config_id != m.calibration.config_id and s.mag_held_mask[:2].all()
             and s.imu_held_mask[3]]
    assert m(other[0], False)[1] == PAUSED_CONFIG


def test_keyboard_teleop_mapping():
    from dermis.sim.env import MAX_EFFORT, MAX_OPENING
    from dermis.teleop.demo import KeyboardTeleop, keys_to_velocity
    np.testing.assert_allclose(keys_to_velocity({"w", "left", "q"}), [0.1, 0.1, -0.1])
    np.testing.assert_allclose(keys_to_velocity({"w", "s"}), [0, 0, 0])            # opposing keys cancel
    kb = KeyboardTeleop(effort=38.0)
    a = kb.update(set(), set())
    assert a[3] == MAX_OPENING and a[4] == 38.0
    a = kb.update({"e"}, {" ", "]", "]"})
    assert a[2] > 0 and a[3] == 0.0 and a[4] == MAX_EFFORT                         # toggled closed, effort capped
    kb.update(set(), {"n"})
    assert kb.command == "next"
