"""Draft-schema normalization (compat.py) and the 3-claw hand mapping/controller."""
import copy
import json

import numpy as np
import pytest

from dermis.server.compat import normalize_frame
from dermis.server.dev_sender import SyntheticGlove
from dermis.server.teleop.calibration import CalibrationCollector
from dermis.server.teleop.controller import ARMED, PAUSED
from dermis.server.teleop.hand import HandController, HandMapping
from dermis.server.validation import schema_errors
from dermis.telemetry.model import DEFAULT_POD_ORDER
from dermis.telemetry.pipeline import RawMessage, TelemetryPipeline

NS = 1_000_000_000
POD_ORDER = (*DEFAULT_POD_ORDER, "wrist")


def sujal_draft(decoded_frame):
    """The shape Sujal proposed (with the pasted JSON's unfinished arrays filled in)."""
    f = copy.deepcopy(decoded_frame)
    del f["pods"]["middle_tip"]["imu"], f["pods"]["thenar_apb"]["imu"]
    f["pods"]["wrist_imu"] = {"ch": 3, "acc_mps2": [0.1, -0.2, 9.8], "gyr_rads": [0.0, 0.0, 0.0],
                              "temp_c": 24.08, "t_us": 12345890, "sensortime": 10597059}
    return f


def test_draft_rejected_by_v1_schema_but_normalizes(decoded_frame):
    f = sujal_draft(decoded_frame)
    assert schema_errors(f)                                   # what --strict-schema would do
    g, notes = normalize_frame(f)
    assert schema_errors(g) == []
    assert g["pods"]["middle_tip"]["imu"] is None and "wrist_imu" not in g["pods"]
    assert g["pods"]["wrist"]["imu"]["valid"] is False         # no 'valid' reported: never invented
    assert normalize_frame(f, assume_valid=True)[0]["pods"]["wrist"]["imu"]["valid"] is True
    assert "imu" not in f["pods"]["middle_tip"]                # input untouched (raw line is recorded)
    assert normalize_frame(decoded_frame) == (decoded_frame, [])   # v1 frames pass through unchanged


def test_draft_record_missing_required_keys_is_dropped(decoded_frame):
    f = sujal_draft(decoded_frame)
    del f["pods"]["wrist_imu"]["t_us"]
    g, notes = normalize_frame(f)
    assert g["pods"]["wrist"]["imu"] is None and schema_errors(g) == []
    assert any("record dropped" in n for n in notes)


class Feed:
    def __init__(self, fmt="v1", stationary_s=1.0, assume_valid=False):
        self.gen = SyntheticGlove(stationary_s=stationary_s, noise_ut=2.0)
        self.pipe = TelemetryPipeline(pod_order=POD_ORDER)
        self.fmt, self.assume_valid = fmt, assume_valid
        self.i, self.now = 0, 10 * NS

    def snap(self, t=None, faults=()):
        t = self.i / 30 if t is None else t
        f = (self.gen.draft_frame_at if self.fmt == "draft" else self.gen.frame_at)(t, self.i, faults=frozenset(faults))
        if self.fmt == "draft":
            f, _ = normalize_frame(f, self.assume_valid)
        self.i += 1
        self.now += NS // 30
        return self.pipe.ingest(RawMessage(f, self.now), "SYNTHETIC")


def calibrated(feed, m):
    col = CalibrationCollector(m, duration_s=0.0, min_samples=10)
    for _ in range(20):
        col.add(feed.snap(t=0.0))
    return col.result()


@pytest.mark.parametrize("fmt,assume_valid,palm_from", [("v1", False, "index_tip"), ("draft", False, "index_tip"),
                                                         ("draft", True, "wrist")])
def test_hand_mapping_claws_and_palm_source(fmt, assume_valid, palm_from):
    feed = Feed(fmt, assume_valid=assume_valid)
    m = HandMapping()
    cal = calibrated(feed, m)
    outs = [m.evaluate(feed.snap(t=t), cal) for t in np.arange(1.0, 7.0, 0.1)]
    assert all(o.usable for o in outs)
    flex = np.array([o.flex for o in outs])
    assert np.all((flex >= 0) & (flex <= 1)) and np.ptp(flex, axis=0).min() > 0.8   # every claw moves 0..1
    assert np.abs(flex[:, 1] - flex[:, 2]).max() > 0.3                              # middle lags index
    assert {o.signals["palm_from"] for o in outs} == {palm_from}
    rp = np.array([o.attitude for o in outs])
    assert np.abs(rp).max() <= m.cfg.max_tilt_rad and np.ptp(rp, axis=0).max() > 0.1


def test_hand_holds_only_the_bad_claw_and_pauses_on_watchdog():
    feed = Feed()
    m = HandMapping()
    c = HandController(m, log=lambda *_: None)
    c.calib = calibrated(feed, m)
    link = {"state": "connected", "epoch": 1}
    c.tick(feed.i, feed.snap(t=0.0), link, now_ns=feed.now)
    assert c.arm(link, now_ns=feed.now)[0]
    for t in np.arange(1.0, 3.0, 1 / 30):
        cmd = c.tick(feed.i, feed.snap(t=t), link, now_ns=feed.now)
    idx_before = cmd.extra["flex"][1]
    for t in np.arange(3.0, 3.5, 1 / 30):                      # index pod invalid: index claw holds, others move
        cmd = c.tick(feed.i, feed.snap(t=t, faults={"null_index"}), link, now_ns=feed.now)
    assert cmd.state == ARMED and "index_tip" in c.status().signals["held"]
    assert cmd.extra["flex"][1] == pytest.approx(idx_before, abs=0.1)
    held = None
    for k in range(14):                                        # silence: after 250 ms the whole hand pauses and holds
        feed.now += NS // 30
        cmd = c.tick(c._seen_seq, None, link, now_ns=feed.now)
        if k >= 8:
            assert cmd.state == PAUSED
            held = cmd.extra if held is None else held
            assert cmd.extra == held
