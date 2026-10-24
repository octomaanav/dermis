"""EXPERIMENTAL IMU motion bursts: the SYNTHETIC punch stream must move the measured hand forward and
back to home, bounded, in real time (~12 s). Also checks the estimator offline at several frame rates."""
import subprocess
import sys
import time

import numpy as np
import pytest

from dermis.server.dev_sender import PUNCH_PERIOD_S, PUNCH_REACH_M, SyntheticGlove
from dermis.server.teleop.calibration import CalibrationCollector
from dermis.server.teleop.controller import ARMED
from dermis.server.teleop.hand import HandMapping
from dermis.server.teleop.motion import MotionEstimator
from dermis.server.teleop.session import TeleopSession
from dermis.telemetry.model import DEFAULT_POD_ORDER
from dermis.telemetry.pipeline import RawMessage, TelemetryPipeline


def _offline(profile, rate, seconds=11.0):
    g = SyntheticGlove(stationary_s=1.0, profile=profile, noise_ut=2.0)
    pipe = TelemetryPipeline(pod_order=(*DEFAULT_POD_ORDER, "wrist"))
    snaps = [pipe.ingest(RawMessage(g.frame_at(i / rate, i), int(i / rate * 1e9)), "S") for i in range(int(seconds * rate))]
    col = CalibrationCollector(HandMapping(), duration_s=0, min_samples=10)
    for s in snaps[: int(0.8 * rate)]:
        col.add(s)
    est = MotionEstimator()
    est.set_reference(col.result())
    xs = []
    for s in snaps[rate:]:
        est.update(s)
        xs.append(est.offset())
    return np.array(xs), est


@pytest.mark.parametrize("rate", [30, 100])
def test_estimator_reproduces_punch_and_returns_home(rate):
    xs, est = _offline("punch", rate)
    t = np.arange(len(xs)) / rate + 1.0
    for k in range(3):
        w = (t >= 1.0 + k * PUNCH_PERIOD_S) & (t < 1.0 + (k + 1) * PUNCH_PERIOD_S)
        assert xs[w, 0].max() > 0.6 * PUNCH_REACH_M                     # each punch reaches forward
        rest = w & (t > 1.0 + k * PUNCH_PERIOD_S + 2.0)                 # ~0.85 s after the return
        assert np.abs(xs[rest]).max() < 0.02                            # back home, no drift
    assert np.all(xs >= est.limits[:, 0] - 1e-9) and np.all(xs <= est.limits[:, 1] + 1e-9)
    assert np.abs(xs[:, 1:]).max() < 0.02                               # a forward punch stays forward


@pytest.mark.parametrize("profile", ["pinch", "updown"])
def test_no_drift_from_slow_tilt(profile):
    xs, _ = _offline(profile, 30)
    assert np.abs(xs).max() < 0.01


@pytest.mark.slow
def test_punch_stream_moves_simulated_hand(tmp_path):
    sess = TeleopSession("127.0.0.1", 0, gui=False, auto=True, calib_dir=str(tmp_path), log=lambda *_: None)
    _, port = sess.start()
    proc = subprocess.Popen([sys.executable, "-m", "dermis.server.dev_sender", "--port", str(port), "--profile",
                             "punch", "--rate", "100", "--duration", "10", "--stationary-s", "2.5"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    rows, t0, nxt = [], time.monotonic(), time.monotonic()
    try:
        while time.monotonic() - t0 < 14:
            r = sess.step()
            rows.append((time.monotonic() - t0, r["state"], r["offset"][0], r["joints"][3]))
            if proc.poll() is not None and r["link"]["state"] != "connected":
                break
            nxt += 1 / 30
            time.sleep(max(0.0, nxt - time.monotonic()))
    finally:
        proc.wait(5)
        sess.close()
    armed = [(t, x, q) for t, s, x, q in rows if s == ARMED]
    assert armed, "never armed"
    xs = np.array([x for _, x, _ in armed])
    assert xs.max() > 0.15, f"measured wrist moved only {xs.max():.3f} m forward"
    assert xs.min() > -0.06 and xs.max() <= 0.30 + 0.005
    assert (np.abs(xs) < 0.02).sum() > 30, "hand never settled back home between punches"
    assert max(q for _, _, q in armed) > 1.0                            # index knuckle curls into a fist
