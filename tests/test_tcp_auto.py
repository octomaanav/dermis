"""--auto operator + --tilt-z with the SYNTHETIC 'updown' sender, end to end (~16 s real time).

No C / R presses: the session must calibrate and arm on its own, move the measured end effector
down and up, and re-arm by itself after the sender reconnects.
"""
import subprocess
import sys
import time

import numpy as np
import pytest

from dermis.server.teleop.controller import ARMED, DISARMED
from dermis.server.teleop.mapping import MappingConfig
from dermis.server.teleop.session import TeleopSession


def _sender(port, seconds):
    return subprocess.Popen([sys.executable, "-m", "dermis.server.dev_sender", "--port", str(port), "--mode",
                             "synthetic", "--profile", "updown", "--duration", str(seconds), "--stationary-s", "2.5"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def _run(sess, proc, until_s):
    rows, t0, next_t = [], time.monotonic(), time.monotonic()
    while time.monotonic() - t0 < until_s:
        r = sess.step()
        rows.append((r["state"], r["ee"][2]))
        if proc.poll() is not None and r["link"]["state"] != "connected":
            break
        next_t += 1 / 30
        time.sleep(max(0.0, next_t - time.monotonic()))
    return rows


@pytest.mark.slow
def test_auto_arms_and_updown_moves_arm(tmp_path):
    sess = TeleopSession("127.0.0.1", 0, gui=False, auto=True, calib_dir=str(tmp_path), robot="arm",
                         mapping_cfg=MappingConfig(tilt_xy=True, pitch_axis="z"), log=lambda *_: None)
    _, port = sess.start()
    try:
        p1 = _sender(port, 9.0)
        rows1 = _run(sess, p1, 14)
        p1.wait(5)
        p2 = _sender(port, 6.0)                       # reconnect: auto must re-arm without any key press
        rows2 = _run(sess, p2, 10)
        p2.wait(5)
    finally:
        sess.close()
    z1 = np.array([z for s, z in rows1 if s == ARMED])
    assert len(z1) > 60, "never armed automatically"
    assert np.ptp(z1) > 0.03, f"arm did not move down/up (z range {np.ptp(z1):.3f} m)"
    assert z1.min() >= 0.10 - 0.01 and z1.max() <= 0.35 + 0.01
    assert any(s == ARMED for s, _ in rows2), "did not re-arm after reconnect"
    assert rows2[-1][0] == DISARMED                    # sender gone -> disarmed, position held
