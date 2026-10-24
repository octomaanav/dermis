"""TelemetryServer over real local sockets (port 0 = any free port)."""
import json
import socket
import time

import pytest

from dermis.server.dev_sender import SyntheticGlove
from dermis.server.tcp_server import TelemetryServer


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def server():
    s = TelemetryServer("127.0.0.1", 0, max_message_bytes=8192, log=lambda *_: None)
    s.start()
    yield s
    s.stop()


def connect(server):
    c = socket.create_connection(("127.0.0.1", server.port), timeout=2)
    c.settimeout(2)
    hello = c.recv(4096)
    return c, hello


def line(obj):
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode()


@pytest.fixture(scope="module")
def gen():
    return SyntheticGlove(stationary_s=0.0, noise_ut=0.0)


def test_fixture_frame_accepted_and_hello(server, decoded_frame):
    c, hello = connect(server)
    assert b'"controller"' in hello
    c.sendall(line(decoded_frame))
    assert wait_for(lambda: server.status()["accepted"] == 1)
    seq, snap = server.slot.get()
    assert snap.frame.frame == decoded_frame["frame"] and snap.frame.t_host_ns is None
    assert snap.frame.source == "TCP-UNVERIFIED"
    # thenar mag is null/BAD_LENGTH; middle IMU is STALE; middle mag TEMP_STALE does not make the field stale
    i = snap.pods.index("thenar_apb")
    assert not snap.mag_valid[i]
    m = snap.pods.index("middle_tip")
    assert snap.imu_stale[m] and snap.mag_fresh[m] and not snap.mag_temp_fresh[m]
    c.close()


def test_fragmented_combined_malformed_oversized(server, gen):
    c, _ = connect(server)
    f0, f1, f2, f3 = (line(gen.frame_at(i / 30, i)) for i in range(4))
    for k in range(0, len(f0), 7):                    # one frame in many writes
        c.sendall(f0[k:k + 7])
        time.sleep(0.001)
    c.sendall(b'{"broken": \n' + b"\xff\xfe\n" + f1 + f2)            # junk + two frames in one write
    c.sendall(b'{"pad":"' + b"x" * 20000 + b'"}\n')                   # oversized
    c.sendall(line({"not": "a frame"}))                                # schema reject
    c.sendall(f3)
    assert wait_for(lambda: server.status()["accepted"] == 4)
    st = server.status()
    assert st["malformed"] == 2 and st["oversized"] == 1 and st["schema_rejected"] == 1
    assert st["state"] == "connected" and st["peak_buffered_bytes"] <= 8192
    c.close()


def test_duplicates_and_out_of_order_dropped(server, gen):
    c, _ = connect(server)
    fr = [line(gen.frame_at(i / 30, i)) for i in range(5)]
    c.sendall(fr[0] + fr[1] + fr[1] + fr[3] + fr[2] + fr[4])
    done = lambda st: st["accepted"] + st["duplicates"] + st["out_of_order"] == 6   # all 6 lines fully processed
    assert wait_for(lambda: done(server.status()))
    st = server.status()
    assert st["accepted"] == 4 and st["duplicates"] == 1 and st["out_of_order"] == 1
    assert server.slot.get()[1].frame.frame == 4
    c.close()


def test_second_client_rejected_then_reconnect_resets(server, gen):
    c1, _ = connect(server)
    c1.sendall(b"".join(line(gen.frame_at(i / 30, 100 + i)) for i in range(3)))
    assert wait_for(lambda: server.status()["accepted"] == 3)
    c2 = socket.create_connection(("127.0.0.1", server.port), timeout=2)
    c2.settimeout(2)
    assert b"controller_busy" in c2.recv(4096)
    assert c2.recv(4096) == b""                         # closed by the server
    assert server.status()["rejected_clients"] == 1
    c1.close()
    assert wait_for(lambda: server.status()["state"] == "disconnected")
    epoch = server.status()["epoch"]
    c3, _ = connect(server)                             # device rebooted: frame numbers restart low
    c3.sendall(line(gen.frame_at(0.0, 0)))
    assert wait_for(lambda: server.status()["accepted"] == 4)
    assert server.status()["epoch"] == epoch + 1 and server.slot.get()[1].frame.frame == 0
    c3.close()


def test_burst_keeps_only_latest(server, gen):
    c, _ = connect(server)
    c.sendall(b"".join(line(gen.frame_at(i / 1000, i)) for i in range(400)))
    assert wait_for(lambda: server.status()["accepted"] == 400, timeout=10)
    seq, snap = server.slot.get()
    assert snap.frame.frame == 399                       # the reader sees the newest frame, not a backlog
    assert server.slot.overwritten >= 1
    c.close()


def test_port_in_use_reports_clearly(server):
    other = TelemetryServer("127.0.0.1", server.port, log=lambda *_: None)
    with pytest.raises(OSError, match="cannot listen"):
        other.start()
