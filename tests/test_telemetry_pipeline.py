import copy
import io
import json
import socket
import threading
import time

import numpy as np
import pytest

from dermis.telemetry.pipeline import RawMessage, TelemetryPipeline
from dermis.telemetry.sources import LineSource, ReplaySource, SyntheticSource, TcpSource


def frame(base, n, t_us=None, config_id=3, flags=(), drop_pods=(), device="dev-a"):
    d = copy.deepcopy(base)
    d["frame"], d["config_id"], d["flags"], d["device_id"] = n, config_id, list(flags), device
    d["t_device_us"] = t_us if t_us is not None else 10_000_000 + n * 3_000
    for loc, pod in d["pods"].items():
        for k in ("mag", "imu"):
            if pod[k]:
                pod[k]["t_us"] = d["t_device_us"] + 500
                pod[k]["flags"] = []
                pod[k]["valid"] = True
        if pod["mag"]["b_ut"] is None:
            pod["mag"].update(b_ut=[1.0, 2.0, 3.0], temp_c=22.0, status=3)
    for loc in drop_pods:
        d["pods"].pop(loc)
    return d


def feed(pipe, d):
    return pipe.ingest(RawMessage(d, time.monotonic_ns()), "SYNTHETIC")


def test_pod_order_is_fixed_regardless_of_json_order(decoded_frame):
    a = frame(decoded_frame, 1)
    b = copy.deepcopy(a)
    b["pods"] = dict(reversed(list(b["pods"].items())))
    sa, sb = feed(TelemetryPipeline(), a), feed(TelemetryPipeline(), b)
    assert sa.pods == sb.pods == ("index_tip", "thumb_tip", "middle_tip", "thenar_apb")
    np.testing.assert_array_equal(sa.mag_raw, sb.mag_raw)


def test_missing_and_unknown_pods(decoded_frame):
    pipe = TelemetryPipeline()
    feed(pipe, frame(decoded_frame, 1))
    s = feed(pipe, frame(decoded_frame, 2, drop_pods=("thumb_tip",)))
    assert list(s.present) == [True, False, True, True]
    assert np.isnan(s.mag_raw[1]).all() and not s.mag_fresh[1]
    assert s.mag_held_mask[1]                    # last fresh thumb value is within max_hold
    assert pipe.counters["pod_set_changes"] == 1
    extra = frame(decoded_frame, 3)
    extra["pods"]["wrist"] = copy.deepcopy(extra["pods"]["index_tip"])
    s = feed(pipe, extra)
    assert "wrist" not in s.pods and pipe.counters["unknown_pods"] == 1


def test_duplicates_out_of_order_and_gaps(decoded_frame):
    pipe = TelemetryPipeline()
    assert feed(pipe, frame(decoded_frame, 10)) is not None
    assert feed(pipe, frame(decoded_frame, 10)) is None                  # duplicate
    s = feed(pipe, frame(decoded_frame, 14))
    assert s.lost_before == 3
    assert feed(pipe, frame(decoded_frame, 12)) is None                  # late: dropped, not re-applied
    c = pipe.counters
    assert (c["duplicates"], c["out_of_order"], c["lost_frames"], c["accepted"]) == (1, 1, 3, 2)


def test_counter_wraparound_is_a_gap_not_a_restart(decoded_frame):
    pipe = TelemetryPipeline()
    feed(pipe, frame(decoded_frame, 2**32 - 2))
    s = feed(pipe, frame(decoded_frame, 1, t_us=10_000_000 + 2**20))
    assert s.lost_before == 2 and pipe.counters["restarts"] == 0


def test_large_backward_jump_is_restart(decoded_frame):
    pipe = TelemetryPipeline()
    feed(pipe, frame(decoded_frame, 50_000))
    s = feed(pipe, frame(decoded_frame, 3, t_us=1_000))
    assert s is not None and pipe.counters["restarts"] == 1
    assert any(e["event"] == "stream_restart" for e in s.events)


def test_device_change_resets(decoded_frame):
    pipe = TelemetryPipeline()
    feed(pipe, frame(decoded_frame, 100))
    s = feed(pipe, frame(decoded_frame, 5, device="dev-b"))
    assert s is not None and pipe.counters["device_changes"] == 1 and s.lost_before == 0


def test_config_change_discards_held_values(decoded_frame):
    pipe = TelemetryPipeline()
    feed(pipe, frame(decoded_frame, 1))
    d = frame(decoded_frame, 2, config_id=4, flags=["CONFIG_CHANGED"])
    d["pods"]["index_tip"]["mag"]["flags"] = ["STALE"]
    s = feed(pipe, d)
    assert s.config_changed and pipe.counters["config_changes"] == 1
    assert not s.mag_held_mask[0]               # old-config value must not be carried over
    assert s.mag_held_mask[1]                   # fresh in the new frame


def test_stale_hold_is_bounded(decoded_frame):
    pipe = TelemetryPipeline(max_hold_us=50_000)
    feed(pipe, frame(decoded_frame, 1, t_us=1_000_000))
    first = None
    for k in range(2, 40):
        d = frame(decoded_frame, k, t_us=1_000_000 + (k - 1) * 3_000)
        d["pods"]["index_tip"]["mag"]["flags"] = ["STALE"]
        d["pods"]["index_tip"]["mag"]["t_us"] = 1_000_500        # repeated old sample
        s = feed(pipe, d)
        assert s.mag_stale[0] and not s.mag_fresh[0]
        if not s.mag_held_mask[0] and first is None:
            first = k
        if s.mag_held_mask[0]:
            assert s.mag_imputed[0]
    age_at_cutoff = 1_000_000 + (first - 1) * 3_000 - 1_000_500
    assert first is not None and 50_000 < age_at_cutoff <= 53_000
    assert np.isnan(s.mag_held[0]).all()


def test_robust_filter_rejects_a_single_sensor_spike(decoded_frame):
    pipe = TelemetryPipeline(mag_filter_window=5, mag_filter_tau_s=0.08)
    baseline = None
    for n in range(1, 8):
        s = feed(pipe, frame(decoded_frame, n))
        baseline = s.mag_filtered[0].copy()
    d = frame(decoded_frame, 8)
    d["pods"]["index_tip"]["mag"]["b_ut"] = (baseline + np.array([50_000.0, -50_000.0, 50_000.0])).tolist()
    s = feed(pipe, d)
    np.testing.assert_allclose(s.mag_filtered[0], baseline, atol=1e-9)


def test_source_watchdog():
    pipe = TelemetryPipeline(source_timeout_s=0.1)
    assert pipe.source_stale()                  # nothing received yet
    pipe.last_recv_ns = time.monotonic_ns()
    assert not pipe.source_stale()
    assert pipe.source_stale(time.monotonic_ns() + int(0.2e9))


def test_bad_lines_counted_not_fatal():
    pipe = TelemetryPipeline()
    for m in LineSource(io.StringIO('{"not": "a frame"}\nnot json\n[1,2]\n')):
        assert pipe.ingest(m, "GLOVE") is None
    assert pipe.counters["adapter_errors"] == 3


ALL_FAULTS = ("drop", "duplicate", "reorder", "stale", "temp_stale", "invalid", "missing_pod", "config_change",
              "gap", "null_valid")


def _run(source_iter, label):
    pipe = TelemetryPipeline()
    snaps = [s for m in source_iter if (s := pipe.ingest(m, label)) is not None]
    return pipe, snaps


def test_synthetic_faults_match_across_sources(tmp_path):
    """The same SYNTHETIC fault stream through replay, stdin and TCP gives identical pipeline results."""
    src = SyntheticSource(faults=ALL_FAULTS, n_frames=400, realtime=False)
    frames = list(src.frames())
    lines = "".join(json.dumps(f) + "\n" for f in frames)
    (tmp_path / "f.jsonl").write_text(lines)

    direct, snaps = _run((RawMessage(f, 0) for f in frames), "SYNTHETIC")
    replay, _ = _run(ReplaySource(str(tmp_path / "f.jsonl"), realtime=False), "REPLAY")
    stdin, _ = _run(LineSource(io.StringIO(lines)), "GLOVE")

    tcp = TcpSource("127.0.0.1", 0, once=True)

    def send():
        with socket.create_connection(("127.0.0.1", tcp.port)) as s:
            s.sendall(lines.encode())
    threading.Thread(target=send, daemon=True).start()
    tcp_pipe, _ = _run(tcp, "GLOVE")

    keys = ("accepted", "duplicates", "out_of_order", "lost_frames", "config_changes", "pod_set_changes")
    ref = {k: direct.counters[k] for k in keys}
    for p in (replay, stdin, tcp_pipe):
        assert {k: p.counters[k] for k in keys} == ref
    assert all(ref[k] > 0 for k in keys)
    assert any(s.mag_stale.any() for s in snaps)
    assert any((~s.mag_valid & s.present).any() for s in snaps)            # invalid + valid-but-null
    assert any(not s.present.all() for s in snaps)
    # TEMP_STALE alone never makes the field stale
    temp_only = [s for s in snaps if not s.mag_temp_fresh[0] and s.present[0] and s.mag_valid[0]]
    assert temp_only and all(s.mag_fresh[0] or s.mag_stale[0] for s in temp_only)
    assert any(s.mag_fresh[0] and not s.mag_temp_fresh[0] for s in temp_only)


def test_synthetic_restart_fault():
    pipe, _ = _run(SyntheticSource(faults=("restart",), n_frames=120, realtime=False), "SYNTHETIC")
    assert pipe.counters["restarts"] >= 1


def test_unknown_fault_rejected():
    with pytest.raises(ValueError):
        SyntheticSource(faults=("melt",))
