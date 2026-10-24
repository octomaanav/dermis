"""Ingestion in a child process (used next to PyBullet, which starves in-process threads)."""
import json
import time

import pytest

from dermis.glove.procsource import ProcessTelemetry


@pytest.mark.slow
def test_process_telemetry_delivers_fresh_snapshots_and_records(tmp_path):
    raw = tmp_path / "raw.jsonl"
    pt = ProcessTelemetry({"kind": "synthetic", "rate": 100.0, "profile": "step"},
                          ["index_tip", "thumb_tip"], record_path=str(raw))
    t0 = time.time()
    while pt.latest() is None and time.time() - t0 < 20:
        time.sleep(0.05)
    assert pt.latest() is not None and not pt.source_stale()
    time.sleep(1.0)
    s = pt.latest()
    assert (time.monotonic_ns() - s.frame.t_recv_ns) / 1e9 < 0.25     # fresh across processes
    info = pt.stop_recording()
    assert not pt.proc.is_alive() and info["frames"] and info["frames"] > 30
    lines = raw.read_text().splitlines()
    assert len(lines) == info["frames"] and "_t_recv_ns" in json.loads(lines[0])
