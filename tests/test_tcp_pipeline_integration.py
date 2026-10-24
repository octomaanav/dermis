"""Hardware-free end-to-end check: receiver + separate synthetic sender process + headless PyBullet.

About 20 s of real time per case (the sender streams in real time). Asserts on measured simulator state.
"""
import pytest

from dermis.server.teleop.pipeline_check import run_check


@pytest.mark.slow
@pytest.mark.parametrize("robot,fmt", [("hand", "v1"), ("hand", "draft"), ("arm", "v1")])
def test_tcp_stream_drives_simulated_robot(tmp_path, robot, fmt):
    r = run_check(str(tmp_path), faults="ci", log=lambda *_: None, robot=robot, fmt=fmt)
    failed = [f"{c['name']}: {c['detail']}" for c in r["checks"] if not c["ok"]]
    assert r["ok"], "\n".join(failed)
    assert (tmp_path / "result.json").exists() and (tmp_path / "received.jsonl").exists()
    if fmt == "draft":
        assert r["server_status"]["normalized"] > 300 and r["server_status"]["schema_rejected"] <= 1
