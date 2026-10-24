"""Dermis console: metric definitions on known sequences, bounded buffers, both console modes over
real HTTP, controls reaching the real backend, and the fragile-cup live panel."""
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from dermis.dashboard.live import LiveHub
from dermis.dashboard.metrics import U32, IngestMetrics, MissingTracker
from dermis.server.dev_sender import SyntheticGlove
from dermis.telemetry.pipeline import RawMessage, TelemetryPipeline

ROOT = Path(__file__).resolve().parents[1]
MS = 1_000_000


# --------------------------------------------------------------------- metrics on known sequences

def test_missing_ids_gaps_duplicates_late_frames_and_resets():
    m = MissingTracker()
    for f in [10, 11, 14, 14, 15, 19]:              # gaps {12, 13} and {16, 17, 18}; 14 repeated
        m.accept(f)
    assert sorted(m.missing) == [12, 13, 16, 17, 18] and m.count == 5
    m.late(13)                                       # a late frame that fills a gap is not missing
    m.late(99)                                       # a late frame never counted missing changes nothing
    assert m.count == 4 and m.late_filled == 1
    m.accept(5)                                      # backwards jump (restart) adds nothing
    assert m.count == 4
    o = MissingTracker()                             # late frame reported before its gap is registered
    o.accept(1); o.late(2); o.accept(3)
    assert o.count == 0 and o.late_filled == 1
    w = MissingTracker()
    for f in [U32 - 2, U32 - 1, 1]:                  # u32 wraparound: only frame 0 is missing
        w.accept(f)
    assert sorted(w.missing) == [0]
    big = MissingTracker()
    big.accept(0); big.accept(50_000)                # huge gap counted without storing every id
    assert big.count == 49_999 and len(big.missing) == 0


def test_ingest_metrics_rates_jitter_scopes_and_reset():
    m = IngestMetrics(window_s=1.0, n_channels=4, t_start_ns=0)
    t = 0
    for k in range(200):                             # 2 s at 100 Hz, alternating 8/12 ms intervals
        t += (8 if k % 2 else 12) * MS
        m.line("accepted", k, t, 40.0)
        m.accepted(k, t, 3, 4)
    m.line("malformed", None, t, 10.0)
    m.line("duplicate", 199, t, 10.0)
    m.line("out_of_order", 150, t, 10.0)
    s = m.summary(now_ns=t)
    assert s["accepted_fps"] == pytest.approx(100.0, abs=1.0)
    assert s["received_fps"] == pytest.approx(103.0, abs=1.0)
    assert s["interval_ms"] == pytest.approx(10.0, abs=0.05) and s["jitter_ms"] == pytest.approx(2.0, abs=0.05)
    assert s["usable_pct"] == 75.0 and "4 channels per frame" in s["usable_denominator"]
    assert s["session"] == {"received": 203, "accepted": 200, "rejected": 1, "duplicates": 1, "out_of_order": 1}
    assert s["rejected_by"] == {"malformed": 1} and s["processing_us"]["median"] == 40.0
    m.stream_reset("connection 2", t, new_connection=True)
    m.accepted(0, t + 10 * MS, 4, 4)
    s2 = m.summary(now_ns=t + 10 * MS)
    assert s2["connection"]["received"] == 0 and s2["session"]["received"] == 203
    assert s2["missing_ids"]["connection"] == 0      # frame 0 after reconnect is not a 199-frame gap


def test_hub_states_bounds_and_decimation(decoded_frame):
    pipe = TelemetryPipeline()
    hub = LiveHub(pipe.pod_order, frames=500)
    snap = pipe.ingest(RawMessage(decoded_frame, time.monotonic_ns()), "TEST")
    hub.on_snapshot(snap)
    pods = {p["pod"]: p for p in hub.pod_summary(lambda pod: None)}
    assert pods["index_tip"]["health"] == "healthy"
    assert pods["middle_tip"]["imu"] == "stale" and pods["middle_tip"]["health"] == "degraded"
    assert pods["middle_tip"]["mag"] == "fresh" and "TEMP_STALE" in pods["middle_tip"]["mag_flags"]   # TEMP_STALE != stale
    assert pods["thenar_apb"]["mag"] == "invalid" and pods["thenar_apb"]["b_abs_ut"] is None
    g = SyntheticGlove(stationary_s=0)
    t0 = time.monotonic_ns()
    for i in range(3000):
        hub.on_snapshot(pipe.ingest(RawMessage(g.frame_at(i / 100, 2000 + i), t0 + i * 10 * MS), "S"))
    assert len(hub.frames) == 500                                # bounded
    s, cur = hub.samples("index_tip", 0, lambda p: None, max_samples=60)
    assert len(s) == 60 and s[-1]["f"] == 2000 + 2999           # decimated, newest kept
    s2, cur2 = hub.samples("index_tip", cur, lambda p: None)
    assert s2 == [] and cur2 == cur
    base = np.array([0.0, 0.0, 27000.0])
    s3, _ = hub.samples("index_tip", cur - 3, lambda p: base)
    assert len(s3) == 3 and s3[0]["db"][2] == pytest.approx(s3[0]["b"][2] - 27000.0, abs=0.02)


# --------------------------------------------------------------------- HTTP helpers

class Client:
    def __init__(self, base):
        self.base = base

    def get(self, p):
        return json.loads(urllib.request.urlopen(self.base + p, timeout=10).read())

    def post(self, p, body=None, header=True):
        r = urllib.request.Request(self.base + p, data=json.dumps(body or {}).encode(),
                                   headers={"Content-Type": "application/json", **({"X-Dermis": "1"} if header else {})})
        try:
            return 200, json.loads(urllib.request.urlopen(r, timeout=30).read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def stream(self, p, n):
        """Read n server-sent events from a fresh connection (like a browser tab), then close it."""
        out = []
        with urllib.request.urlopen(self.base + p, timeout=10) as r:
            for line in r:
                if line.startswith(b"data: "):
                    out.append(json.loads(line[6:]))
                    if len(out) >= n:
                        break
        return out


def monitor_args(tmp_path, **kw):
    a = dict(source="synthetic", path=None, rate=100.0, loop=False, host="127.0.0.1", port=0, serial_port=None,
             wire=False, faults="", pods="index_tip,thumb_tip,middle_tip,thenar_apb", calibration=None, http_port=0,
             data_root=str(tmp_path / "eps"), runs_root=str(tmp_path / "runs"), results_root=str(tmp_path / "res"),
             recordings=str(tmp_path / "tel"))
    a.update(kw)
    return argparse.Namespace(**a)


# --------------------------------------------------------------------- monitor console

def test_monitor_console_live_raw_and_refusals(tmp_path):
    from dermis.dashboard.server import serve_in_thread
    srv, dash = serve_in_thread(monitor_args(tmp_path, faults="duplicate,reorder,invalid,stale"))
    c = Client(f"http://127.0.0.1:{srv.server_address[1]}")
    try:
        time.sleep(2.5)
        con = c.get("/api/console")
        assert con["kind"] == "monitor" and con["status"]["source"] == "SYNTHETIC" and "missing_ids" in con["definitions"]
        assert con["status"]["control_mode"] == "NO ROBOT CONTROL"
        live = c.get("/api/live?pod=index_tip")
        m = live["metrics"]
        assert m["accepted_fps"] > 50 and live["samples"] and live["pods"][0]["pod"] == "index_tip"
        assert m["session"]["duplicates"] >= 1 and m["session"]["out_of_order"] >= 1
        raw = c.get("/api/raw")
        assert raw["last_line"]["verdict"] == "accepted" and json.loads(raw["last_line"]["text"])["frame"] == raw["device"]["frame"]
        assert c.post("/api/control/arm", header=False)[0] == 403                       # needs X-Dermis header
        code, r = c.post("/api/control/arm")
        assert code == 400 and "no robot in this process" in r["error"]
        code, r = c.post("/api/control/demo_start")
        assert code == 400
        code, r = c.post("/api/control/record_start", {"task": "t1"})
        assert code == 200 and r["path"].endswith(".jsonl")
        time.sleep(0.5)
        code, r = c.post("/api/control/record_stop", {"outcome": "ok"})
        assert code == 200 and r["frames"] > 10 and Path(r["path"] + ".meta.json").exists()
        rec = c.get("/api/recordings")
        assert rec[0]["task"] == "t1" and rec[0]["outcome"] == "ok"
        ev = c.stream("/api/stream?pod=thumb_tip", 3)                                  # browser tab
        assert len(ev) == 3 and ev[0]["pod"] == "thumb_tip" and ev[0]["status"]["source"] == "SYNTHETIC"
        ev2 = c.stream("/api/stream?pod=thumb_tip", 2)                                 # refresh: new stream works
        assert ev2[0]["cursor"]["s"] >= ev[-1]["cursor"]["s"]
    finally:
        srv.shutdown()


def test_monitor_calibration_confirmation_and_replay_labels(tmp_path, fixture_path):
    from dermis.dashboard.server import serve_in_thread
    srv, dash = serve_in_thread(monitor_args(tmp_path, source="replay", path=str(fixture_path("decoded_frame.json")),
                                             rate=50.0, loop=True))
    c = Client(f"http://127.0.0.1:{srv.server_address[1]}")
    try:
        time.sleep(1.0)
        st = c.get("/api/console")["status"]
        assert st["source"] == "REPLAY" and "REPLAY" in st["source_note"]
        live = c.get("/api/live")
        assert live["metrics"]["session"]["duplicates"] >= 1          # the single fixture frame repeats: duplicates
        code, r = c.post("/api/control/calibrate", {"frames": 20})
        assert code == 400 and "no new frame accepted" in r["error"]   # refuses instead of waiting 30 s
    finally:
        srv.shutdown()


# --------------------------------------------------------------------- teleop console (integration)

@pytest.mark.slow
def test_teleop_console_controls_faults_reconnect(tmp_path):
    from dermis.dashboard.server import serve_teleop
    from dermis.server.teleop.session import TeleopSession
    sess = TeleopSession("127.0.0.1", 0, gui=False, calib_dir=str(tmp_path / "cal"), log=lambda *_: None)
    _, tcp_port = sess.start()
    srv, con = serve_teleop(sess, 0, data_root=str(tmp_path / "eps"), runs_root=str(tmp_path / "runs"),
                            results_root=str(tmp_path / "res"), recordings=str(tmp_path / "tel"))
    c = Client(f"http://127.0.0.1:{srv.server_address[1]}")
    out, done = {}, threading.Event()

    def script():
        try:
            st = c.get("/api/console")["status"]
            out["before"] = (st["source"], st["connection"]["state"], st["control_mode"])
            sender = subprocess.Popen([sys.executable, "-m", "dermis.server.dev_sender", "--port", str(tcp_port),
                                       "--rate", "100", "--duration", "9", "--stationary-s", "3",
                                       "--faults", "malformed@4,duplicate@4.3,pause@6:1.0,reboot@7.5:0.5"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            time.sleep(1.5)
            out["cal"] = c.post("/api/control/calibrate")
            time.sleep(1.5)
            out["cal_again"] = c.post("/api/control/calibrate")
            out["arm"] = c.post("/api/control/arm")
            out["armed_mode"] = c.get("/api/console")["status"]["control_mode"]
            out["rec"] = c.post("/api/control/record_start", {"task": "integration"})
            out["demo"] = c.post("/api/control/demo_start", {"task": "integration"})
            time.sleep(1.0)
            out["mark"] = c.post("/api/control/demo_mark", {"text": "contact"})
            time.sleep(0.5)
            out["demo_stop"] = c.post("/api/control/demo_stop", {"outcome": "success"})
            out["rec_stop"] = c.post("/api/control/record_stop", {"outcome": "ok"})
            sender.wait(20)
            time.sleep(0.6)
            out["live"] = c.get("/api/live?pod=index_tip")
            out["episodes"] = c.get("/api/episodes")
            ep = [e for e in out["episodes"] if e["kind"] == "teleop_demonstration"]
            out["episode"] = c.get("/api/episode?path=" + urllib.request.quote(ep[0]["path"])) if ep else None
        except Exception as e:                      # surfaced by the asserts below
            out["error"] = repr(e)
        finally:
            done.set()

    threading.Thread(target=script, daemon=True).start()
    t_end = time.monotonic() + 40
    try:
        while not done.is_set() and time.monotonic() < t_end:          # the simulation (main) thread
            t0 = time.perf_counter()
            row = sess.step()
            con.provider.service(row, (time.perf_counter() - t0) * 1000)
            time.sleep(1 / 30)
    finally:
        sess.close()
        srv.shutdown()
    assert "error" not in out, out.get("error")
    assert out["before"] == ("NO SOURCE", "listening", "DISARMED")
    assert out["cal"][0] == 200 and out["cal"][1]["ok"]
    assert out["cal_again"][1].get("needs_confirmation")              # never replaced silently
    assert out["arm"][1]["ok"] and out["armed_mode"] == "TELEOPERATION"
    assert out["rec_stop"][1]["frames"] > 50 and Path(out["rec_stop"][1]["path"]).exists()
    assert out["demo_stop"][1]["ticks"] > 20
    live = out["live"]
    m, st = live["metrics"], live["status"]
    assert m["session"]["rejected"] >= 2 and m["rejected_by"].get("malformed", 0) >= 2
    assert m["session"]["duplicates"] >= 1
    assert st["control_mode"] == "DISARMED" and "re-arm" in (st["control_reason"] or "")   # sender gone
    kinds = {e["kind"] for e in live["events"]} | {e["kind"] for e in c_events(con)}
    assert "control" in kinds and "connection" in kinds
    assert any("PAUSED" in e["text"] for e in c_events(con))           # watchdog pause on the timeline
    assert m["connection"]["received"] < m["session"]["received"]     # reconnect started a new connection scope
    ep = out["episode"]
    assert ep and ep["bc_ready"] is False and any(a["kind"] == "marker: contact" for a in ep["annotations"])
    assert ep["meta"]["source"] == "SYNTHETIC" and ep["meta"]["termination"] == "success"


def c_events(con):
    return con.provider.hub.events_since(0)[0]


# --------------------------------------------------------------------- files: evaluation, training, inference

def test_evaluation_grouping_variation_and_inference(tmp_path):
    from dermis.dashboard.server import FileViews
    res = tmp_path / "res"
    (res / "cup").mkdir(parents=True)
    pol = lambda n: {"kind": "behavioral_cloning", "label": "AUTONOMOUS", "name": n}
    summ = lambda k, a=10: {"attempts": a, "successes": k, "success_rate": k / a, "episodes_with_drop": 1,
                            "episodes_with_excess_force": 0, "timeouts": 2, "terminations": {}}
    for i, k in enumerate((4, 6, 9)):
        (res / f"bc_x_s{i}.json").write_text(json.dumps({"suite": "heldout_v1", "policy": pol(f"bc_x_s{i}"), "summary": summ(k)}))
    (res / "other.json").write_text(json.dumps({"suite": "dev_v1", "policy": pol("bc_y"), "summary": summ(5)}))
    (res / "cup" / "c.json").write_text(json.dumps({"task": "fragile_cup", "suite": "cup_dev_v1", "damage_model": "instant",
        "controller": {"name": "fixed_14N", "label": "SCRIPTED", "kind": "scripted"},
        "summary": {"attempts": 4, "successes": 3, "success_rate": 0.75, "terminations": {"damage_break": 1}}}))
    fv = FileViews()
    fv.a = argparse.Namespace(data_root=str(tmp_path / "eps"), runs_root=str(ROOT / "runs/benchmark/runs"),
                              results_root=str(res), recordings=str(tmp_path / "tel"))
    fv.roots = [Path(fv.a.runs_root), res]
    ev = fv.evaluation()
    assert set(ev["groups"]) >= {"grasp_lift_place | heldout_v1", "grasp_lift_place | dev_v1", "fragile_cup | cup_dev_v1 | instant"}
    cup = ev["groups"]["fragile_cup | cup_dev_v1 | instant"][0]
    assert cup["excess_load"] == 1 and cup["label"] == "SCRIPTED"
    v = [x for x in ev["variation"] if x["family"] == "bc_x"][0]
    assert v["runs"] == 3 and (v["min"], v["max"]) == (0.4, 0.9) and v["mean"] == pytest.approx(0.6333, abs=1e-3)
    ckpt = ROOT / "runs/benchmark/runs/bc_contact"
    if (ckpt / "model.pt").exists():
        t = fv.inference_timing(str(ckpt), n=50)
        assert 0 < t["median_ms"] < 50
    with pytest.raises(PermissionError):
        fv.inference_timing("/etc")


def test_training_failure_and_job_failure_states(tmp_path):
    from dermis.dashboard.server import FileViews, JobManager
    run = tmp_path / "runs" / "broken"
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"state": "failed", "epoch": 3, "epochs": 60, "error": "CUDA out of memory"}))
    fv = FileViews()
    fv.a = argparse.Namespace(data_root=str(tmp_path), runs_root=str(tmp_path / "runs"), results_root=str(tmp_path), recordings=str(tmp_path))
    fv.roots = [tmp_path]
    r = fv.runs()[0]
    assert r["status"]["state"] == "failed" and r["error"] == "CUDA out of memory" and not r["has_checkpoint"]
    jm = JobManager(tmp_path / "jobs")
    with pytest.raises(ValueError):
        jm.start(["rm", "-rf", "/"])
    j = jm.start(["train", "--config", str(tmp_path / "missing.yaml")])
    for _ in range(120):
        if jm.view(j["id"])["state"] != "running":
            break
        time.sleep(0.25)
    assert jm.view(j["id"])["state"].startswith("failed")


# --------------------------------------------------------------------- fragile cup live panel

@pytest.mark.slow
def test_cup_worker_pour_and_select(tmp_path):
    from dermis.dashboard.providers import CupWorker
    hub = LiveHub(("index_tip",))
    w = CupWorker(controllers=["grip_load_expert", "fixed_14N"], hub=hub).start()
    try:
        for _ in range(40):
            time.sleep(0.25)
            if w.status()["snapshot"] or w.status()["error"]:
                break
        st = w.status()
        assert st["running"], st["error"]
        assert st["kinds"]["grip_load_expert"].startswith("SCRIPTED") and "never a policy input" in st["labels"]["applied_load_n"]
        assert w.control("pour", {})["ok"]
        for _ in range(60):
            time.sleep(0.25)
            if w.status()["snapshot"]["outcome"]:
                break
        assert w.status()["snapshot"]["outcome"] is not None
        assert w.control("select", {"policy": "fixed_14N"})["active"] == "fixed_14N"
        with pytest.raises(ValueError):
            w.control("select", {"policy": "nope"})
        with pytest.raises(ValueError):
            w.control("effort", {"n": 30})                         # no motor-command path exists
    finally:
        w.stop()


def test_monitor_tcp_source_is_never_labelled_glove(tmp_path):
    """--source tcp: the dev sender's synthetic-* device is SYNTHETIC, a fixture device id is TCP-UNVERIFIED."""
    import socket
    from dermis.dashboard.server import serve_in_thread
    with socket.socket() as s0:
        s0.bind(("127.0.0.1", 0)); port = s0.getsockname()[1]
    srv, dash = serve_in_thread(monitor_args(tmp_path, source="tcp", port=port, rate=None))
    c = Client(f"http://127.0.0.1:{srv.server_address[1]}")
    try:
        assert c.get("/api/console")["status"]["source"] == "NO SOURCE"
        for mode, want in (("synthetic", "SYNTHETIC"), ("fixture", "TCP-UNVERIFIED")):
            p = subprocess.Popen([sys.executable, "-m", "dermis.server.dev_sender", "--port", str(port), "--mode", mode,
                                  "--duration", "1.5", "--repeat", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            p.wait(20)
            time.sleep(0.4)
            assert c.get("/api/console")["status"]["source"] == want, mode
    finally:
        srv.shutdown()
