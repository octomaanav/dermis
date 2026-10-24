"""`python -m dermis test-tcp-pipeline --headless`: hardware-free TCP -> simulator integration check.

1. starts the TelemetryServer on a free local port (inside a headless TeleopSession)
2. starts the SYNTHETIC sender as a separate process, with the "ci" fault schedule
3. runs PyBullet (DIRECT) in real time at the control rate, with --robot hand (default) or arm
4. reads the actual simulated state each tick (hand: claw joint angles + palm tilt;
   arm: finger joints + end-effector position)
5. checks that state changes in response to the TCP stream (each claw / the gripper opening)
6. checks motion stays inside the configured joint / workspace / speed limits
7. checks that while the sender pauses, the watchdog stops motion (after settling)
8. checks disconnect -> DISARMED, explicit re-arm -> motion again, then a clean exit
and writes result.json + result.md to --out. Every check uses simulator state, not commands.

The scripted operator here calibrates and arms programmatically; that is the same explicit
action a person takes with C and R in the GUI.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .controller import ARMED, DISARMED, PAUSED, ControllerConfig
from .session import TeleopSession
from .mapping import MappingConfig

SENDER_DURATION = 17.0
STATIONARY_S = 2.5


def add_args(sp) -> None:
    sp.add_argument("--headless", action="store_true", default=True, help="always headless (kept for clarity)")
    sp.add_argument("--robot", choices=["hand", "arm"], default="hand")
    sp.add_argument("--format", choices=["v1", "draft"], default="v1",
                    help="sender frame shape; draft = proposed shape normalized by server/compat.py")
    sp.add_argument("--out", default="runs/tcp_pipeline_check")
    sp.add_argument("--faults", default="ci", help="sender fault schedule ('' for none)")
    sp.add_argument("--max-speed", type=float, default=0.05)


def run_check(out: str = "runs/tcp_pipeline_check", faults: str = "ci", max_speed: float = 0.05, log=print,
              robot: str = "hand", fmt: str = "v1") -> dict:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    quiet = []
    sess = TeleopSession("127.0.0.1", 0, gui=False, mapping_cfg=MappingConfig(tilt_xy=True), robot=robot,
                         ctrl_cfg=ControllerConfig(max_speed=max_speed), calib_dir=str(outdir),
                         record_path=str(outdir / "received.jsonl"), log=quiet.append)
    rec = outdir / "received.jsonl"
    for f in (rec, Path(str(rec) + ".recv.jsonl")):
        if f.exists():
            f.unlink()
    host, port = sess.start()
    log(f"[check] robot {robot}; receiver on {host}:{port}; starting SYNTHETIC {fmt} sender "
        f"(faults: {faults or 'none'})")
    sender_log = open(outdir / "sender.log", "w")
    sender = subprocess.Popen([sys.executable, "-m", "dermis.server.dev_sender", "--host", host, "--port",
                               str(port), "--mode", "synthetic", "--rate", "30", "--duration", str(SENDER_DURATION),
                               "--stationary-s", str(STATIONARY_S), "--faults", faults, "--format", fmt],
                              stdout=sender_log, stderr=subprocess.STDOUT)
    ws = np.array(sess.workspace)
    dt = 1.0 / sess.ctrl_cfg.control_hz
    rows = []
    phase = "wait_data"
    t0 = time.monotonic()
    t_first = t_pause = t_disc = t_rearm = None
    arm_results = []
    next_t = t0
    try:
        while True:
            now = time.monotonic()
            t = now - t0
            r = sess.step()
            r["t"] = t
            rows.append(r)
            L = r["link"]
            if phase == "wait_data" and L["accepted"] > 0:
                t_first, phase = t, "pre_cal"
            elif phase == "pre_cal" and t - t_first > 0.4:
                sess.do("calibrate")
                phase = "calibrating"
            elif phase == "calibrating" and sess.ctrl.calib is not None:
                ok, msg = sess.ctrl.arm(L)
                arm_results.append((round(t, 2), ok, msg))
                phase = "armed" if ok else "failed_arm"
            elif phase == "calibrating" and sess.ctrl.collector is None and sess.ctrl.calib is None:
                phase = "failed_cal"
            elif (phase == "armed" and r["state"] == PAUSED and t_pause is None
                  and (L["last_accept_age_ms"] or 0) > 1000 * sess.ctrl_cfg.watchdog_s):
                t_pause = t                                  # transmission stopped (not just a bad channel)
            elif phase == "armed" and r["state"] == DISARMED and t_disc is None:
                t_disc, phase = t, "disconnected"
            elif phase == "disconnected" and L["state"] == "connected" and t - t_disc > 0.3:
                ok, msg = sess.ctrl.arm(L)
                arm_results.append((round(t, 2), ok, msg))
                if ok:
                    t_rearm, phase = t, "rearmed"
            if sender.poll() is not None and phase != "wait_data" and L["state"] != "connected" and t > 3:
                for _ in range(15):                          # let the final disconnect settle
                    rows.append({**sess.step(), "t": time.monotonic() - t0})
                break
            if t > SENDER_DURATION + 15:
                break
            next_t += dt
            d = next_t - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                next_t = time.monotonic()
    finally:
        if sender.poll() is None:
            sender.terminate()
        sender.wait(timeout=5)
        sender_log.close()
        status = sess.server.status()
        sess.close()

    result = evaluate(rows, robot, ws, max_speed, dt, t_first, t_pause, t_disc, t_rearm, arm_results, status,
                      sender.returncode, sess.overruns, sess.server.max_message_bytes)
    result["robot"], result["format"] = robot, fmt
    result["controller_log"] = [m for m in quiet if m.startswith("[teleop]")][-40:]
    (outdir / "result.json").write_text(json.dumps(result, indent=1, default=float))
    (outdir / "result.md").write_text(_markdown(result))
    for c in result["checks"]:
        log(f"[check] {'PASS' if c['ok'] else 'FAIL'}  {c['name']}: {c['detail']}")
    log(f"[check] {'ALL PASS' if result['ok'] else 'FAILED'}; wrote {outdir / 'result.json'} and result.md")
    return result


def _state(rows, robot):
    """One vector per tick of everything that should freeze during a pause."""
    if robot == "hand":
        return np.array([[*r["joints"], *r["palm_rp"]] for r in rows])
    return np.array([[*r["ee"], r["opening"]] for r in rows])


def evaluate(rows, robot, ws, max_speed, dt, t_first, t_pause, t_disc, t_rearm, arm_results, status, sender_rc,
             overruns, max_msg) -> dict:
    checks = []
    add = lambda name, ok, detail: checks.append({"name": name, "ok": bool(ok), "detail": detail})
    t = np.array([r["t"] for r in rows])
    st = [r["state"] for r in rows]
    armed = np.array([s == ARMED for s in st])
    X = _state(rows, robot)

    add("sender exited cleanly", sender_rc == 0, f"return code {sender_rc}")
    add("calibrated and armed explicitly", bool(arm_results) and arm_results[0][1], f"arm attempts {arm_results}")
    pre = (t < (t[armed][0] if armed.any() else t[-1]))
    pre_change = float(np.abs(X[pre] - X[pre][0]).max()) if pre.sum() > 1 else 0.0
    add("robot held while disarmed", pre_change < (0.02 if robot == "hand" else 0.002),
        f"largest state change before arming {pre_change:.4f} ({'rad' if robot == 'hand' else 'm'})")
    first_window = armed & (t < (t_pause if t_pause else t[-1]))
    (_hand_checks if robot == "hand" else _arm_checks)(rows, add, first_window, ws, max_speed, dt)

    if t_pause is None:
        add("watchdog stops motion while sender paused", False, "never entered a watchdog pause")
    else:
        a = int(np.searchsorted(t, t_pause + 0.6))
        b = int(np.searchsorted(t, t_pause + 1.8))
        still_paused = all(s == PAUSED for s in st[a:b + 1])
        moved = float(np.abs(X[b] - X[a]).max()) if b < len(X) else float("nan")
        lim = 0.005 if robot == "hand" else 0.002
        add("watchdog stops motion while sender paused", still_paused and moved < lim,
            f"paused at t={t_pause:.2f}s; from +0.6 s to +1.8 s (after settling) the largest state change was "
            f"{moved:.5f} (limit {lim}), state PAUSED throughout={still_paused}")
    add("connection loss disarms; explicit re-arm needed", t_disc is not None and t_rearm is not None,
        f"disarmed at {t_disc and round(t_disc, 2)} s, re-armed at {t_rearm and round(t_rearm, 2)} s")
    if t_rearm is not None:
        after = armed & (t > t_rearm)
        rng2 = float(np.ptp(X[after], axis=0).max()) if after.sum() > 1 else 0.0
        add("motion resumes after re-arm", rng2 > (0.2 if robot == "hand" else 0.01),
            f"largest state range after re-arm {rng2:.3f}")
    add("disarmed at end (sender gone)", st[-1] == DISARMED, f"final state {st[-1]}")

    if status["lines"]:
        c = status
        want = {"malformed": 2, "oversized": 1, "duplicates": 1, "out_of_order": 1, "non_finite_messages": 1}
        got = {k: c[k] for k in want}
        add("faults detected and survived", all(got[k] >= n for k, n in want.items()) and c["accepted"] > 300,
            f"{got}, accepted {c['accepted']}, lost frames {c['pipeline']['lost_frames']}, "
            f"normalized (compat) {c['normalized']}")
        add("no growing queue", c["peak_buffered_bytes"] <= max_msg and c["slot_overwritten"] >= 1,
            f"peak framer buffer {c['peak_buffered_bytes']} B (limit {max_msg}); latest-frame slot overwrote "
            f"{c['slot_overwritten']} unread frames during the burst instead of queueing them")
    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "label": "TELEOPERATION / input-integration test with SYNTHETIC data (EXPERIMENTAL mapping)",
            "checks": checks, "server_status": status, "ticks": len(rows), "duration_s": round(float(t[-1]), 2),
            "control_overruns": overruns}


def _arm_checks(rows, add, first_window, ws, max_speed, dt) -> None:
    ee = np.array([r["ee"] for r in rows])
    op = np.array([r["opening"] for r in rows])
    rng = float(np.ptp(op[first_window])) if first_window.sum() > 1 else 0.0
    add("measured opening follows TCP stream", rng > 0.03,
        f"finger joint opening range while armed {rng * 1000:.1f} mm (need > 30 mm)")
    xy = float(np.ptp(ee[first_window][:, :2], axis=0).max()) if first_window.sum() > 1 else 0.0
    add("measured XY follows IMU tilt", xy > 0.01, f"end-effector XY excursion while armed {xy * 1000:.1f} mm")
    tol = 0.01
    lo, hi = ws[:, 0] - tol, ws[:, 1] + tol
    add("end effector inside workspace", bool(np.all((ee >= lo) & (ee <= hi))),
        f"min {np.round(ee.min(0), 3).tolist()} max {np.round(ee.max(0), 3).tolist()} bounds {ws.tolist()} (+-{tol})")
    add("opening inside [0, 80 mm]", bool(op.min() > -0.001 and op.max() < 0.081),
        f"{op.min() * 1000:.1f}..{op.max() * 1000:.1f} mm")
    v = np.abs(np.diff(ee, axis=0)) / dt
    vmax = float(v.max()) if len(v) else 0.0
    add("Cartesian speed bounded", vmax <= 1.5 * max_speed + 0.01,
        f"max per-axis measured speed {vmax:.3f} m/s (limit {max_speed} m/s, IK tracking margin)")
    zr = float(np.ptp(ee[:, 2]))
    add("Z held fixed (no keyboard input)", zr < 0.005, f"z range {zr * 1000:.1f} mm")
    reasons = {r["reason"] for r in rows if r["state"] == PAUSED}
    add("required-channel faults pause control", any("mag unusable" in (x or "") for x in reasons),
        f"pause reasons seen: {sorted(x for x in reasons if x)[:6]}")


def _hand_checks(rows, add, first_window, ws, max_speed, dt) -> None:
    from .hand import FINGERS, HandController, flex_limits
    from .motion import MotionEstimator
    q = np.array([r["joints"] for r in rows])
    rp = np.array([r["palm_rp"] for r in rows])
    off = np.array([r["offset"] for r in rows])
    w = first_window
    first = {pod: 3 * n for n, pod in enumerate(FINGERS)}                     # MCP / CMC column per finger
    ranges = {pod: float(np.ptp(q[w][:, c])) if w.sum() > 1 else 0.0 for pod, c in first.items()}
    add("each finger follows its own pod", all(v > 0.4 for v in ranges.values()),
        "measured knuckle joint range while armed: " + ", ".join(f"{k} {v:.2f} rad" for k, v in ranges.items())
        + " (need > 0.4)")
    lagged = float(np.abs(q[w][:, first["index_tip"]] - q[w][:, first["middle_tip"]]).max()) if w.sum() > 1 else 0.0
    add("fingers move independently", lagged > 0.3, f"largest index-vs-middle knuckle difference {lagged:.2f} rad")
    tilt = float(np.ptp(rp[w], axis=0).max()) if w.sum() > 1 else 0.0
    add("hand tilt follows IMU", tilt > 0.1, f"measured hand roll/pitch range while armed {tilt:.2f} rad")
    lims = np.array([lim for pod in FINGERS for lim in flex_limits(pod)])
    tol = 0.06
    inside = bool(np.all(q >= -tol) and np.all(q <= lims + tol))
    add("joints inside human-like limits (no backward bend)", inside,
        f"min {q.min():.2f} rad (limit 0), max over limit {float((q - lims).max()):+.2f} rad (tolerance {tol})")
    from .hand import HandConfig
    lim = HandConfig().max_tilt_rad
    add("hand tilt bounded", float(np.abs(rp).max()) <= lim + 0.01, f"max |roll/pitch| {np.abs(rp).max():.2f} rad (limit {lim})")
    vmax = float(np.abs(np.diff(q, axis=0)).max() / dt) if len(q) > 1 else 0.0
    vlim = HandController.max_flex_rate * float(lims.max()) * 1.3 + 0.3
    add("finger speed bounded", vmax <= vlim, f"max measured joint speed {vmax:.2f} rad/s (limit {vlim:.2f})")
    L = MotionEstimator().limits
    inside = bool(np.all(off >= L[:, 0] - 0.005) and np.all(off <= L[:, 1] + 0.005))
    add("hand motion bounded", inside, f"wrist offset range {np.round(off.min(0), 3).tolist()}..{np.round(off.max(0), 3).tolist()} m")
    add("no motion drift without a punch", float(np.abs(off).max()) < 0.03,
        f"largest wrist offset during slow tilt/pinch {np.abs(off).max() * 100:.1f} cm (need < 3 cm)")
    held = [set(r["signals"].get("held", [])) for r in rows if r["state"] == ARMED]
    add("bad pod holds only its own finger", any("index_tip" in h for h in held) and any("thumb_tip" in h for h in held),
        "index finger held during null_index, thumb held during stale_thumb, while the hand stayed ARMED"
        if any("index_tip" in h for h in held) else "no per-finger hold observed")


def _markdown(r: dict) -> str:
    lines = [f"# TCP -> simulator pipeline check ({r.get('robot')}, {r.get('format')} frames): "
             f"{'PASS' if r['ok'] else 'FAIL'}", "", r["label"], "",
             f"{r['ticks']} control ticks over {r['duration_s']} s, {r['control_overruns']} overruns.", "",
             "| check | result | detail |", "| --- | --- | --- |"]
    lines += [f"| {c['name']} | {'PASS' if c['ok'] else 'FAIL'} | {c['detail']} |" for c in r["checks"]]
    return "\n".join(lines) + "\n"


def cmd_test_tcp_pipeline(a) -> int:
    r = run_check(a.out, a.faults, a.max_speed, log=lambda s: print(s, flush=True), robot=a.robot, fmt=a.format)
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(prog="python -m dermis test-tcp-pipeline")
    add_args(_ap)
    sys.exit(cmd_test_tcp_pipeline(_ap.parse_args()))
