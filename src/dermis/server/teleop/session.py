"""`python -m dermis teleop --source tcp`: TELEOPERATION test of TCP glove telemetry -> simulated robot.

This is an input-integration test with an EXPERIMENTAL mapping. It is not training and not
autonomous control: every motion comes from received telemetry through TeleopController.

Threads
    main thread     PyBullet (GUI or DIRECT), keyboard, status display, control ticks at --control-hz
    dermis-tcp      asyncio TelemetryServer: framing, validation, pipeline -> LatestSlot
    stdin (opt.)    reads typed commands into a queue; the main thread executes them
No PyBullet call is ever made outside the main thread, and the main thread never waits on TCP.

Robot (--robot, robots.py):
    hand (default)  3-claw hand (hand.py): each claw follows its own pod's magnetic change, the palm
                    tilts with the wrist/index IMU, claw tips show each pod's sensor state.
    arm             Franka Panda parallel-jaw gripper (sim/env.py): position target integrated from
                    the commanded velocity, clamped to the teleop workspace, tracked with IK.
"""
from __future__ import annotations

import os
import queue
import re
import sys
import threading
import time


from ...telemetry.pipeline import TelemetryPipeline
from ..tcp_server import DEFAULT_HOST, DEFAULT_MAX_MESSAGE_BYTES, DEFAULT_PORT, TelemetryServer
from ...telemetry.model import DEFAULT_POD_ORDER
from .calibration import Calibration
from .controller import ARMED, DISARMED, PAUSED, ControllerConfig
from .mapping import MappingConfig
from .robots import TELEOP_WORKSPACE, make_robot

BANNER = "TELEOP TEST (EXPERIMENTAL mapping) - not training, not autonomous"
LAMP = {ARMED: (0.1, 0.8, 0.1, 1), PAUSED: (1.0, 0.55, 0.0, 1), DISARMED: (0.85, 0.1, 0.1, 1)}
POD_ORDER = (*DEFAULT_POD_ORDER, "wrist")        # wrist: IMU-only pod in the draft frame shape (compat.py)


def _env(name, fallback):
    v = os.environ.get(name)
    return type(fallback)(v) if v not in (None, "") else fallback


class TeleopSession:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, gui=False, control_hz=30, substeps=8,
                 max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES, mapping_cfg: MappingConfig | None = None,
                 ctrl_cfg: ControllerConfig | None = None, source_label=None, record_path=None,
                 calibration_path=None, calib_dir="runs/teleop", gui_text=False, auto=False, robot="hand",
                 hand_pods=None, compat=True, assume_valid=False, motion=True, motion_mode="free", log=print):
        self.log = log
        self.gui = gui
        self.gui_text = gui_text
        self.ctrl_cfg = ctrl_cfg or ControllerConfig(control_hz=control_hz)
        self.ctrl_cfg.control_hz = control_hz
        self.robot = make_robot(robot, gui, control_hz, substeps, self.ctrl_cfg, hand_pods, motion, motion_mode)
        self.workspace = TELEOP_WORKSPACE
        calib = Calibration.load(calibration_path) if calibration_path else None
        self.server = TelemetryServer(host, port, max_message_bytes, TelemetryPipeline(pod_order=POD_ORDER),
                                      source_label, record_path, compat=compat, assume_valid=assume_valid, log=log)
        self.ctrl = self.robot.make_controller(mapping_cfg, self.ctrl_cfg, calib, calib_dir, log)
        if getattr(self.robot, "motion", None) is not None:
            self.server.observers.append(self.robot.motion.update)    # every frame, network thread
        self.mapping = self.ctrl.mapping
        self.commands: queue.Queue = queue.Queue()
        self.tick_count = 0
        self.overruns = 0
        self._last_cmd = None
        self.quit = False
        self.auto = auto                      # auto-operator: calibrate and arm without C / R (see _auto_operate)
        self._auto_paused = False
        self._auto_epoch = None
        self._auto_epoch_t = 0.0
        self._auto_next_try = 0.0

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> tuple[str, int]:
        host, port = self.server.start()
        self.log(f"[teleop] {BANNER}")
        self.log(f"[teleop] robot: {self.robot.name}; mapping: {self.mapping.label}")
        self.log(f"[teleop] listening for decoded-frame NDJSON on {host}:{port}")
        if host == "0.0.0.0":
            self.log("[teleop] bound to all interfaces: clients connect to this Mac's LAN IP, not 0.0.0.0")
        if self.gui:
            self._draw_static()
        return host, port

    def close(self) -> None:
        self.server.stop()
        self.robot.close()

    # -- one control tick (main thread) -----------------------------------------

    def step(self, z_input: float = 0.0) -> dict:
        self._drain_commands()
        link = self.server.status()
        seq, snap = self.server.slot.get()
        cmd = self.ctrl.tick(seq, snap, link, z_input)
        motion = getattr(self.robot, "motion", None)
        if motion is not None:
            for notice in motion.pop_notices():
                self.log(f"[teleop] {notice}")
        if self.auto:
            self._auto_operate(link)
        self._last_cmd = cmd
        self.robot.apply(cmd)
        self.tick_count += 1
        signals = self.ctrl.status().signals
        return {"tick": self.tick_count, "state": cmd.state, "reason": cmd.reason, "cmd_v": cmd.v.tolist(),
                "cmd_opening": cmd.opening, "cmd": cmd.extra, "signals": signals, "link": link,
                **self.robot.measure()}

    def _drain_commands(self) -> None:
        while True:
            try:
                c = self.commands.get_nowait()
            except queue.Empty:
                return
            self.do(c)

    def _auto_operate(self, link: dict) -> None:
        """--auto: press C and R for the operator. Calibrates once frames have flowed for 0.5 s on a
        connection (the sender must be still and unloaded then, as the synthetic stream is for its
        first seconds), arms when input is usable, and re-arms after reconnects. Calibration is
        never redone while armed, and a manual X pauses the auto-operator until R is pressed."""
        c, now = self.ctrl, time.monotonic()
        if self._auto_paused or link["state"] != "connected":
            return
        if link["epoch"] != self._auto_epoch:
            self._auto_epoch, self._auto_epoch_t = link["epoch"], now
        if c.state == DISARMED and c.collector is None:
            if now - self._auto_epoch_t < 0.5 or now < self._auto_next_try or not link["accepted"]:
                return
            self._auto_next_try = now + 2.0                 # retry spacing, so refusals don't flood the log
            if c.calib is None:
                self.log("[teleop] auto: calibrating (keep the hand still and unloaded)")
                c.start_calibration()
            elif c.status().control_valid:
                c.arm(link)

        # A forward-only session needs a second, deliberate movement calibration.
        # Start it once the controller has installed the fresh stationary reference;
        # this avoids requiring C/R/F keyboard input in the normal demo flow.
        motion = getattr(self.robot, "motion", None)
        if (c.state == ARMED and motion is not None and motion.mode == "forward"
                and motion.reference() is c.calib and not motion.forward_axis_ready()
                and not motion.forward_capture_active()):
            ok, message = motion.start_forward_axis_capture()
            if ok:
                self.log(f"[teleop] auto: {message}")

    def do(self, c: str) -> None:
        if c == "calibrate":
            self.ctrl.start_calibration()
        elif c == "arm":
            self._auto_paused = False
            self.ctrl.arm(self.server.status())
        elif c == "disarm":
            if self.auto:
                self._auto_paused = True
                self.log("[teleop] auto paused by manual disarm; press R to arm and resume auto")
            self.ctrl.disarm("operator disarm")
        elif c == "status":
            self.log(self.status_line())
        elif c == "calibrate_forward":
            motion = getattr(self.robot, "motion", None)
            if motion is None:
                self.log("[teleop] forward-axis capture unavailable: wrist motion is disabled")
            else:
                _ok, message = motion.start_forward_axis_capture()
                self.log(f"[teleop] {message}")
        elif c == "quit":
            self.quit = True

    # -- display -------------------------------------------------------------------

    def status_line(self) -> str:
        s = self.ctrl.status()
        L = self.server.status()
        rej = L["malformed"] + L["oversized"] + L["schema_rejected"] + L["adapter_rejected"]
        return (f"[{s.state}] {s.reason or 'driving'} | link {L['state']} ({L['source'] or '-'}) "
                f"{L['rate_hz']:.0f} Hz | frames rx {L['lines']} ok {L['accepted']} rej {rej} "
                f"dup {L['duplicates']} ooo {L['out_of_order']} | usable age "
                f"{'-' if s.usable_age_ms is None else f'{s.usable_age_ms:.0f} ms'} "
                f"control_valid={s.control_valid} calib={'yes' if s.calibrated else 'no'} | "
                f"{self.robot.summary()}" + (f" | normalized {L['normalized']}" if L["normalized"] else ""))

    def _draw_static(self) -> None:
        self.robot.draw_static(LAMP[DISARMED])
        self._shown = None

    def draw_status(self) -> None:
        """State lamp (green ARMED / orange PAUSED / red DISARMED), plus one short text line with
        --gui-text. Only redrawn on change. A PyBullet debug-text item costs ~15-25 ms per GUI step
        (more than half the 30 Hz budget), so by default the full live status goes to the terminal."""
        s = self.ctrl.status()
        self.robot.show_status(s.signals)
        src = self.server.status()["source"] or "no source"
        why = re.sub(r" for \d+ ms", "", s.reason or "telemetry driving gripper")
        if s.calibrating:
            why = "calibrating: hold still, unloaded"
        text = f"{s.state}: {why[:60]}  [{src} via TCP, EXPERIMENTAL]"
        if text == self._shown:
            return
        if self._shown is None or self._shown.split(":")[0] != s.state:
            self.robot.set_lamp(LAMP[s.state])
        if self.gui_text:
            self.robot.set_text(text)
        self._shown = text

    def keyboard(self) -> tuple[float, bool]:
        """Read PyBullet keys (main thread). Returns (z_input, quit)."""
        import pybullet as p
        try:
            ev = p.getKeyboardEvents(physicsClientId=self.robot.client)
        except p.error:
            return 0.0, True
        trig = lambda k: ev.get(k, 0) & p.KEY_WAS_TRIGGERED
        down = lambda k: ev.get(k, 0) & p.KEY_IS_DOWN
        if trig(ord("c")):
            self.do("calibrate")
        if trig(ord("r")):
            self.do("arm")
        if trig(ord("x")):
            self.do("disarm")
        if trig(ord("f")):
            self.do("calibrate_forward")
        z = (1.0 if down(ord("e")) else 0.0) - (1.0 if down(ord("q")) else 0.0)
        return z, False


def _stdin_reader(q: queue.Queue, stop: threading.Event) -> None:
    names = {"c": "calibrate", "r": "arm", "a": "arm", "x": "disarm", "s": "status", "f": "calibrate_forward", "q": "quit"}
    for line in sys.stdin:
        if stop.is_set():
            return
        c = names.get(line.strip().lower()[:1])
        if c:
            q.put(c)


def add_args(sp) -> None:
    sp.add_argument("--source", choices=["tcp"], default="tcp", help="telemetry input (TCP NDJSON)")
    sp.add_argument("--robot", choices=["hand", "arm"], default="hand",
                    help="hand: 3-claw hand showing each pod (default); arm: Panda gripper")
    sp.add_argument("--hand-pods", default="thumb_tip,index_tip,middle_tip", help="hand: one finger per pod")
    sp.add_argument("--no-motion", action="store_true",
                    help="hand: keep the wrist in place (no EXPERIMENTAL IMU motion bursts, tilt only)")
    sp.add_argument("--motion-mode", choices=["free", "forward"], default="free",
                    help="hand: free is bounded IMU bursts; forward learns one direction with F and removes sideways drift")
    sp.add_argument("--strict-schema", action="store_true",
                    help="reject frames that are not exactly v1-shaped (no compat normalization)")
    sp.add_argument("--assume-valid", action="store_true",
                    help="compat: treat records with no 'valid' field as valid when all values are finite")
    sp.add_argument("--host", default=_env("DERMIS_TCP_HOST", DEFAULT_HOST),
                    help="listen address; 0.0.0.0 to accept LAN clients (env DERMIS_TCP_HOST)")
    sp.add_argument("--port", type=int, default=_env("DERMIS_TCP_PORT", DEFAULT_PORT), help="env DERMIS_TCP_PORT")
    sp.add_argument("--max-message-bytes", type=int, default=_env("DERMIS_TCP_MAX_MESSAGE_BYTES", DEFAULT_MAX_MESSAGE_BYTES))
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--gui", action="store_true", default=True)
    g.add_argument("--headless", action="store_true")
    sp.add_argument("--gui-text", action="store_true",
                    help="also draw a state text line in the 3D view (costs GUI frame rate)")
    sp.add_argument("--control-hz", type=int, default=30)
    sp.add_argument("--substeps", type=int, default=8, help="physics substeps per control tick")
    # The production glove is a 10 Hz Wi-Fi source. Its measured p99 host receive interval is
    # about 382 ms, so 250 ms caused false pause/resume cycles even though frames kept arriving.
    sp.add_argument("--watchdog-ms", type=float, default=_env("DERMIS_WATCHDOG_MS", 500.0),
                    help="pause after this much telemetry silence (default 500 ms for the 10 Hz Wi-Fi glove)")
    sp.add_argument("--max-speed", type=float, default=0.05, help="Cartesian speed limit per axis, m/s")
    sp.add_argument("--resume-policy", choices=["auto", "manual"], default="auto",
                    help="after a watchdog/channel pause (connection loss always needs a re-arm)")
    sp.add_argument("--tilt-xy", action="store_true", help="EXPERIMENTAL: index IMU tilt -> bounded XY velocity")
    sp.add_argument("--tilt-z", action="store_true",
                    help="EXPERIMENTAL: index IMU pitch -> bounded Z velocity (arm down/up), roll -> Y")
    sp.add_argument("--grip-pods", default="index_tip,thumb_tip")
    sp.add_argument("--grip-axis", type=int, choices=[0, 1, 2], default=2)
    sp.add_argument("--grip-sign", type=float, default=1.0)
    sp.add_argument("--grip-gain-per-ut", type=float, default=1 / 6000)
    sp.add_argument("--grip-saturation-ut", type=float, default=8000.0)
    sp.add_argument("--source-label", help="override the displayed source label (e.g. GLOVE once verified)")
    sp.add_argument("--record", help="append received schema-valid frames (raw NDJSON) here")
    sp.add_argument("--calibration", help="load a saved teleop calibration JSON (still checked against the stream)")
    sp.add_argument("--calib-dir", default="runs/teleop")
    sp.add_argument("--duration", type=float, help="stop after this many seconds")
    sp.add_argument("--dashboard-port", type=int, default=None,
                    help="also serve the Dermis console on http://127.0.0.1:PORT (live stream, controls, sim feed)")
    sp.add_argument("--dashboard-feed-hz", type=float, default=0.0,
                    help="console camera feed (GUI only, GPU render on the sim thread, ~36 ms per frame "
                         "at 320x200 measured on an M2: costs control ticks; default off)")
    sp.add_argument("--auto", action="store_true",
                    help="development convenience: calibrate and arm automatically (also after reconnects); "
                         "X still disarms and pauses auto until R")


def session_from_args(a, log=print) -> TeleopSession:
    mcfg = MappingConfig(grip_pods=tuple(a.grip_pods.split(",")), grip_axis=a.grip_axis, grip_sign=a.grip_sign,
                         grip_gain_per_ut=a.grip_gain_per_ut, grip_saturation_ut=a.grip_saturation_ut,
                         tilt_xy=a.tilt_xy or a.tilt_z, pitch_axis="z" if a.tilt_z else "x")
    ccfg = ControllerConfig(control_hz=a.control_hz, watchdog_s=a.watchdog_ms / 1000.0, max_speed=a.max_speed,
                            resume_policy=a.resume_policy)
    return TeleopSession(a.host, a.port, gui=not a.headless, control_hz=a.control_hz, substeps=a.substeps,
                         max_message_bytes=a.max_message_bytes, mapping_cfg=mcfg, ctrl_cfg=ccfg,
                         source_label=a.source_label, record_path=a.record, calibration_path=a.calibration,
                         calib_dir=a.calib_dir, gui_text=a.gui_text, auto=a.auto, robot=a.robot,
                         hand_pods=a.hand_pods.split(","), compat=not a.strict_schema,
                         assume_valid=a.assume_valid, motion=not a.no_motion, motion_mode=a.motion_mode, log=log)


def cmd_teleop(a) -> int:
    log = lambda s: print(s, flush=True)
    try:
        sess = session_from_args(a, log)
    except Exception as e:                                   # e.g. no display for the GUI
        log(f"[teleop] could not start the simulator: {e}")
        return 2
    try:
        sess.start()
    except OSError as e:
        log(f"[teleop] {e}")
        sess.robot.close()
        return 2
    if a.auto:
        log("[teleop] AUTO: will calibrate and arm on its own once a sender streams. With --motion-mode forward, "
            "it also starts the one-jab forward-axis capture. X disarms (pauses auto), R re-arms. Ctrl+C quits.")
    else:
        log("[teleop] DISARMED. Keys (click the sim window): C calibrate, R arm, F forward-axis capture, X disarm, E/Q z. "
            "Or type c / r / f / x / s / q + Enter here. Ctrl+C quits.")
    stop = threading.Event()
    if sys.stdin and sys.stdin.isatty():
        threading.Thread(target=_stdin_reader, args=(sess.commands, stop), daemon=True).start()
    console = None
    if getattr(a, "dashboard_port", None):
        from ...dashboard.server import serve_teleop
        try:
            srv, console = serve_teleop(sess, a.dashboard_port, a.dashboard_feed_hz)
            log(f"[teleop] console on http://127.0.0.1:{srv.server_address[1]}  "
                f"(feed {a.dashboard_feed_hz:g} Hz, control {a.control_hz} Hz)")
        except OSError as e:
            log(f"[teleop] console not started: {e} (try another --dashboard-port)")
    period = 1.0 / a.control_hz
    t_start = time.monotonic()
    next_t = t_start
    last_print = 0.0
    try:
        while True:
            z, quit_ = (sess.keyboard() if sess.gui else (0.0, False))
            if quit_ or sess.quit or (a.duration and time.monotonic() - t_start > a.duration):
                break
            t_tick = time.perf_counter()
            row = sess.step(z)
            if console is not None:
                console.provider.service(row, (time.perf_counter() - t_tick) * 1000)
            now = time.monotonic()
            if sess.gui:
                sess.draw_status()
            if now - last_print >= 1.0:
                last_print = now
                log(sess.status_line())
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                sess.overruns += 1
                next_t = time.monotonic()                 # never try to catch up with a burst of ticks
    except KeyboardInterrupt:
        log("[teleop] interrupted")
    finally:
        stop.set()
        log(sess.status_line())
        sess.close()
        log("[teleop] stopped")
    return 0


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(prog="python -m dermis teleop", description=BANNER)
    add_args(_ap)
    sys.exit(cmd_teleop(_ap.parse_args()))
