"""Human demonstrations: KEYBOARD (GUI) or GLOVE (experimental mapping; GUI or headless).

Keyboard controls (click the PyBullet window first):
    W/S or Up/Down      +x / -x            A/D or Left/Right   +y / -y
    E / Q               up / down          SPACE               toggle gripper closed/open
    [ / ]               grip effort -/+ 2 N
    N                   finish episode: save it and start the next scenario
    X                   discard the current episode and start the next scenario
    ESC                 save nothing further and quit

Every tick records the observation shown to the operator and the action actually issued.
Episodes end themselves on success, timeout or object out of bounds and are then saved.
"""
from __future__ import annotations

import time

import numpy as np

from ..data.episodes import EpisodeRecorder, EpisodeStore
from ..sim.env import MAX_EFFORT, MAX_OPENING, MIN_EFFORT, GraspLiftPlaceEnv, TaskConfig, clip_action
from ..sim.rollout import run_episode

SPEED = 0.10                    # m/s per held key


def keys_to_velocity(down: set[str], speed: float = SPEED) -> np.ndarray:
    """down: names of held keys ('w', 'up', 'e', ...)."""
    v = np.zeros(3, np.float32)
    for names, axis, sign in ((("w", "up"), 0, 1), (("s", "down"), 0, -1), (("a", "left"), 1, 1),
                              (("d", "right"), 1, -1), (("e",), 2, 1), (("q",), 2, -1)):
        if any(n in down for n in names):
            v[axis] += sign * speed
    return v


class KeyboardTeleop:
    label = "KEYBOARD"

    def __init__(self, effort: float = 15.0):
        self.closed, self.effort = False, effort
        self.command: str | None = None          # "next" | "discard" | "quit"

    def update(self, down: set[str], pressed: set[str]) -> np.ndarray:
        if " " in pressed:
            self.closed = not self.closed
        if "[" in pressed:
            self.effort = max(MIN_EFFORT, self.effort - 2)
        if "]" in pressed:
            self.effort = min(MAX_EFFORT, self.effort + 2)
        for key, cmd in (("n", "next"), ("x", "discard"), ("esc", "quit")):
            if key in pressed:
                self.command = cmd
        v = keys_to_velocity(down)
        return np.array([*v, 0.0 if self.closed else MAX_OPENING, self.effort], np.float32)


def _poll_keys(env: GraspLiftPlaceEnv) -> tuple[set, set]:
    import pybullet as p
    names = {ord(c): c for c in "wsadeqnx[] "}
    names.update({p.B3G_UP_ARROW: "up", p.B3G_DOWN_ARROW: "down", p.B3G_LEFT_ARROW: "left",
                  p.B3G_RIGHT_ARROW: "right", 27: "esc"})
    ev = p.getKeyboardEvents(physicsClientId=env.client)
    down = {n for k, n in names.items() if ev.get(k, 0) & p.KEY_IS_DOWN}
    pressed = {n for k, n in names.items() if ev.get(k, 0) & p.KEY_WAS_TRIGGERED}
    return down, pressed


def run_demo(a) -> int:
    source = a.demo_source
    if source == "keyboard" and a.headless:
        print("keyboard demos need the GUI window")
        return 2
    store = EpisodeStore(a.out)
    cfg = TaskConfig()
    env = GraspLiftPlaceEnv(cfg, gui=not a.headless)
    seed = a.seed_start if a.seed_start is not None else int(time.time()) % 900_000   # stays < eval seed ranges
    runner = mapping = None
    extra = {}
    if source == "glove":
        runner, mapping, extra = _open_glove(a)
        if runner is None:
            env.close()
            return 2
    saved = 0
    try:
        while a.max_episodes is None or saved < a.max_episodes:
            kb = KeyboardTeleop()
            label = source.upper() + (f" [{runner.label} telemetry, EXPERIMENTAL mapping]" if runner else "")
            env.set_label(label)
            print(f"[{source.upper()}] episode seed {seed}: {label}")

            def controller(obs, e):
                down, pressed = _poll_keys(e) if e.gui else (set(), set())
                kb_action = kb.update(down, pressed)
                if kb.command in ("discard", "quit"):
                    return None, None
                if kb.command == "next":
                    return None, None
                if mapping is None:
                    return kb_action, None, "KEYBOARD"
                act, status = mapping(runner.latest(), runner.source_stale())
                if runner.error:
                    raise runner.error
                act = act.copy()
                act[2] += kb_action[2]                      # z from E/Q: the experimental mapping has none
                return act, None, status

            factory = lambda e: EpisodeRecorder(source, e.cfg.to_dict(), e.scenario.to_dict(), e.cfg.dt,
                                                extra={**extra, "operator_note": None})
            outcome, ep = run_episode(env, seed, controller, factory, realtime=True)
            if kb.command == "discard":
                print("  discarded")
            elif len(ep):
                if runner is not None and runner.latest() is not None:
                    ep.meta["telemetry_config_id"] = runner.latest().frame.config_id
                    ep.meta["telemetry_device_id"] = runner.latest().frame.device_id
                path = store.save(ep)
                saved += 1
                print(f"  saved {path}: {ep.meta['termination']} ({len(ep)} ticks)")
            if kb.command == "quit":
                break
            seed += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    print(f"saved {saved} episode(s) to {a.out}")
    return 0


def _open_glove(a):
    from ..telemetry.calibration import Calibration
    from ..telemetry.runner import TelemetryRunner
    from ..telemetry.sources import open_source
    from .mapping import ExperimentalTiltGripMapping
    if not a.experimental_glove_teleop:
        print("GLOVE teleoperation uses a provisional, unvalidated mapping. Re-run with "
              "--experimental-glove-teleop to enable it deliberately.")
        return None, None, {}
    if not a.calibration:
        print("GLOVE teleoperation needs --calibration (capture one with `python -m dermis calibrate`)")
        return None, None, {}
    cal = Calibration.load(a.calibration)
    src = open_source(a.glove_input, path=a.glove_path, port=a.glove_port, serial_port=a.serial_port, wire=a.wire,
                      loop=True)
    runner = TelemetryRunner(src).start()
    snap = runner.wait_for_frame(3.0)
    mapping = ExperimentalTiltGripMapping(cal, enabled=True)
    if snap is not None:
        mapping.capture_rest(snap)
    extra = {"calibration_id": cal.calibration_id, "telemetry_source": runner.label,
             "teleop_mapping": "ExperimentalTiltGripMapping (EXPERIMENTAL)"}
    return runner, mapping, extra
