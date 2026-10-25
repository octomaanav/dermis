"""Display-only PyBullet GUI for fragile_cup.

Physics always runs in the env's own headless (DIRECT) client, exactly as in evaluation; the
viewer is a second, GUI client that only *shows* the state: once per control tick it copies the
robot joint positions and the cup pose, redraws once, and updates the text overlay. Nothing in
the viewer world is simulated (gravity off, motors disabled, cup static), so the GUI can never
change what the policy experiences. Buttons and keys are read from the viewer window.

Load display: a translucent blue fill inside the cup, drawn from the backend's APPLIED FORCE
(fixed scale 0..FILL_FULL_N), attached to the cup's pose. It is a picture of a number, not a fluid.
Status: cup.status.HoldStatus (measured state, with hysteresis); movement and load status are separate.
"""
from __future__ import annotations

import time

import pybullet as p
import pybullet_data

from .env import CupHoldEnv
from .status import HoldStatus, outcome_text

FILL_LEVELS = 10
FILL_FULL_N = 9.5           # fill scale: max configured peak load (8 N) + max disturbance (1.5 N)


class CupViewer:
    # A redraw costs ~80 ms with the default renderer on this macOS (OpenGL over Metal). "--opengl2" was
    # faster but segfaulted with debug text/lines/keyboard polling, so it is not the default.
    DEFAULT_OPTIONS = ""

    def __init__(self, env: CupHoldEnv, controls: bool = True, options: str | None = None):
        self.env = env
        self.client = p.connect(p.GUI, options=self.DEFAULT_OPTIONS if options is None else options)
        self._p = dict(physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), **self._p)
        p.setGravity(0, 0, 0, **self._p)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, **self._p)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1 if controls else 0, **self._p)
        for flag in (p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                     p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, p.COV_ENABLE_SHADOWS):
            p.configureDebugVisualizer(flag, 0, **self._p)          # software-rendered previews cost ~200 ms/frame
        x, y = env.cfg.cup_xy
        p.resetDebugVisualizerCamera(0.62, 60, -14, (x, y, 0.11), **self._p)   # gripper, whole cup and the ground
        p.loadURDF("plane.urdf", **self._p)
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, **self._p)
        self.nj = p.getNumJoints(self.robot, **self._p)
        p.setJointMotorControlArray(self.robot, list(range(self.nj)), p.VELOCITY_CONTROL,
                                    forces=[0.0] * self.nj, **self._p)
        c = env.cfg
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=c.cup_radius, length=c.cup_height,
                                  rgbaColor=(0.95, 0.95, 0.98, 0.9), **self._p)
        self.cup = p.createMultiBody(0, -1, vis, (x, y, c.cup_height / 2), **self._p)
        catch = p.createVisualShape(p.GEOM_CYLINDER, radius=0.09, length=0.002, rgbaColor=(0.95, 0.55, 0.15, 0.55), **self._p)
        p.createMultiBody(0, -1, catch, (x, y, 0.001), **self._p)                  # catch plane: a fall is obvious
        self._fill_h = c.cup_height * 0.9 / FILL_LEVELS
        disk = p.createVisualShape(p.GEOM_CYLINDER, radius=c.cup_radius * 0.82, length=self._fill_h * 0.96,
                                   rgbaColor=(0.2, 0.5, 1.0, 0.45), **self._p)
        self.fill = [p.createMultiBody(0, -1, disk, (0, 0, -5), **self._p) for _ in range(FILL_LEVELS)]
        self._fill_key = None
        self.hold = HoldStatus()
        self._hs = None
        self._last_t = -1.0
        self._texts: dict = {}
        self._shown: dict = {}
        self._arrow = None
        self._arrow_load = None
        self._last_values = 0.0

    def button(self, name: str) -> int:
        return p.addUserDebugParameter(name, 1, 0, 0, **self._p)

    def read(self, b: int) -> float:
        return p.readUserDebugParameter(b, **self._p)

    def keys(self) -> tuple[set, set]:
        names = {ord(c): c for c in "wsprnctax123456789 "}
        names.update({p.B3G_UP_ARROW: "up", p.B3G_DOWN_ARROW: "down"})
        ev = p.getKeyboardEvents(**self._p)
        down = {n for k, n in names.items() if ev.get(k, 0) & p.KEY_IS_DOWN}
        pressed = {n for k, n in names.items() if ev.get(k, 0) & p.KEY_WAS_TRIGGERED}
        return down, pressed

    def connected(self) -> bool:
        return p.isConnected(self.client)

    def sync(self, env: CupHoldEnv | None = None) -> None:
        """Copy the pose into the window. PyBullet redraws on the first GUI call after a pause, so callers
        batch all GUI calls of one display frame back-to-back (see live.run_gui)."""
        env = env or self.env
        qs = [[p.getJointState(env.robot, j, physicsClientId=env.client)[0]] for j in range(self.nj)]
        p.resetJointStatesMultiDof(self.robot, list(range(self.nj)), qs, **self._p)
        pos, orn = p.getBasePositionAndOrientation(env.cup, physicsClientId=env.client)
        p.resetBasePositionAndOrientation(self.cup, pos, orn, **self._p)
        self._draw_load(env, pos)                          # the pose resets above already redraw the window
        self._draw_fill(env, pos, orn)

    def _draw_fill(self, env, pos, orn) -> None:
        """Stacked translucent disks inside the cup: level = applied load / FILL_FULL_N (visual only)."""
        k = int(round(FILL_LEVELS * min(1.0, max(0.0, env.load_n) / FILL_FULL_N)))
        key = (k, tuple(round(v, 3) for v in pos), tuple(round(v, 3) for v in orn))
        if key == self._fill_key:
            return
        self._fill_key = key
        R = p.getMatrixFromQuaternion(orn)
        R = [R[0:3], R[3:6], R[6:9]]
        H = env.cfg.cup_height
        for i, body in enumerate(self.fill):
            if i < k:
                z = -H / 2 + 0.004 + (i + 0.5) * self._fill_h
                w = [pos[r] + R[r][2] * z for r in range(3)]
                p.resetBasePositionAndOrientation(body, w, orn, **self._p)
            else:
                p.resetBasePositionAndOrientation(body, (0, 0, -5), (0, 0, 0, 1), **self._p)

    def _draw_load(self, env, pos) -> None:
        """Downward arrow beside the cup, length proportional to the applied load proxy.
        Redrawn only when the load changed by > 0.1 N or the cup moved (each redraw costs a frame)."""
        key = (round(env.load_n, 1), round(pos[2], 3))
        if key == self._arrow_load:
            return
        self._arrow_load = key
        L = 0.012 * env.load_n
        start = (pos[0], pos[1] - env.cfg.cup_radius - 0.01, pos[2])
        kw = dict(lineColorRGB=(0.85, 0.2, 0.1), lineWidth=4, **self._p)
        if self._arrow is not None:
            kw["replaceItemUniqueId"] = self._arrow
        self._arrow = p.addUserDebugLine(start, (start[0], start[1], start[2] - max(L, 1e-4)), **kw)

    def text(self, key: str, text: str, row: int, color=(0.1, 0.1, 0.1)) -> None:
        """Draw only when the content changed: one text redraw costs ~80 ms in PyBullet's GUI."""
        if self._shown.get(key) == (text, color):
            return
        self._shown[key] = (text, color)
        kw = dict(textColorRGB=color, textSize=1.3, **self._p)
        if key in self._texts:
            kw["replaceItemUniqueId"] = self._texts[key]
        x, y = self.env.cfg.cup_xy
        self._texts[key] = p.addUserDebugText(text or " ", (x, y - 0.12, 0.30 - 0.022 * row), **kw)

    def hold_status(self, env: CupHoldEnv) -> dict:
        """HoldStatus. When a LiveSession drives the loop it sets self.external_status every control tick
        (hysteresis counts ticks, and several ticks can pass per display frame); otherwise (evaluate --gui,
        which syncs every tick) the viewer updates its own instance once per tick."""
        if getattr(self, "external_status", None) is not None:
            return self.external_status
        if env.t < self._last_t or env.stats.ticks == 0:
            self.hold.reset()
        if env.t != self._last_t:
            self._hs = self.hold.update(env)
            self._last_t = env.t
        return self._hs or self.hold.update(env)

    def status_values(self, env: CupHoldEnv) -> str:
        """Overlay line: applied load + ramp, per-finger and total normal load, displacement (also printed to the terminal)."""
        c = env._contact
        hs = self.hold_status(env)
        st = env.schedule_t
        ramp = "load not started (P)" if st is None else (
            f"ramp {max(0.0, env.scenario.schedule.duration - st):4.1f} s left" if st < env.scenario.schedule.duration else "ramp done")
        return (f"Simulated load: {env.load_n:4.1f} N ({ramp}) | fingertip normal per finger L {c['normal'][0]:4.1f} N, "
                f"R {c['normal'][1]:4.1f} N (total {c['normal'].sum():4.1f} N) | cup down vs hand {hs['slip_cm']:4.2f} cm")

    def status(self, env: CupHoldEnv | None = None, force: bool = False) -> None:
        """Two overlay lines. Each text redraw costs a frame (~80-130 ms here), so the numbers line is redrawn at
        most once per second, and the status line immediately when movement/load status or outcome changes
        (otherwise once per second for the hold timer)."""
        env = env or self.env
        if env.scenario is None:
            return
        hs = self.hold_status(env)
        now = time.monotonic()
        key = (hs["movement"], hs["load_status"], hs["outcome"])
        changed = key != getattr(self, "_status_key", None)
        due = now - self._last_values >= 1.0
        if not (force or changed or due or env.done):
            return
        self._status_key, self._last_values = key, now
        self.text("values", self.status_values(env), 2)
        req = f"{hs['hold_s']:.0f} / {hs['required_s']:.1f} s" if hs["required_s"] else f"{hs['hold_s']:.0f} s (no load yet)"
        load = hs["load_status"] + (f" ({hs['damage_ns']:.0f}/{hs['budget_ns']:g} N*s)" if hs["damage_ns"] is not None else "")
        line = (f"{hs['movement']}   |   load: {load}   |   hold {req}" if not env.done
                else f"{outcome_text(hs['outcome'])}   |   {hs['movement']}   |   load: {load}")
        bad = hs["movement"] == "DROPPED" or hs["load_status"] == "EXCESSIVE LOAD" or (env.done and not env.stats.success)
        col = (0.1, 0.55, 0.2) if (env.done and env.stats.success) or (not bad and hs["movement"] == "HOLDING") else \
              (0.8, 0.5, 0.0) if not bad else (0.75, 0.1, 0.1)
        self.text("outcome", line, 3, col)
        limits = f"{env.cfg.break_force_n:g} N per finger" + (
            f", {env.cfg.sustained_safe_n:g} N sustained" if "sustained" in env.cfg.damage_model else "")
        self.text("limits", f"limits are SIMULATED benchmark thresholds ({limits}); the cup is rigid and does not deform", 5,
                  (0.45, 0.45, 0.45))

    def close(self) -> None:
        if p.isConnected(self.client):
            p.disconnect(self.client)

