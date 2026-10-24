"""Display-only PyBullet GUI for fragile_cup.

Physics always runs in the env's own headless (DIRECT) client, exactly as in evaluation; the
viewer is a second, GUI client that only *shows* the state: once per control tick it copies the
robot joint positions and the cup pose, redraws once, and updates the text overlay. Nothing in
the viewer world is simulated (gravity off, motors disabled, cup static), so the GUI can never
change what the policy experiences. Buttons and keys are read from the viewer window.
"""
from __future__ import annotations

import time

import pybullet as p
import pybullet_data

from .env import CupHoldEnv


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
        p.resetDebugVisualizerCamera(0.45, 70, -15, (x, y, 0.12), **self._p)
        p.loadURDF("plane.urdf", **self._p)
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, **self._p)
        self.nj = p.getNumJoints(self.robot, **self._p)
        p.setJointMotorControlArray(self.robot, list(range(self.nj)), p.VELOCITY_CONTROL,
                                    forces=[0.0] * self.nj, **self._p)
        c = env.cfg
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=c.cup_radius, length=c.cup_height,
                                  rgbaColor=(0.95, 0.95, 0.98, 0.9), **self._p)
        self.cup = p.createMultiBody(0, -1, vis, (x, y, c.cup_height / 2), **self._p)
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

    def status_values(self, env: CupHoldEnv) -> str:
        c, st = env._contact, env.stats
        slip = float(env._grasp_rel[2] - (env._cup_pos() - env._ee_pos())[2])
        dmg = f" | damage {st.damage_ns:4.1f}/{env.cfg.damage_budget_ns:g} N*s" if "sustained" in env.cfg.damage_model else ""
        return (f"load (proxy) {env.load_n:4.1f} N | fingertip normal {c['normal'][0]:4.1f}/{c['normal'][1]:4.1f} N | "
                f"shear_z {c['shear_z'][0]:4.1f}/{c['shear_z'][1]:4.1f} N | slip {slip * 1000:4.1f} mm | "
                f"effort {env.effort:4.1f} N{dmg} | {env.phase()}")

    def status(self, env: CupHoldEnv | None = None, force: bool = False) -> None:
        """Live numbers at 2 Hz (one combined line), outcome line whenever it changes."""
        env = env or self.env
        if env.scenario is None:
            return
        now = time.monotonic()
        if force or env.done or now - self._last_values >= 0.5:
            self._last_values = now
            self.text("values", self.status_values(env), 2)
        st = env.stats
        out = st.termination or "running"
        col = (0.1, 0.55, 0.2) if out == "success" else (0.75, 0.1, 0.1) if st.termination else (0.1, 0.1, 0.1)
        limits = f"{env.cfg.break_force_n:g} N per finger" + (
            f", {env.cfg.sustained_safe_n:g} N sustained" if "sustained" in env.cfg.damage_model else "")
        self.text("outcome", f"outcome: {out}   (damage = SIMULATED threshold: {limits})", 3, col)

    def close(self) -> None:
        if p.isConnected(self.client):
            p.disconnect(self.client)

