"""Simulated robots the TCP teleop session can drive. Each pairs a mapping + controller with a sim.

    hand   3-claw hand (hand.py): per-pod claw flex + palm tilt. Shows every sensor moving. Default.
    arm    Franka Panda parallel-jaw gripper (sim/env.py): grip from index/thumb, optional tilt velocity.

Interface used by TeleopSession (main thread only):
    client, apply(cmd), measure() -> dict, summary() -> str, draw_static(lamp_color),
    set_lamp(color), set_text(text), show_status(signals), close()
"""
from __future__ import annotations

import numpy as np

from .controller import ControllerConfig, TeleopController
from .hand import HOME, HandConfig, HandController, HandMapping, HandSim
from .motion import MotionEstimator
from .mapping import MagGripTiltMapping, MappingConfig

TELEOP_WORKSPACE = ((0.40, 0.65), (-0.15, 0.15), (0.10, 0.35))
START_EE = (0.52, 0.0, 0.25)
EFFORT_N = 10.0


class _Base:
    lamp_pos = (0.0, 0.0, 0.0)
    client: int

    def _p(self):
        import pybullet as p
        return p

    def draw_static(self, lamp_color) -> None:
        p = self._p()
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.02 if isinstance(self, HandRobot) else 0.035,
                                  rgbaColor=lamp_color, physicsClientId=self.client)
        self._lamp = p.createMultiBody(0, -1, vis, self.lamp_pos, physicsClientId=self.client)
        self._label = None

    def set_lamp(self, color) -> None:
        self._p().changeVisualShape(self._lamp, -1, rgbaColor=color, physicsClientId=self.client)

    def set_text(self, text: str) -> None:
        p = self._p()
        kw = dict(textColorRGB=(0.8, 0.1, 0.1), textSize=1.3, physicsClientId=self.client)
        if self._label is not None:
            kw["replaceItemUniqueId"] = self._label
        self._label = p.addUserDebugText(text, self.text_pos, **kw)

    def show_status(self, signals: dict) -> None:
        pass


class ArmRobot(_Base):
    name = "arm"
    lamp_pos = (0.40, -0.30, 0.42)
    text_pos = (0.35, -0.35, 0.47)

    def __init__(self, gui, control_hz, substeps, ctrl_cfg: ControllerConfig, workspace=TELEOP_WORKSPACE):
        from ...sim.env import GraspLiftPlaceEnv, TaskConfig
        self.workspace = tuple(tuple(map(float, w)) for w in workspace)
        task = TaskConfig(control_hz=control_hz, physics_hz=control_hz * substeps, timeout_s=1e7,
                          max_speed=ctrl_cfg.max_speed, workspace=self.workspace, start_ee=START_EE)
        self.task = task
        self.gui = gui
        self.env = GraspLiftPlaceEnv(task, gui=gui)
        self.env.reset(0)
        self.client = self.env.client

    @staticmethod
    def make_controller(mapping_cfg, ctrl_cfg, calib, calib_dir, log, initial_opening=0.08):
        return TeleopController(MagGripTiltMapping(mapping_cfg or MappingConfig()), ctrl_cfg,
                                initial_opening=initial_opening, calibration=calib, calib_dir=calib_dir, log=log)

    def apply(self, cmd) -> None:
        action = np.array([cmd.v[0], cmd.v[1], cmd.v[2], cmd.opening, EFFORT_N], dtype=np.float32)
        if self.gui:                          # the GUI re-renders inside stepSimulation; render once per tick instead
            p = self._p()
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client)
            self.env.step(action)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client)
        else:
            self.env.step(action)
        if self.env.done:                     # a teleop session is not an episode: never end on success/timeout
            self.env.done = False
            self.env.stats.termination = None

    def measure(self) -> dict:
        return {"ee": self.env._ee_pos().tolist(), "opening": self.env._opening(),
                "target": self.env.target_pos.tolist()}

    def summary(self) -> str:
        ee = self.env._ee_pos()
        return f"opening {self.env._opening() * 1000:.1f} mm ee ({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})"

    def draw_static(self, lamp_color) -> None:
        p = self._p()
        (x0, x1), (y0, y1), (z0, z1) = self.workspace
        kw = dict(lineColorRGB=(0.9, 0.6, 0.1), lineWidth=1.5, physicsClientId=self.client)
        corners = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
        for a in corners:
            for b in corners:
                if sum(u != v for u, v in zip(a, b)) == 1 and a < b:
                    p.addUserDebugLine(a, b, **kw)
        p.resetDebugVisualizerCamera(0.9, 50, -30, (0.52, 0.0, 0.15), physicsClientId=self.client)
        if self.env._label_id is not None:               # the env's own (empty) label text costs frame time
            p.removeUserDebugItem(self.env._label_id, physicsClientId=self.client)
            self.env._label_id = None
        super().draw_static(lamp_color)

    def close(self) -> None:
        self.env.close()


class HandRobot(_Base):
    name = "hand"
    lamp_pos = (HOME[0] - 0.12, HOME[1], HOME[2] + 0.10)     # behind and above the hand's home
    text_pos = (HOME[0] - 0.12, HOME[1] - 0.05, HOME[2] + 0.15)

    def __init__(self, gui, control_hz, substeps, ctrl_cfg: ControllerConfig, pods=None, motion=True):
        self.gui = gui
        self.hand_cfg = HandConfig(grip_pods=tuple(pods)) if pods else HandConfig()
        self.sim = HandSim(gui, control_hz * substeps, control_hz, self.hand_cfg.grip_pods)
        self.client = self.sim.client
        self.motion = MotionEstimator() if motion else None

    def make_controller(self, mapping_cfg, ctrl_cfg, calib, calib_dir, log, initial_opening=0.0):
        return HandController(HandMapping(self.hand_cfg), ctrl_cfg, calibration=calib, calib_dir=calib_dir, log=log,
                              motion=self.motion)

    def apply(self, cmd) -> None:
        self.sim.apply(cmd)

    def measure(self) -> dict:
        return self.sim.measure()

    def summary(self) -> str:
        m = self.sim.measure()
        q = np.degrees(m["joints"][0::3])
        fingers = " ".join(f"{pod.split('_')[0]} {a:+.0f}°" for pod, a in zip(self.sim.pods, q))
        r, pch = np.degrees(m["palm_rp"])
        dx, dy, dz = (100 * np.array(m["offset"])).tolist()
        return f"fingers {fingers} | tilt roll {r:+.0f}° pitch {pch:+.0f}° | hand offset ({dx:+.0f},{dy:+.0f},{dz:+.0f}) cm"

    def show_status(self, signals: dict) -> None:
        self.sim.show_status(signals.get("status", {}))

    def close(self) -> None:
        self.sim.close()


def make_robot(kind: str, gui: bool, control_hz: int, substeps: int, ctrl_cfg: ControllerConfig, hand_pods=None,
               motion: bool = True):
    if kind == "arm":
        return ArmRobot(gui, control_hz, substeps, ctrl_cfg)
    if kind == "hand":
        return HandRobot(gui, control_hz, substeps, ctrl_cfg, hand_pods, motion)
    raise ValueError(f"unknown robot {kind!r}; use hand or arm")
