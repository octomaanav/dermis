#!/usr/bin/env python3
"""PyBullet scene for the foreign-body extraction task: a Franka Panda above a "patient"
block with a hole and a small red "tumor" resting in it.

    python sim_env.py                     # mock_stream.py (decoded_frame.json) drives the pinch
    python sim_env.py --synthetic         # a fake hand that pinches and releases
    python sim_env.py --keyboard-grip     # no glove: hold SPACE to close the gripper
    python sim_env.py --port /dev/cu.usbmodem1101 [--wire]   # real glove (see glove_stream.py)

Controls (click the PyBullet window first):
    W / S                move gripper +x / -x (away from / toward the robot base)
    A / D                move gripper +y / -y
    Shift / Ctrl         move gripper up / down (E / Q also work)
    Arrow keys           same as WASD
    SPACE                hold to close the gripper (overrides the glove)
    C                    calibrate glove: current hand pose becomes "open"
    R                    reset the scene
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time

import pybullet as p
import pybullet_data

import mock_stream
from glove_stream import GloveStream, add_glove_args

CONTROL_HZ = 30
PHYSICS_HZ = 240
SUBSTEPS = PHYSICS_HZ // CONTROL_HZ

# Panda (pybullet_data/franka_panda/panda.urdf)
ARM_JOINTS = list(range(7))
FINGER_JOINTS = (9, 10)
EE_LINK = 11                                   # panda_grasptarget, between the fingertips
FINGER_OPEN = 0.04                             # metres per finger
FINGER_FORCE = 10.0                            # N; much higher and the tapered pads squeeze the tumor out
HOME_JOINTS = (0.0, -0.3, 0.0, -2.2, 0.0, 1.9, math.pi / 4)
EE_DOWN = p.getQuaternionFromEuler((math.pi, 0.0, 0.0))
MOVE_SPEED = 0.15                              # m/s while an arrow key is held
WORKSPACE = ((0.30, 0.75), (-0.30, 0.30), (0.02, 0.50))
MOVE_KEYS = {                                  # key -> (axis, direction)
    ord("w"): (0, +1), ord("s"): (0, -1), ord("a"): (1, +1), ord("d"): (1, -1),
    p.B3G_SHIFT: (2, +1), p.B3G_CONTROL: (2, -1), ord("e"): (2, +1), ord("q"): (2, -1),
    p.B3G_UP_ARROW: (0, +1), p.B3G_DOWN_ARROW: (0, -1), p.B3G_LEFT_ARROW: (1, +1), p.B3G_RIGHT_ARROW: (1, -1),
}

# Patient block with a square hole, and the tumor inside it
PATIENT_POS = (0.55, 0.0, 0.0)                 # nominal; reset() moves it when randomizing
PATIENT_JITTER = (0.05, 0.08)                  # ± x, y of the patient block on the table
TUMOR_JITTER = (0.025, 0.008)                  # ± x, y of the tumor in the hole (open fingers span ±0.071 in y)
PATIENT_HALF = 0.12                            # outer half-width
HOLE_HALF = 0.085                              # hole half-width
FLOOR_T = 0.015                                # thickness under the hole
HOLE_DEPTH = 0.045
TUMOR_HALF = 0.0125
TUMOR_MASS = 0.01
START_TARGET = (PATIENT_POS[0], PATIENT_POS[1], 0.20)


class SurgerySim:
    def __init__(self, gui: bool = True, randomize: bool = True, seed: int | None = None):
        self.randomize = randomize
        self.rng = random.Random(seed)
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / PHYSICS_HZ)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)   # free up letter keys
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(0.7, 50, -35, (PATIENT_POS[0], 0.0, 0.05))

        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        for j in FINGER_JOINTS:
            p.changeDynamics(self.robot, j, lateralFriction=1.5, spinningFriction=0.01)
        self.patient = self._make_patient()
        self.tumor = self._make_tumor()

        self.target = list(START_TARGET)
        self.grip = 0.0                         # 0 open .. 1 closed
        self.reset()

    # -- scene ---------------------------------------------------------------

    def _make_patient(self) -> int:
        wall_h = HOLE_DEPTH / 2
        wall_z = FLOOR_T + wall_h
        wall_t = (PATIENT_HALF - HOLE_HALF) / 2
        wall_c = HOLE_HALF + wall_t
        half_extents = [
            (PATIENT_HALF, PATIENT_HALF, FLOOR_T / 2),       # floor
            (PATIENT_HALF, wall_t, wall_h),                  # +y wall
            (PATIENT_HALF, wall_t, wall_h),                  # -y wall
            (wall_t, HOLE_HALF, wall_h),                     # +x wall
            (wall_t, HOLE_HALF, wall_h),                     # -x wall
        ]
        positions = [(0, 0, FLOOR_T / 2), (0, wall_c, wall_z), (0, -wall_c, wall_z),
                     (wall_c, 0, wall_z), (-wall_c, 0, wall_z)]
        n = len(half_extents)
        col = p.createCollisionShapeArray([p.GEOM_BOX] * n, halfExtents=half_extents,
                                          collisionFramePositions=positions)
        vis = p.createVisualShapeArray([p.GEOM_BOX] * n, halfExtents=half_extents,
                                       visualFramePositions=positions)
        body = p.createMultiBody(0, col, vis, PATIENT_POS)
        p.changeVisualShape(body, -1, rgbaColor=(0.93, 0.76, 0.66, 1))
        return body

    def _make_tumor(self) -> int:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[TUMOR_HALF] * 3)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[TUMOR_HALF] * 3, rgbaColor=(0.85, 0.1, 0.1, 1))
        body = p.createMultiBody(TUMOR_MASS, col, vis, (PATIENT_POS[0], PATIENT_POS[1], FLOOR_T + TUMOR_HALF + 0.001))
        p.changeDynamics(body, -1, lateralFriction=1.5, spinningFriction=0.01, rollingFriction=0.001)
        return body

    def reset(self) -> None:
        """Arm to the start pose; patient and tumor to a new random spot (or nominal if not randomizing)."""
        for j, q in zip(ARM_JOINTS, HOME_JOINTS):
            p.resetJointState(self.robot, j, q)
        for j in FINGER_JOINTS:
            p.resetJointState(self.robot, j, FINGER_OPEN)
        jitter = (lambda r: self.rng.uniform(-r, r)) if self.randomize else (lambda r: 0.0)
        px, py = PATIENT_POS[0] + jitter(PATIENT_JITTER[0]), PATIENT_POS[1] + jitter(PATIENT_JITTER[1])
        tumor = (px + jitter(TUMOR_JITTER[0]), py + jitter(TUMOR_JITTER[1]), FLOOR_T + TUMOR_HALF + 0.001)
        p.resetBasePositionAndOrientation(self.patient, (px, py, 0.0), (0, 0, 0, 1))
        p.resetBasePositionAndOrientation(self.tumor, tumor, (0, 0, 0, 1))
        p.resetBaseVelocity(self.tumor, (0, 0, 0), (0, 0, 0))
        self.target = list(START_TARGET)
        self.grip = 0.0
        for j, q in zip(ARM_JOINTS, self._ik(self.target)):   # snap to the start pose, no swing
            p.resetJointState(self.robot, j, q)
        for _ in range(PHYSICS_HZ // 4):
            self._apply_controls()
            p.stepSimulation()

    # -- control -------------------------------------------------------------

    def _ik(self, pos) -> list:
        q = p.calculateInverseKinematics(self.robot, EE_LINK, pos, EE_DOWN,
                                         maxNumIterations=100, residualThreshold=1e-4)
        return list(q[:len(ARM_JOINTS)])

    def _apply_controls(self) -> None:
        p.setJointMotorControlArray(self.robot, ARM_JOINTS, p.POSITION_CONTROL,
                                    targetPositions=self._ik(self.target), forces=[240.0] * 7)
        width = FINGER_OPEN * (1.0 - self.grip)
        for j in FINGER_JOINTS:
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=width, force=FINGER_FORCE)

    def set_grip_from_bends(self, bends: dict) -> None:
        """Pinch closure = mean of the index and thumb bend angles (each 0..1)."""
        self.grip = max(0.0, min(1.0, sum(bends.values()) / max(1, len(bends))))

    def read_keys(self) -> dict:
        """Poll the GUI keyboard once per tick, move the target, and return the raw events."""
        keys = p.getKeyboardEvents()
        d = MOVE_SPEED / CONTROL_HZ
        for key, (axis, sign) in MOVE_KEYS.items():
            if keys.get(key, 0) & p.KEY_IS_DOWN:
                self.target[axis] += sign * d
        for i, (lo, hi) in enumerate(WORKSPACE):
            self.target[i] = max(lo, min(hi, self.target[i]))
        return keys

    def step(self) -> None:
        """Advance one control tick (1 / CONTROL_HZ seconds of sim time)."""
        self._apply_controls()
        for _ in range(SUBSTEPS):
            p.stepSimulation()

    # -- observation ---------------------------------------------------------

    def get_state(self) -> dict:
        ee = p.getLinkState(self.robot, EE_LINK, computeForwardKinematics=True)[4]
        tumor, _ = p.getBasePositionAndOrientation(self.tumor)
        fingers = [p.getJointState(self.robot, j)[0] for j in FINGER_JOINTS]
        return {
            "ee_x": ee[0], "ee_y": ee[1], "ee_z": ee[2],
            "tumor_x": tumor[0], "tumor_y": tumor[1], "tumor_z": tumor[2],
            "finger_l": fingers[0], "finger_r": fingers[1],
            # action: what the operator commanded this tick
            "target_x": self.target[0], "target_y": self.target[1], "target_z": self.target[2],
            "grip_cmd": self.grip,
        }

    def close(self) -> None:
        p.disconnect(self.client)


def triggered(keys: dict, key: int) -> bool:
    return bool(keys.get(key, 0) & p.KEY_WAS_TRIGGERED)


def make_arg_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_glove_args(ap)
    ap.add_argument("--synthetic", action="store_true", help="fake hand that pinches and releases")
    ap.add_argument("--keyboard-grip", action="store_true", help="ignore the glove; the gripper stays open unless SPACE is held")
    ap.add_argument("--direct", action="store_true", help="headless (no GUI window)")
    return ap


def open_glove(args: argparse.Namespace):
    """Started GloveStream for the chosen source (default: mock_stream), or None for --keyboard-grip."""
    if args.keyboard_grip:
        return None
    if args.port or args.synthetic:
        return GloveStream.from_args(args).start()
    return mock_stream.open_stream(args.mock).start()


def control_tick(sim: SurgerySim, glove) -> dict:
    """Shared per-tick input handling for sim_env.py and record_demo.py. Returns the key events."""
    keys = sim.read_keys()
    if glove is None:
        sim.grip = 0.0
    else:
        if glove.error:
            raise glove.error
        if triggered(keys, ord("c")):
            glove.calibrate()
            print("glove calibrated")
        sim.set_grip_from_bends(glove.latest())
    if keys.get(ord(" "), 0) & p.KEY_IS_DOWN:    # SPACE closes the gripper in every mode
        sim.grip = 1.0
    if triggered(keys, ord("r")):
        sim.reset()
    sim.step()
    return keys


def main() -> int:
    args = make_arg_parser(__doc__).parse_args()
    glove = open_glove(args)
    sim = SurgerySim(gui=not args.direct)
    period = 1.0 / CONTROL_HZ
    try:
        while p.isConnected(sim.client):
            t0 = time.monotonic()
            control_tick(sim, glove)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except (KeyboardInterrupt, p.error):
        pass
    finally:
        if p.isConnected(sim.client):
            sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
