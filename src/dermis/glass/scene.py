"""Display-only scene for the glass demo: one copy of the hand + glass + water fill per lane, each mirroring
a separate physics GlassEnv. Nothing here feeds back into physics.

The water fill is a picture of the SIMULATED water mass (stacked blue discs, height = volume / glass cross
section at 1 kg/L); it is not a fluid. A cracked glass (simulated threshold) is tinted red; it is not
shown breaking because breaking is not modelled. The glass is drawn as base + rim + wall staves
(PyBullet's offscreen renderer has no transparency); the physics body is a solid cylinder.
"""
from __future__ import annotations

import numpy as np
import pybullet as p

from . import hand3

FILL_DISCS = 20
GLASS_RGBA = (0.8, 0.9, 0.97, 1.0)
CRACK_RGBA = (0.9, 0.2, 0.15, 1.0)
WATER_RGBA = (0.2, 0.5, 0.95, 0.8)


class Lane:
    def __init__(self, client: int, offset_x: float, radius: float, height: float, glass_y: float):
        P = dict(physicsClientId=client)
        self.P, self.off = P, np.array([offset_x, 0.0, 0.0])
        self.hand, self.joints, _ = hand3.build(client, base_pos=tuple(self.off + (0, 0, hand3.Z0)), visual=True)
        for j in range(-1, p.getNumJoints(self.hand, **P)):
            p.setCollisionFilterGroupMask(self.hand, j, 0, 0, **P)
        self.r, self.h = radius, height
        self.glass = p.createMultiBody(0, -1, _glass_visual(client, radius, height), tuple(self.off + (0, glass_y, hand3.Z0)), **P)
        p.changeVisualShape(self.glass, -1, rgbaColor=GLASS_RGBA, **P)
        dh = height / FILL_DISCS
        self.dh = dh
        dv = p.createVisualShape(p.GEOM_CYLINDER, radius=radius * 0.92, length=dh * 0.98, rgbaColor=WATER_RGBA, **P)
        self.discs = [p.createMultiBody(0, -1, dv, (0, 0, -5), **P) for _ in range(FILL_DISCS)]
        self.cracked = False

    def sync(self, state: dict):
        P = self.P
        for j, q in zip(self.joints, state["joints"]):
            p.resetJointState(self.hand, j, q, **P)
        pos = np.array(state["glass_pos"]) + self.off
        orn = state["glass_quat"]
        p.resetBasePositionAndOrientation(self.glass, pos.tolist(), orn, **P)
        level = min(1.0, state["water_kg"] / 1000.0 / (np.pi * (self.r * 0.92) ** 2) / self.h)
        n = int(round(level * FILL_DISCS))
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        for i, d in enumerate(self.discs):
            if i < n:
                local = np.array([0, 0, -self.h / 2 + self.dh * (i + 0.5)])
                p.resetBasePositionAndOrientation(d, (pos + R @ local).tolist(), orn, **P)
            else:
                p.resetBasePositionAndOrientation(d, (0, 0, -5), (0, 0, 0, 1), **P)
        if state["termination"] == "crack" and not self.cracked:
            p.changeVisualShape(self.glass, -1, rgbaColor=CRACK_RGBA, **P)
            self.cracked = True


def _glass_visual(client: int, r: float, h: float) -> int:
    """Glass drawn as base + rim + thin wall staves, so the water inside stays visible in renderers
    without transparency (PyBullet's TinyRenderer). Coloured on the body with changeVisualShape."""
    n = 16
    shapes, radii, lens, pos = [p.GEOM_CYLINDER], [r], [0.004], [(0, 0, -h / 2 + 0.002)]
    for k in range(n):
        a = 2 * np.pi * k / n
        shapes += [p.GEOM_CYLINDER, p.GEOM_SPHERE]
        radii += [0.0022, 0.0035]
        lens += [h, 0.0]
        pos += [(r * np.cos(a), r * np.sin(a), 0.0), (r * np.cos(a), r * np.sin(a), h / 2)]
    return p.createVisualShapeArray(shapeTypes=shapes, radii=radii, lengths=lens, visualFramePositions=pos,
                                    physicsClientId=client)


def lane_camera(offset_x: float, w: int, h: int):
    target = (offset_x, 0.05, hand3.Z0 - 0.04)
    view = p.computeViewMatrixFromYawPitchRoll(target, 0.30, yaw=200, pitch=-18, roll=0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(45, w / h, 0.02, 3.0)
    return view, proj
