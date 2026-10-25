"""glass_hold: a three-finger hand (thumb opposing index + middle, thenar pad on the palm) holds a glass
while water is poured into it. SIMULATED (PyBullet, rigid bodies).

What is modelled, and what is not:
  - Water is added as MASS: the glass body's mass rises over the pour (changeDynamics), so gravity really
    asks more of the grip. There is no fluid simulation; the blue fill in the viewer is drawn from that mass.
  - "Crack" is a SIMULATED threshold: the episode ends as 'crack' when any pod's true contact normal force
    exceeds cfg.crack_force_n. The glass is rigid and is never deformed or broken in the physics.
  - 'drop': the glass falls more than drop_m below where it was when the setup fixture was removed.
  - 'success': neither happens, and the glass is still within success_slip_m of that height at the end.
Setup: the glass is held by a temporary fixed constraint for setup_s while the fingers close at a common
setup effort; the constraint is then removed and stays removed (no attachment, teleport or reset after).
The controller's only action is grip effort (N*m motor torque limit per joint; the thumb gets
THUMB_TORQUE_SCALE times that, because it opposes two fingers). Physics is identical for every controller.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pybullet as p
import pybullet_data

from . import hand3
from .sensors import SyntheticGlove

THUMB_TORQUE_SCALE = 2.0
EFFORT_LOW, EFFORT_HIGH = 0.005, 0.4


@dataclass
class GlassConfig:
    control_hz: int = 30
    substeps: int = 8
    setup_s: float = 1.5
    setup_effort: float = 0.04
    crack_force_n: float = 16.0
    drop_m: float = 0.03
    success_slip_m: float = 0.015
    hold_after_pour_s: float = 2.0
    glass_height: float = 0.10

    def to_dict(self):
        return asdict(self)


@dataclass
class GlassScenario:
    seed: int
    glass_mass: float
    radius: float
    friction: float
    water_kg: float
    pour_start_s: float
    pour_s: float

    @classmethod
    def sample(cls, seed: int) -> "GlassScenario":
        r = np.random.default_rng(seed)
        return cls(seed, round(float(r.uniform(0.12, 0.18)), 4), round(float(r.uniform(0.031, 0.035)), 4),
                   round(float(r.uniform(0.5, 0.6)), 3), round(float(r.uniform(0.22, 0.35)), 3),
                   round(float(r.uniform(0.8, 1.5)), 2), round(float(r.uniform(4.0, 7.0)), 2))

    def water_at(self, t: float) -> float:
        """Water mass (kg) at time t after the fixture is removed: linear pour, then constant."""
        return self.water_kg * float(np.clip((t - self.pour_start_s) / self.pour_s, 0.0, 1.0))

    def to_dict(self):
        return asdict(self)


class GlassEnv:
    def __init__(self, cfg: GlassConfig | None = None):
        self.cfg = cfg or GlassConfig()
        self.client = p.connect(p.DIRECT)
        self._p = dict(physicsClientId=self.client)
        self.dt = 1.0 / (self.cfg.control_hz * self.cfg.substeps)
        self.glass = None

    def close(self):
        if self.client is not None:
            p.disconnect(**self._p)
            self.client = None

    # -- setup -------------------------------------------------------------------------------------
    def reset(self, sc: GlassScenario) -> dict:
        P, c = self._p, self.cfg
        p.resetSimulation(**P)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), **P)
        p.setGravity(0, 0, -9.81, **P)
        p.setTimeStep(self.dt, **P)
        p.loadURDF("plane.urdf", **P)
        self.sc = sc
        self.hand, self.joints, self.limits = hand3.build(self.client, visual=False)
        self.glass_y = hand3.PALM_HALF[1] + 2 * hand3.THENAR_HALF[1] + sc.radius + 0.001
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=sc.radius, height=c.glass_height, **P)
        self.glass = p.createMultiBody(sc.glass_mass, col, -1, (0, self.glass_y, hand3.Z0), **P)
        p.changeDynamics(self.glass, -1, lateralFriction=sc.friction, **P)
        self.glove = SyntheticGlove(sc.seed)
        self.t, self.water, self.effort = 0.0, 0.0, c.setup_effort
        self.termination, self.events, self.ticks = None, [], 0
        self.peak_normal = {k: 0.0 for k in hand3.PODS}
        self.contact = self._contact()
        frame0 = self.glove.make(self.contact)                     # unloaded frame (fingers open) = baseline
        fix = p.createConstraint(self.glass, -1, -1, -1, p.JOINT_FIXED, (0, 0, 0), (0, 0, 0), (0, self.glass_y, hand3.Z0), **P)
        for _ in range(int(c.setup_s / self.dt)):
            self._motors(c.setup_effort)
            p.stepSimulation(**P)
        p.removeConstraint(fix, **P)
        self.z0 = self.glass_pos()[2]
        self.contact = self._contact()
        self.frame = self.glove.make(self.contact)
        self.baseline_frame = frame0
        return self.frame

    @property
    def end_t(self) -> float:
        return self.sc.pour_start_s + self.sc.pour_s + self.cfg.hold_after_pour_s

    @property
    def done(self) -> bool:
        return self.termination is not None

    def glass_pos(self):
        return p.getBasePositionAndOrientation(self.glass, **self._p)[0]

    def drop_m(self) -> float:
        return self.z0 - self.glass_pos()[2]

    # -- step --------------------------------------------------------------------------------------
    def _motors(self, effort: float):
        for i, (j, lim) in enumerate(zip(self.joints, self.limits)):
            scale = THUMB_TORQUE_SCALE if i < 3 else 1.0
            p.setJointMotorControl2(self.hand, j, p.POSITION_CONTROL, targetPosition=lim, force=effort * scale,
                                    maxVelocity=hand3.MAX_JOINT_SPEED, **self._p)

    def _contact(self) -> dict:
        out = {k: {"normal": 0.0, "shear_z": 0.0, "shear_h": 0.0} for k in hand3.PODS}
        out["palm"] = {"normal": 0.0, "shear_z": 0.0, "shear_h": 0.0}
        if self.glass is None:
            return out
        for cp in p.getContactPoints(self.hand, self.glass, **self._p):
            d = hand3.digit_of_link(cp[3]) or "palm"
            f1, f2 = cp[10] * np.array(cp[11]), cp[12] * np.array(cp[13])
            fr = -(f1 + f2)                                          # friction acting on the glass
            o = out[d]
            o["normal"] += cp[9]
            o["shear_z"] += fr[2]
            o["shear_h"] += float(np.hypot(fr[0], fr[1]))
        return out

    def step(self, effort: float) -> dict:
        c, P = self.cfg, self._p
        self.effort = float(np.clip(effort, EFFORT_LOW, EFFORT_HIGH))
        self.water = self.sc.water_at(self.t)
        p.changeDynamics(self.glass, -1, mass=self.sc.glass_mass + self.water, **P)
        for _ in range(c.substeps):
            self._motors(self.effort)
            p.stepSimulation(**P)
            con = self._contact()
            for k in hand3.PODS:
                self.peak_normal[k] = max(self.peak_normal[k], con[k]["normal"])
            if max(con[k]["normal"] for k in hand3.PODS) > c.crack_force_n and self.termination is None:
                self._end("crack", pod=max(hand3.PODS, key=lambda k: con[k]["normal"]))
        self.t += 1.0 / c.control_hz
        self.ticks += 1
        self.contact = con
        self.frame = self.glove.make(con)
        if self.termination is None:
            if self.drop_m() > c.drop_m:
                self._end("drop")
            elif self.t >= self.end_t:
                self._end("success" if self.drop_m() < c.success_slip_m else "excess_slip")
        return self.frame

    def _end(self, term: str, **kw):
        self.termination = term
        self.events.append({"event": "termination", "termination": term, "t": round(self.t, 3), **kw})

    # -- state for traces / viewer ------------------------------------------------------------------
    def state(self) -> dict:
        P = self._p
        pos, orn = p.getBasePositionAndOrientation(self.glass, **P)
        return {"t": round(self.t, 4), "water_kg": round(self.water, 4), "glass_total_kg": round(self.sc.glass_mass + self.water, 4),
                "weight_n": round((self.sc.glass_mass + self.water) * 9.81, 3), "effort_nm": round(self.effort, 4),
                "joints": [round(p.getJointState(self.hand, j, **P)[0], 5) for j in self.joints],
                "glass_pos": [round(v, 5) for v in pos], "glass_quat": [round(v, 5) for v in orn],
                "drop_cm": round(self.drop_m() * 100, 3),
                "contact": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in self.contact.items()},
                "vertical_support_n": round(sum(v["shear_z"] for v in self.contact.values()), 3),
                "termination": self.termination}


def scenario_json(sc: GlassScenario) -> str:
    return json.dumps(sc.to_dict(), sort_keys=True)
