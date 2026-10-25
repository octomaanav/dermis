"""Real-time pick-up: a judge wearing the glove grasps a real cup on a real table, and the simulated
three-finger hand picks up a simulated cup from a simulated table.

The glove cannot sense hand position (no position sensor; double-integrating the IMU is not done), so the
LIFT is triggered by the grasp, not tracked:
    OPEN -> grasp detected (thumb and index-or-middle above on_n for on_frames) -> CLOSING -> LIFTING to
    lift_m -> HELD; grasp released (thumb or both fingers below off_n for off_frames) -> LOWERING -> OPENING.
What IS live: the grip. Each pod drives its own digit (thumb_tip -> thumb, index_tip -> index,
middle_tip -> middle; effort = grip_gain * force), every control tick while CLOSING / LIFTING / HELD.
The simulated cup is held only by friction from those efforts, so a weak real grip drops it and a crushing
one passes the SIMULATED crack threshold. thenar_apb is shown but drives nothing (the thenar pad is fixed).
While LOWERING the efforts are frozen at the strongest grip of the last 0.3 s before the release, because
the judge lets go of the real cup only after it is back on the real table.

Physics: the hand base is dynamic and carried by one force-limited fixed constraint (an arm stand-in); the
cup is a free rigid body resting on the table. Nothing attaches or teleports the cup; after a drop or crack
the scene is reset explicitly (event 'reset').
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import pybullet as p
import pybullet_data

from . import hand3
from .env import THUMB_TORQUE_SCALE

DIGIT_POD = {"thumb_tip": "thumb_tip", "index_tip": "index_tip", "middle_tip": "middle_tip"}


@dataclass
class PickupConfig:
    control_hz: int = 50                 # control clock; acts on the latest glove frame (firmware: 10 Hz)
    substeps: int = 5                    # physics at 250 Hz
    table_top: float = 0.40
    cup_radius: float = 0.033
    cup_height: float = 0.10
    cup_mass: float = 0.25
    cup_friction: float = 0.55
    lift_m: float = 0.10
    close_s: float = 0.4
    lift_s: float = 1.2
    lower_s: float = 1.0
    open_s: float = 0.6
    grip_gain: float = 0.012             # N*m of digit effort per N of glove force (provisional mapping)
    effort_min: float = 0.005
    effort_max: float = 0.4
    open_effort: float = 0.05
    on_n: float = 1.0
    off_n: float = 0.5
    on_frames: int = 3
    off_frames: int = 5
    crack_force_n: float | None = 16.0   # SIMULATED threshold; None disables
    drop_m: float = 0.03
    reset_after_s: float = 1.5

    def to_dict(self):
        return asdict(self)


class PickupSim:
    """Table, cup and carried hand in one DIRECT physics client."""

    def __init__(self, cfg: PickupConfig | None = None):
        self.cfg = cfg or PickupConfig()
        self.client = p.connect(p.DIRECT)
        self._p = dict(physicsClientId=self.client)
        self.dt = 1.0 / (self.cfg.control_hz * self.cfg.substeps)
        self.reset()

    def close(self):
        if self.client is not None:
            p.disconnect(**self._p)
            self.client = None

    def reset(self):
        P, c = self._p, self.cfg
        p.resetSimulation(**P)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), **P)
        p.setGravity(0, 0, -9.81, **P)
        p.setTimeStep(self.dt, **P)
        p.loadURDF("plane.urdf", **P)
        self.table_half = (0.30, 0.25, 0.02)
        tc = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.table_half, **P)
        self.table = p.createMultiBody(0, tc, -1, (0, 0.05, c.table_top - self.table_half[2]), **P)
        p.changeDynamics(self.table, -1, lateralFriction=0.6, **P)
        self.cup_y = hand3.PALM_HALF[1] + 2 * hand3.THENAR_HALF[1] + c.cup_radius + 0.001
        self.cup_z0 = c.table_top + c.cup_height / 2
        cc = p.createCollisionShape(p.GEOM_CYLINDER, radius=c.cup_radius, height=c.cup_height, **P)
        self.cup = p.createMultiBody(c.cup_mass, cc, -1, (0, self.cup_y, self.cup_z0), **P)
        p.changeDynamics(self.cup, -1, lateralFriction=c.cup_friction, **P)
        self.home = np.array([0.0, 0.0, self.cup_z0 + 0.008])          # palm just clear of the table
        self.hand, self.joints, self.limits = hand3.build(self.client, tuple(self.home), visual=False, base_mass=0.5)
        self.carrier = p.createConstraint(self.hand, -1, -1, -1, p.JOINT_FIXED, (0, 0, 0), (0, 0, 0), tuple(self.home), **P)
        p.changeConstraint(self.carrier, tuple(self.home), maxForce=400, **P)
        self.lift = 0.0
        self.peak_normal = {k: 0.0 for k in hand3.PODS}
        self.contact = {k: 0.0 for k in hand3.PODS}
        self.cracked = False
        for _ in range(int(0.3 / self.dt)):                              # settle the cup on the table
            self._motors(None)
            p.stepSimulation(**P)

    def _motors(self, efforts: dict | None):
        for i, (j, lim) in enumerate(zip(self.joints, self.limits)):
            digit = hand3.DIGITS[i // 3]
            scale = THUMB_TORQUE_SCALE if digit == "thumb_tip" else 1.0
            if efforts is None:                                            # open the hand
                p.setJointMotorControl2(self.hand, j, p.POSITION_CONTROL, targetPosition=0.0,
                                        force=self.cfg.open_effort * scale, maxVelocity=hand3.MAX_JOINT_SPEED, **self._p)
            else:
                p.setJointMotorControl2(self.hand, j, p.POSITION_CONTROL, targetPosition=lim, force=efforts[digit] * scale,
                                        maxVelocity=hand3.MAX_JOINT_SPEED, **self._p)

    def _contact(self) -> dict:
        out = {k: 0.0 for k in hand3.PODS}
        for cp in p.getContactPoints(self.hand, self.cup, **self._p):
            d = hand3.digit_of_link(cp[3])
            if d:
                out[d] += cp[9]
        return out

    def step(self, efforts: dict | None, lift: float):
        """One control tick: digit efforts (N*m, None = open) and lift fraction 0..1 of lift_m."""
        c, P = self.cfg, self._p
        self.lift = float(np.clip(lift, 0.0, 1.0))
        z = c.lift_m * self.lift * self.lift * (3 - 2 * self.lift)        # smoothstep: no velocity jump at either end
        p.changeConstraint(self.carrier, tuple(self.home + (0, 0, z)), maxForce=400, **P)
        for _ in range(c.substeps):
            self._motors(efforts)
            p.stepSimulation(**P)
            con = self._contact()
            for k in hand3.PODS:
                self.peak_normal[k] = max(self.peak_normal[k], con[k])
            if c.crack_force_n is not None and max(con.values()) > c.crack_force_n:
                self.cracked = True
        self.contact = con

    def cup_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.cup, **self._p)[0])

    def hand_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.hand, **self._p)[0])

    def cup_lag_m(self) -> float:
        """How far the cup has fallen behind the hand, relative to where it sat when grasped (0 = moving with it)."""
        return float((self.hand_pos()[2] - self.home[2]) - (self.cup_pos()[2] - self.cup_z0))

    def state(self) -> dict:
        P = self._p
        pos, orn = p.getBasePositionAndOrientation(self.cup, **P)
        return {"hand_pos": [round(v, 5) for v in self.hand_pos()], "joints": [round(p.getJointState(self.hand, j, **P)[0], 5) for j in self.joints],
                "cup_pos": [round(v, 5) for v in pos], "cup_quat": [round(v, 5) for v in orn], "lift": round(self.lift, 4),
                "cup_height_cm": round((pos[2] - self.cup_z0) * 100, 2), "contact_n": {k: round(v, 3) for k, v in self.contact.items()}}


class GripTrigger:
    """Grasp-triggered pick-up state machine. update() takes per-pod glove forces (N; None = pod invalid,
    forces=None = stream stale) and returns (digit efforts or None, lift fraction)."""

    def __init__(self, cfg: PickupConfig):
        self.cfg = cfg
        self.events = []                                     # kept across resets
        self.reset()

    def reset(self):
        self.state, self.t_state, self.lift, self.on, self.off = "OPEN", 0.0, 0.0, 0, 0
        self.frozen = None
        self.recent = deque(maxlen=max(1, int(0.3 * self.cfg.control_hz)))
        self.efforts = None

    def _go(self, s: str, t: float, **kw):
        self.state, self.t_state = s, t
        self.events.append({"t": round(t, 3), "event": s, **kw})

    def live_efforts(self, forces: dict) -> dict:
        c = self.cfg
        return {d: float(np.clip(c.grip_gain * (forces.get(pod) or 0.0), c.effort_min, c.effort_max)) for d, pod in DIGIT_POD.items()}

    def update(self, forces: dict | None, t: float, outcome: str | None = None) -> tuple[dict | None, float]:
        c, dtc = self.cfg, 1.0 / self.cfg.control_hz
        if forces is None:                                   # stale stream: keep the last command, change nothing
            return self.efforts, self.lift
        f = {k: (v or 0.0) for k, v in forces.items()}
        grasp = f.get("thumb_tip", 0) >= c.on_n and max(f.get("index_tip", 0), f.get("middle_tip", 0)) >= c.on_n
        released = f.get("thumb_tip", 0) < c.off_n or max(f.get("index_tip", 0), f.get("middle_tip", 0)) < c.off_n
        self.on = self.on + 1 if grasp else 0
        self.off = self.off + 1 if released else 0
        el = t - self.t_state
        s = self.state
        if outcome and s not in ("DROPPED", "CRACKED"):
            self._go(outcome, t)
            s, el = self.state, 0.0
        if s == "OPEN":
            if self.on >= c.on_frames:
                self._go("CLOSING", t)
        elif s in ("CLOSING", "LIFTING", "HELD"):
            self.recent.append(self.live_efforts(f))
            if self.off >= c.off_frames:
                self.frozen = {d: max(e[d] for e in self.recent) for d in DIGIT_POD}
                self._go("OPENING" if s == "CLOSING" else "LOWERING", t)
            elif s == "CLOSING" and el >= c.close_s:
                self._go("LIFTING", t)
            elif s == "LIFTING":
                self.lift = min(1.0, self.lift + dtc / c.lift_s)
                if self.lift >= 1.0:
                    self._go("HELD", t)
        elif s == "LOWERING":
            self.lift = max(0.0, self.lift - dtc / c.lower_s)
            if self.lift <= 0.0:
                self._go("OPENING", t)
        elif s == "OPENING":
            if el >= c.open_s:
                self._go("OPEN", t)
        elif s in ("DROPPED", "CRACKED"):
            self.lift = max(0.0, self.lift - dtc / c.lower_s)
            if self.off < c.off_frames:                       # wait for the judge to let go, then the reset delay
                self.t_state = t
            elif el >= c.reset_after_s and self.lift <= 0.0:
                self._go("RESET", t)
        s = self.state
        if s in ("CLOSING", "LIFTING", "HELD"):
            # while a release is being confirmed, keep the recent grip instead of dropping to the live value
            self.efforts = ({d: max(e[d] for e in self.recent) for d in DIGIT_POD} if self.off > 0 and self.recent
                            else self.live_efforts(f))
        elif s == "LOWERING":
            self.efforts = self.frozen
        else:
            self.efforts = None
        return self.efforts, self.lift


class PickupSession:
    """Glove forces in -> state machine -> physics, one control tick per glove frame."""

    def __init__(self, cfg: PickupConfig | None = None):
        self.cfg = cfg or PickupConfig()
        self.sim = PickupSim(self.cfg)
        self.trig = GripTrigger(self.cfg)
        self.t = 0.0
        self.resets = 0

    def close(self):
        self.sim.close()

    def tick(self, forces: dict | None) -> dict:
        sim, trig = self.sim, self.trig
        outcome = None
        if trig.state in ("CLOSING", "LIFTING", "HELD", "LOWERING"):
            if sim.cracked:
                outcome = "CRACKED"
            elif sim.lift > 0.2 and sim.cup_lag_m() > self.cfg.drop_m:
                outcome = "DROPPED"
        efforts, lift = trig.update(forces, self.t, outcome)
        if trig.state == "RESET":
            sim.reset()
            trig.reset()
            self.resets += 1
            trig.events.append({"t": round(self.t, 3), "event": "reset"})
            efforts, lift = None, 0.0
        sim.step(efforts, lift)
        self.t += 1.0 / self.cfg.control_hz
        return {"t": round(self.t, 3), "state": trig.state, "efforts": efforts, "forces": forces, **sim.state()}
