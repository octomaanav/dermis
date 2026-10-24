"""assist_hand: presentation environment for "Teach an Assistive Hand" (SIMULATED research prototype).

Not a prosthesis, not muscle-controlled, not validated. The hand is the 3-finger hand built by
dermis.server.teleop.hand.HandSim (reused unmodified), but here its finger joints are driven by
FORCE-LIMITED position motors instead of the teleop path's kinematic posing, so grip effort is a
real torque limit and the object is held only through contact.

Grasp: a HOOK GRIP of a mug handle. Palm down, the index and middle fingers curl under a horizontal
handle and trap it against the palm; the mug body hangs beside the hand. This grip was chosen after
testing: with this hand's kinematics the thumb moves along, not around, a cup's axis and cannot
oppose the fingers radially, and a side-on cylinder grasp was squeezed out of the fingers
(docs/assist_demo.md, "Why a mug handle"). The thumb follows the closure command visually and is
capped so it does not touch the mug; it carries no load. The load is borne through contact
(normal forces under the curled fingers, with friction along the handle).

Physics, schedules and thresholds are identical in every control mode:
    - same setup: a temporary fixture constraint holds the mug while the fingers close at the
      setup effort, then is REMOVED (tests assert zero constraints afterwards); settle; grasp check
    - load proxy: the fragile_cup LoadSchedule (external downward force on the mug, not a fluid)
    - damage: per-digit contact normal force > break_force_n is a SIMULATED threshold; the mug is
      rigid and never deforms
Frames/units: world z up, m, s, N, N*m. 30 Hz control, 240 Hz physics, quantities averaged over substeps.

Action (2 floats): closure in [0, 1] (target fraction of each finger's flex range; 0 = open hand),
effort in [EFFORT_MIN, EFFORT_MAX] N*m (motor torque limit of every finger joint).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pybullet as p

from ..cup.env import CupConfig, CupScenario

EFFORT_MIN, EFFORT_MAX = 0.01, 0.6
ACTION_NAMES = ("closure", "effort")
ACTION_LOW = np.array([0.0, EFFORT_MIN], np.float32)
ACTION_HIGH = np.array([1.0, EFFORT_MAX], np.float32)
SETUP_EFFORT = 0.25
THUMB_FRACTION = 0.2          # thumb flex = closure * this: stays clear of the mug (no load path)
MAX_JOINT_SPEED = 4.0         # rad/s motor velocity cap (uncapped tiny phalanges snap at > 30 rad/s)
HANDLE = dict(radius=0.011, length=0.07, x=0.085, z=0.19)
MUG = dict(radius=0.03, height=0.08, dy=-0.065, dz=-0.025)

OBS_GROUPS = {
    "gripper": ["closure_meas", "closure_vel", "closure_cmd_prev", "effort_prev"],
    "object": ["obj_rel_x", "obj_rel_y", "obj_rel_z", "obj_vx", "obj_vy", "obj_vz"],
    "contact": ["normal_index", "normal_middle", "normal_palm", "support_z", "contact_index", "contact_middle"],
}
OBS_NAMES = tuple(n for g in OBS_GROUPS.values() for n in g)
CONTACT_EPS = 0.05
HOLD_STATES = ("HOLDING", "MOVING RELATIVE TO HAND", "DROPPED", "EXCESSIVE SIMULATED LOAD", "COMPLETED", "RELEASED")


def obs_indices(groups) -> list[int]:
    return [OBS_NAMES.index(n) for g in groups for n in OBS_GROUPS[g]]


def default_config() -> CupConfig:
    """Same load schedules and outcome rules as fragile_cup; hand-specific damage threshold kept at the
    fragile_cup instant value (25 N per finger/digit, SIMULATED). Mass/friction ranges as fragile_cup."""
    return CupConfig(damage_model="instant", break_force_n=25.0, mass_range=(0.12, 0.2), friction_range=(0.4, 0.9))


@dataclass
class AssistStats:
    ticks: int = 0
    termination: str | None = None    # success | drop | released | damage_break | excess_slip | setup_failed | timeout | aborted
    success: bool = False
    peak_normal_n: float = 0.0
    peak_load_n: float = 0.0
    max_disp_m: float = 0.0
    final_disp_m: float = 0.0
    moving_ticks: int = 0
    mean_effort: float = 0.0
    failure_phase: str | None = None
    events: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("events")
        return d


class AssistHandEnv:
    def __init__(self, cfg: CupConfig | None = None):
        from ..server.teleop.hand import FINGERS, LINKS_PER_FINGER, HandSim
        self.cfg = cfg or default_config()
        c = self.cfg
        self.hand = HandSim(gui=False, physics_hz=c.physics_hz, control_hz=c.control_hz)
        self.client, self.body = self.hand.client, self.hand.body
        self._p = dict(physicsClientId=self.client)
        self.LPF = LINKS_PER_FINGER
        self.pods = self.hand.pods
        self.joints, self.limits = list(self.hand.joints), list(self.hand.limits)
        self.thumb_n = self.pods.index("thumb_tip") if "thumb_tip" in self.pods else None
        for j in range(p.getNumJoints(self.body, **self._p)):
            # no explicit joint damping: with ~1e-6 kg m^2 phalanges at 240 Hz it is numerically unstable
            p.changeDynamics(self.body, j, jointDamping=0.0, lateralFriction=1.0, **self._p)
        p.changeDynamics(self.body, -1, lateralFriction=1.0, **self._p)
        qy = p.getQuaternionFromEuler((math.pi / 2, 0, 0))
        col = p.createCollisionShapeArray(
            [p.GEOM_CYLINDER, p.GEOM_CYLINDER], radii=[HANDLE["radius"], MUG["radius"]],
            lengths=[HANDLE["length"], MUG["height"]],
            collisionFramePositions=[(0, 0, 0), (0, MUG["dy"], MUG["dz"])],
            collisionFrameOrientations=[qy, (0, 0, 0, 1)], **self._p)
        vis = p.createVisualShapeArray(
            [p.GEOM_CYLINDER, p.GEOM_CYLINDER], radii=[HANDLE["radius"], MUG["radius"]],
            lengths=[HANDLE["length"], MUG["height"]],
            visualFramePositions=[(0, 0, 0), (0, MUG["dy"], MUG["dz"])],
            visualFrameOrientations=[qy, (0, 0, 0, 1)], **self._p)
        self.mug = p.createMultiBody(0.15, col, vis, self.grasp_point, **self._p)
        self.scenario: CupScenario | None = None
        self.done = True
        self.stats = AssistStats()
        self.fixture = None

    @property
    def grasp_point(self):
        return (HANDLE["x"], 0.0, HANDLE["z"])

    # -- episode -----------------------------------------------------------------------
    def reset(self, scenario: CupScenario | int, pour_at_s: float | None = None) -> np.ndarray:
        if not isinstance(scenario, CupScenario):
            scenario = CupScenario.sample(int(scenario), self.cfg)
        self.scenario = sc = scenario
        c = self.cfg
        if self.fixture is not None:
            p.removeConstraint(self.fixture, **self._p)
        for j in self.joints:
            p.resetJointState(self.body, j, 0.0, **self._p)
        p.changeDynamics(self.mug, -1, mass=sc.mass, lateralFriction=sc.friction, **self._p)
        p.resetBasePositionAndOrientation(self.mug, self.grasp_point, (0, 0, 0, 1), **self._p)
        p.resetBaseVelocity(self.mug, (0, 0, 0), (0, 0, 0), **self._p)
        self.fixture = p.createConstraint(self.mug, -1, -1, -1, p.JOINT_FIXED, (0, 0, 0), (0, 0, 0),
                                          self.grasp_point, **self._p)
        self.cmd = np.array([1.0, SETUP_EFFORT], np.float32)
        self.load_n = 0.0
        self._contact = self._zero()
        for _ in range(int(1.5 * c.physics_hz)):            # setup: close on the fixed handle
            self._run_substep()
        p.removeConstraint(self.fixture, **self._p)         # release into the grasp: no attachment remains
        self.fixture = None
        for _ in range(int(0.5 * c.physics_hz)):
            self._run_substep()
        self.stats = AssistStats()
        self._grasp_rel = self._obj_rel()
        self._prev_rel = self._grasp_rel.copy()
        self._prev_closure = self._closure_meas()
        self._effort_sum = 0.0
        self._moving_run = 0
        self.hold_state = "HOLDING"
        self.t = 0.0
        self.pour_t0 = pour_at_s
        self.done = False
        self._contact = self._measure(4)
        if not (self._contact["normal"][0] > 0.5 and self._contact["normal"][1] > 0.5 and
                np.linalg.norm(self._grasp_rel - np.array([0.0, 0.0, 0.0])) < 0.05):
            self.stats.termination, self.done = "setup_failed", True
        return self.observe()

    def pour(self) -> bool:
        if self.pour_t0 is None and not self.done:
            self.pour_t0 = self.t
            self.stats.events.append({"tick": self.stats.ticks, "event": "add_load"})
            return True
        return False

    @property
    def schedule_t(self):
        return None if self.pour_t0 is None else self.t - self.pour_t0

    @property
    def end_t(self):
        return None if self.pour_t0 is None else self.pour_t0 + self.scenario.schedule.duration + self.cfg.final_hold_s

    def phase(self) -> str:
        st = self.schedule_t
        if st is None or st < 0:
            return "pre_load"
        if st >= self.scenario.schedule.duration:
            return "final_hold"
        return self.scenario.schedule.segment(st)

    def step(self, action) -> tuple[np.ndarray, bool, dict]:
        if self.done:
            raise RuntimeError("episode finished; reset()")
        c = self.cfg
        self.cmd = clip_action(action)
        self._prev_rel = self._obj_rel()
        self._prev_closure = self._closure_meas()
        acc = self._zero()
        n = c.substeps
        for k in range(n):
            st = self.schedule_t
            self.load_n = 0.0 if st is None or st < 0 else self.scenario.schedule.load(st + k / c.physics_hz)
            self._run_substep()
            f = self._forces()
            for key in ("normal", "support_z"):
                acc[key] = acc[key] + f[key] / n
        self._contact = acc
        self.t += c.dt
        info = self._outcome()
        return self.observe(), self.done, info

    # -- internals ------------------------------------------------------------------------
    def _targets(self) -> list[float]:
        closure, out = float(self.cmd[0]), []
        for n, pod in enumerate(self.pods):
            frac = closure * THUMB_FRACTION if n == self.thumb_n else closure
            out += [frac * lim for lim in self.limits[3 * n:3 * n + 3]]
        return out

    def _run_substep(self) -> None:
        effort = float(self.cmd[1])
        for j, t in zip(self.joints, self._targets()):
            p.setJointMotorControl2(self.body, j, p.POSITION_CONTROL, targetPosition=t, force=effort,
                                    maxVelocity=MAX_JOINT_SPEED, **self._p)
        if self.load_n:
            pos = p.getBasePositionAndOrientation(self.mug, **self._p)[0]
            p.applyExternalForce(self.mug, -1, (0, 0, -self.load_n), pos, p.WORLD_FRAME, **self._p)
        p.stepSimulation(**self._p)

    def _zero(self) -> dict:
        return {"normal": np.zeros(3), "support_z": 0.0}           # normal: index, middle, palm

    def _forces(self) -> dict:
        """One contact query per substep; each point is assigned once to index / middle / palm / other.
        normal = sum of contact normal force magnitudes per digit; support_z = vertical component of the
        net force the hand exerts on the mug (normal + both friction directions)."""
        normal = np.zeros(3)
        Fz = 0.0
        for c in p.getContactPoints(self.body, self.mug, **self._p):
            link = c[3]
            if link == -1:
                k = 2
            else:
                pod = self.pods[link // self.LPF]
                k = 0 if pod == "index_tip" else 1 if pod == "middle_tip" else None
            f = -(c[9] * np.array(c[7]) + c[10] * np.array(c[11]) + c[12] * np.array(c[13]))
            Fz += f[2]
            if k is not None:
                normal[k] += c[9]
        return {"normal": normal, "support_z": Fz}

    def _measure(self, n: int) -> dict:
        acc = self._zero()
        for _ in range(n):
            self._run_substep()
            f = self._forces()
            acc["normal"] = acc["normal"] + f["normal"] / n
            acc["support_z"] += f["support_z"] / n
        return acc

    def _obj_rel(self) -> np.ndarray:
        return np.array(p.getBasePositionAndOrientation(self.mug, **self._p)[0]) - np.array(self.grasp_point)

    def _closure_meas(self) -> float:
        q = []
        for n, pod in enumerate(self.pods):
            if n == self.thumb_n:
                continue
            for j, lim in zip(self.joints[3 * n:3 * n + 3], self.limits[3 * n:3 * n + 3]):
                q.append(p.getJointState(self.body, j, **self._p)[0] / lim)
        return float(np.mean(q))

    def joint_state(self) -> dict:
        qs = [p.getJointState(self.body, j, **self._p) for j in self.joints]
        return {"q": [s[0] for s in qs], "qd": [s[1] for s in qs], "limits": list(self.limits)}

    def _outcome(self) -> dict:
        c, st = self.cfg, self.stats
        st.ticks += 1
        nrm = self._contact["normal"]
        st.peak_normal_n = max(st.peak_normal_n, float(nrm.max()))
        st.peak_load_n = max(st.peak_load_n, self.load_n)
        self._effort_sum += float(self.cmd[1])
        st.mean_effort = self._effort_sum / st.ticks
        rel = self._obj_rel()
        disp = float(np.linalg.norm(rel - self._grasp_rel))
        st.max_disp_m, st.final_disp_m = max(st.max_disp_m, disp), disp
        rel_speed = float(np.linalg.norm(rel - self._prev_rel)) / c.dt
        # documented relative-motion detector: mug moving > 10 mm/s relative to the (fixed) hand for >= 3 ticks
        self._moving_run = self._moving_run + 1 if rel_speed > 0.01 else 0
        moving = self._moving_run >= 3
        st.moving_ticks += int(moving)
        phase = self.phase()
        touching = (nrm[:2] > CONTACT_EPS).any()
        term = None
        if nrm.max() > c.break_force_n:
            term = "damage_break"
        elif disp > c.drop_slip_m or (not touching and disp > 0.01):
            term = "released" if self.cmd[0] < 0.3 else "drop"
        elif self.end_t is not None and self.t >= self.end_t - 1e-9:
            term = "success" if disp <= c.success_slip_m else "excess_slip"
        elif self.pour_t0 is None and self.t >= c.live_timeout_s:
            term = "timeout"
        self.hold_state = ("EXCESSIVE SIMULATED LOAD" if term == "damage_break" else "DROPPED" if term == "drop" else
                           "RELEASED" if term == "released" else "COMPLETED" if term in ("success", "excess_slip") else
                           "MOVING RELATIVE TO HAND" if moving else "HOLDING")
        if term:
            st.termination, st.success = term, term == "success"
            if not st.success:
                st.failure_phase = phase
            self.done = True
        return {"load_n": self.load_n, "normal_index": float(nrm[0]), "normal_middle": float(nrm[1]),
                "normal_palm": float(nrm[2]), "support_z": float(self._contact["support_z"]), "disp_m": disp,
                "rel_speed_mps": rel_speed, "moving": moving, "hold_state": self.hold_state, "phase": phase,
                "closure_cmd": float(self.cmd[0]), "effort_cmd": float(self.cmd[1]),
                "closure_meas": self._closure_meas(), "termination": st.termination}

    def observe(self) -> np.ndarray:
        dt = self.cfg.dt
        rel = self._obj_rel()
        vel = (rel - self._prev_rel) / dt if hasattr(self, "_prev_rel") else np.zeros(3)
        cm = self._closure_meas()
        cv = (cm - self._prev_closure) / dt if hasattr(self, "_prev_closure") else 0.0
        nrm = self._contact["normal"]
        return np.concatenate([[cm, cv, self.cmd[0], self.cmd[1]], rel, vel,
                               nrm, [self._contact["support_z"]], (nrm[:2] > CONTACT_EPS).astype(float)]).astype(np.float32)

    def privileged(self) -> dict:
        return {"load_n": self.load_n, "mass": self.scenario.mass, "friction": self.scenario.friction}

    def close(self) -> None:
        self.hand.close()


def clip_action(a) -> np.ndarray:
    a = np.nan_to_num(np.asarray(a, np.float32).reshape(2), nan=0.0)
    return np.clip(a, ACTION_LOW, ACTION_HIGH)
