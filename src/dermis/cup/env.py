"""fragile_cup: hold a cup-like rigid cylinder in the Panda gripper while a downward load ramps up.

SIMULATION BENCHMARK. The cup is a rigid cylinder: it never deforms. "Damage" is a *simulated
threshold* on fingertip contact load. The "contents" are an external downward force on the
cup's centre of mass: a LOAD PROXY, not a fluid simulation. Nothing here validates real glove
measurements, real cup mechanics or prosthetic performance.

Frames and units: world frame, z up, metres, seconds, newtons, kilograms. 30 Hz control,
240 Hz physics (8 substeps per tick); contact quantities are averaged over a tick's substeps.

Arm: fixed. Only the grip is controlled. Every episode starts with the same scripted setup
(reset()): the open gripper straddles the cup standing on the table, the fingers close with
SETUP_EFFORT_N, the arm lifts the cup LIFT_HEIGHT, the scene settles, and the grasp is checked.
That procedure does not depend on the policy being evaluated. The arm is then locked.

Action (1 float): finger effort limit, N, clipped to [MIN_EFFORT, MAX_EFFORT]. The fingers are
always commanded to close (target opening 0); with the cup between them the squeeze force is
approximately this limit. Grip effort is the only thing a policy controls.

Contact forces (cup <- finger), see _finger_forces():
    For each contact point between a finger link and the cup, PyBullet reports the normal force
    fn along contactNormalOnB (pointing from the cup toward the finger) and two lateral friction
    forces f1, f2 along their directions. The force the finger exerts on the cup is
        F = -(fn * n + f1 * d1 + f2 * d2)
    summed over that finger's contact points of one substep, then averaged over substeps. Each
    finger is queried once per substep, contacts are grouped by link index, so nothing is
    counted twice. F is decomposed against the finger's actual closing axis u (from the finger
    link positions, pointing from that finger toward the cup centre):
        normal   = F . u                (squeeze, N)
        shear    = F - normal * u       (tangential, N); shear_z = its world-z component
    Static check (tests/test_cup_env.py): shear_z(left) + shear_z(right) ~= m g + applied load.
    Contacts between the cup and the palm/hand link are NOT finger contacts; they are logged
    separately (palm_force) and never enter the contact features.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pybullet as p
import pybullet_data

from ..sim.env import ARM_JOINTS, EE_DOWN, EE_LINK, FINGER_JOINTS, HOME_JOINTS, MAX_OPENING
from .schedule import LoadSchedule

HAND_LINK = 8
MIN_EFFORT, MAX_EFFORT = 0.5, 40.0
ACTION_NAMES = ("effort",)
SETUP_EFFORT_N = 8.0
LIFT_HEIGHT = 0.10

OBS_GROUPS = {
    # identical for both policies: gripper state, own previous command, object pose/velocity
    "gripper": ["opening", "opening_vel", "effort_prev"],
    "object": ["obj_rel_x", "obj_rel_y", "obj_rel_z", "obj_vx", "obj_vy", "obj_vz"],
    # simulated finger-object contact only (contact-aware policy)
    "contact": ["normal_l", "normal_r", "shear_z_l", "shear_z_r", "shear_l", "shear_r", "contact_l", "contact_r"],
}
OBS_NAMES = tuple(n for g in OBS_GROUPS.values() for n in g)
CONTACT_EPS_N = 0.05


def obs_indices(groups) -> list[int]:
    return [OBS_NAMES.index(n) for g in groups for n in OBS_GROUPS[g]]


@dataclass
class CupConfig:
    control_hz: int = 30
    physics_hz: int = 240
    cup_radius: float = 0.03
    cup_height: float = 0.09
    cup_xy: tuple = (0.55, 0.0)
    mass_range: tuple = (0.08, 0.15)            # kg, empty cup
    friction_range: tuple = (0.35, 0.9)         # cup surface; finger pads are 1.0 -> effective = this
    # load proxy schedule (see schedule.py)
    pre_pour_s: float = 1.0
    max_load_range: tuple = (2.0, 8.0)          # N, peak downward load
    ramp_rate_range: tuple = (0.6, 2.0)         # N/s while ramping
    n_ramps_range: tuple = (2, 4)               # ramp segments separated by pauses
    pause_range: tuple = (0.5, 1.5)             # s
    disturbance_count_range: tuple = (0, 3)
    disturbance_amp: float = 1.5                # N, bounded |extra load| of one disturbance
    disturbance_len_range: tuple = (0.15, 0.4)  # s
    final_hold_s: float = 2.0
    # outcome definitions
    break_force_n: float = 25.0                 # per-finger normal above this = instantaneous damage (simulated)
    damage_model: str = "instant"               # "instant" | "instant+sustained"
    sustained_safe_n: float = 8.0               # per-finger normal tolerated indefinitely (sustained model)
    damage_budget_ns: float = 15.0              # integral of max(0, normal - safe) dt allowed (sustained model)
    drop_slip_m: float = 0.03                   # cup slid this far down relative to the gripper = drop
    drop_no_contact_ticks: int = 3              # both fingers off the cup this long = drop
    success_slip_m: float = 0.015               # total slip allowed for a successful hold
    live_timeout_s: float = 120.0               # live mode: no Pour within this -> timeout

    @property
    def substeps(self) -> int:
        return self.physics_hz // self.control_hz

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "CupConfig":
        d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (d or {}).items()}
        return cls(**d)


@dataclass
class CupScenario:
    seed: int
    mass: float
    friction: float
    schedule: LoadSchedule

    @classmethod
    def sample(cls, seed: int, cfg: CupConfig) -> "CupScenario":
        rng = np.random.default_rng([seed, 4242])
        mass = float(rng.uniform(*cfg.mass_range))
        mu = float(rng.uniform(*cfg.friction_range))
        return cls(int(seed), mass, mu, LoadSchedule.sample(rng, cfg))

    def to_dict(self) -> dict:
        return {"seed": self.seed, "mass": self.mass, "friction": self.friction, "schedule": self.schedule.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "CupScenario":
        return cls(int(d["seed"]), float(d["mass"]), float(d["friction"]), LoadSchedule.from_dict(d["schedule"]))


@dataclass
class CupStats:
    ticks: int = 0
    termination: str | None = None    # success | drop | damage_break | damage_sustained | excess_slip | setup_failed | timeout | aborted
    success: bool = False
    peak_normal_n: float = 0.0
    peak_load_n: float = 0.0
    damage_ns: float = 0.0
    max_slip_m: float = 0.0
    final_slip_m: float = 0.0
    mean_effort_n: float = 0.0
    load_at_end_n: float = 0.0
    failure_phase: str | None = None   # pre_pour | ramp | pause | final_hold (where a failure happened)
    events: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class CupHoldEnv:
    def __init__(self, cfg: CupConfig | None = None):
        """Physics only, always headless (DIRECT). Display is a separate CupViewer (viewer.py)."""
        self.cfg = cfg or CupConfig()
        self.client = p.connect(p.DIRECT)
        self._p = dict(physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), **self._p)
        p.setGravity(0, 0, -9.81, **self._p)
        p.setTimeStep(1.0 / self.cfg.physics_hz, **self._p)
        self.plane = p.loadURDF("plane.urdf", **self._p)
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, **self._p)
        for j in FINGER_JOINTS:
            p.changeDynamics(self.robot, j, lateralFriction=1.0, spinningFriction=0.001, **self._p)
        c = self.cfg
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=c.cup_radius, height=c.cup_height, **self._p)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=c.cup_radius, length=c.cup_height,
                                  rgbaColor=(0.95, 0.95, 0.98, 0.9), **self._p)
        self.cup = p.createMultiBody(0.1, col, vis, (*c.cup_xy, c.cup_height / 2), **self._p)
        self.scenario: CupScenario | None = None
        self.done = True

    # -- episode -------------------------------------------------------------------

    def reset(self, scenario: CupScenario | int, pour_at_s: float | None = None) -> np.ndarray:
        """Identical setup for every policy; see module docstring. pour_at_s=None -> the schedule
        starts only when pour() is called (live mode)."""
        if not isinstance(scenario, CupScenario):
            scenario = CupScenario.sample(int(scenario), self.cfg)
        self.scenario = sc = scenario
        c = self.cfg
        p.changeDynamics(self.cup, -1, mass=sc.mass, lateralFriction=sc.friction, spinningFriction=0.002,
                         rollingFriction=0.0, **self._p)
        p.resetBasePositionAndOrientation(self.cup, (*c.cup_xy, c.cup_height / 2), (0, 0, 0, 1), **self._p)
        p.resetBaseVelocity(self.cup, (0, 0, 0), (0, 0, 0), **self._p)
        grasp_z = c.cup_height / 2 + 0.01                       # just above the cup's centre of mass
        self.arm_target = np.array([c.cup_xy[0], c.cup_xy[1], grasp_z])
        for j, q in zip(ARM_JOINTS, HOME_JOINTS):
            p.resetJointState(self.robot, j, q, **self._p)
        for _ in range(3):                                      # IK from the current solution converges better
            for j, q in zip(ARM_JOINTS, self._ik(self.arm_target)):
                p.resetJointState(self.robot, j, q, **self._p)
        for j in FINGER_JOINTS:
            p.resetJointState(self.robot, j, MAX_OPENING / 2, **self._p)
        self._arm_q = self._ik(self.arm_target)
        self.effort = SETUP_EFFORT_N
        self.load_n = 0.0
        self._contact = self._zero_contact()
        # setup: close, lift, settle (not policy-controlled, not recorded)
        self._run(int(0.5 * c.physics_hz), close=True)
        steps = int(1.0 * c.physics_hz)
        start = self.arm_target.copy()
        for k in range(0, steps, c.substeps):
            self.arm_target = start + np.array([0, 0, LIFT_HEIGHT * (k + c.substeps) / steps])
            self._arm_q = self._ik(self.arm_target)
            self._run(c.substeps, close=True)
        self._run(int(0.5 * c.physics_hz), close=True)
        # the arm stays at this joint target for the rest of the episode (locked)
        self.stats = CupStats()
        self._grasp_rel = self._cup_pos() - self._ee_pos()
        self._prev_open = self._opening()
        self._prev_cup_rel = self._grasp_rel.copy()
        self._no_contact = 0
        self.t = 0.0
        self.pour_t0 = pour_at_s
        self.done = False
        self._effort_sum = 0.0
        self._contact = self._measure_contact(4)
        ok = self._contact["normal"][0] > 1.0 and self._contact["normal"][1] > 1.0 and abs(self._grasp_rel[2]) < 0.03
        if not ok:
            self.stats.termination = "setup_failed"
            self.done = True
        return self.observe()

    def pour(self) -> bool:
        """Start the load schedule now (live mode). Ignored if already pouring."""
        if self.pour_t0 is None and not self.done:
            self.pour_t0 = self.t
            self.stats.events.append({"tick": self.stats.ticks, "event": "pour"})
            return True
        return False

    @property
    def schedule_t(self) -> float | None:
        return None if self.pour_t0 is None else self.t - self.pour_t0

    @property
    def end_t(self) -> float | None:
        return None if self.pour_t0 is None else self.pour_t0 + self.scenario.schedule.duration + self.cfg.final_hold_s

    def phase(self) -> str:
        st = self.schedule_t
        if st is None or st < 0:
            return "pre_pour"
        if st >= self.scenario.schedule.duration:
            return "final_hold"
        return self.scenario.schedule.segment(st)

    def step(self, action) -> tuple[np.ndarray, bool, dict]:
        if self.done:
            raise RuntimeError("step() after episode end; call reset()")
        c = self.cfg
        self.effort = clip_effort(action)
        self._prev_open = self._opening()
        self._prev_cup_rel = self._cup_pos() - self._ee_pos()
        n = c.substeps
        acc = self._zero_contact()
        palm = 0.0
        for k in range(n):
            st = self.schedule_t
            self.load_n = 0.0 if st is None or st < 0 else self.scenario.schedule.load(st + k / c.physics_hz)
            self._run(1, close=True)
            fc = self._finger_forces()
            for key in ("normal", "shear_z", "shear"):
                acc[key] += fc[key]
            palm += fc["palm"]
        for key in ("normal", "shear_z", "shear"):
            acc[key] /= n
        acc["palm"] = palm / n
        self._contact = acc
        self.t += c.dt
        info = self._update_outcome()
        return self.observe(), self.done, info

    def _update_outcome(self) -> dict:
        c, st = self.cfg, self.stats
        st.ticks += 1
        normal = self._contact["normal"]
        st.peak_normal_n = max(st.peak_normal_n, float(normal.max()))
        st.peak_load_n = max(st.peak_load_n, self.load_n)
        self._effort_sum += self.effort
        st.mean_effort_n = self._effort_sum / st.ticks
        if "sustained" in c.damage_model:
            st.damage_ns += float(np.maximum(0.0, normal - c.sustained_safe_n).max()) * c.dt
        rel = self._cup_pos() - self._ee_pos()
        slip = float(self._grasp_rel[2] - rel[2])          # downward slide of the cup relative to the gripper
        st.max_slip_m = max(st.max_slip_m, slip)
        st.final_slip_m = slip
        touching = normal > CONTACT_EPS_N
        self._no_contact = 0 if touching.any() else self._no_contact + 1
        phase = self.phase()
        term = None
        if normal.max() > c.break_force_n:
            term = "damage_break"
        elif "sustained" in c.damage_model and st.damage_ns > c.damage_budget_ns:
            term = "damage_sustained"
        elif slip > c.drop_slip_m or self._no_contact >= c.drop_no_contact_ticks:
            term = "drop"
        elif self.end_t is not None and self.t >= self.end_t - 1e-9:
            term = "success" if slip <= c.success_slip_m else "excess_slip"
        elif self.pour_t0 is None and self.t >= c.live_timeout_s:
            term = "timeout"
        if term:
            st.termination = term
            st.success = term == "success"
            st.load_at_end_n = self.load_n
            if not st.success:
                st.failure_phase = phase
            self.done = True
        return {"load_n": self.load_n, "normal_l": float(normal[0]), "normal_r": float(normal[1]),
                "shear_z_l": float(self._contact["shear_z"][0]), "shear_z_r": float(self._contact["shear_z"][1]),
                "palm_n": float(self._contact["palm"]), "slip_m": slip, "effort": self.effort, "phase": phase,
                "damage_ns": st.damage_ns, "termination": st.termination}

    # -- observation ------------------------------------------------------------------

    def observe(self) -> np.ndarray:
        """Policy observation (OBS_NAMES order). Never contains load, schedule, friction, mass,
        damage, or expert state."""
        dt = self.cfg.dt
        opening = self._opening()
        rel = self._cup_pos() - self._ee_pos()
        vel = (rel - self._prev_cup_rel) / dt
        c = self._contact
        return np.concatenate([
            [opening, (opening - self._prev_open) / dt, self.effort],
            rel, vel,
            c["normal"], c["shear_z"], c["shear"], (c["normal"] > CONTACT_EPS_N).astype(float),
        ]).astype(np.float32)

    def privileged(self) -> dict:
        """Ground truth for scripted controllers and analysis; never a learned-policy input."""
        return {"load_n": self.load_n, "mass": self.scenario.mass, "friction": self.scenario.friction,
                "contact": self._contact, "phase": self.phase(), "damage_ns": self.stats.damage_ns}

    # -- physics internals ----------------------------------------------------------------

    def _run(self, n: int, close: bool) -> None:
        q = self._arm_q
        for _ in range(n):
            p.setJointMotorControlArray(self.robot, ARM_JOINTS, p.POSITION_CONTROL, targetPositions=q,
                                        forces=[240.0] * 7, **self._p)
            for j in FINGER_JOINTS:
                p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=0.0 if close else MAX_OPENING / 2,
                                        force=self.effort, maxVelocity=0.1, **self._p)
            if self.load_n:
                pos = self._cup_pos()
                p.applyExternalForce(self.cup, -1, (0, 0, -self.load_n), tuple(pos), p.WORLD_FRAME, **self._p)
            p.stepSimulation(**self._p)

    def _finger_forces(self) -> dict:
        """Force each finger exerts on the cup, decomposed against that finger's closing axis."""
        ls = [np.array(p.getLinkState(self.robot, j, computeForwardKinematics=True, **self._p)[0]) for j in FINGER_JOINTS]
        axis = ls[1] - ls[0]
        axis /= max(1e-9, np.linalg.norm(axis))
        u = (axis, -axis)                                      # left finger pushes toward right, and vice versa
        F = [np.zeros(3), np.zeros(3)]
        palm = 0.0
        for cpt in p.getContactPoints(self.robot, self.cup, **self._p):
            link = cpt[3]
            n = np.array(cpt[7])
            f = -(cpt[9] * n + cpt[10] * np.array(cpt[11]) + cpt[12] * np.array(cpt[13]))
            if link in FINGER_JOINTS:
                F[FINGER_JOINTS.index(link)] += f
            else:
                palm += float(np.linalg.norm(f))
        normal = np.array([F[i] @ u[i] for i in range(2)])
        shear_v = [F[i] - normal[i] * u[i] for i in range(2)]
        return {"normal": normal, "shear_z": np.array([s[2] for s in shear_v]),
                "shear": np.array([np.linalg.norm(s) for s in shear_v]), "palm": palm, "vec": F}

    def _measure_contact(self, n: int) -> dict:
        acc = self._zero_contact()
        for _ in range(n):
            self._run(1, close=True)
            fc = self._finger_forces()
            for key in ("normal", "shear_z", "shear"):
                acc[key] += fc[key] / n
        return acc

    @staticmethod
    def _zero_contact() -> dict:
        return {"normal": np.zeros(2), "shear_z": np.zeros(2), "shear": np.zeros(2), "palm": 0.0}

    def _ik(self, pos) -> list:
        q = p.calculateInverseKinematics(self.robot, EE_LINK, list(pos), EE_DOWN, maxNumIterations=200,
                                         residualThreshold=1e-5, **self._p)
        return list(q[:len(ARM_JOINTS)])

    def _ee_pos(self) -> np.ndarray:
        return np.array(p.getLinkState(self.robot, EE_LINK, computeForwardKinematics=True, **self._p)[4])

    def _cup_pos(self) -> np.ndarray:
        return np.array(p.getBasePositionAndOrientation(self.cup, **self._p)[0])

    def _opening(self) -> float:
        return float(sum(p.getJointState(self.robot, j, **self._p)[0] for j in FINGER_JOINTS))

    def joint_effort_limits(self) -> list:
        """What the finger motors were last commanded (for tests)."""
        return [self.effort, self.effort]

    def close(self) -> None:
        if p.isConnected(self.client):
            p.disconnect(self.client)


def clip_effort(action) -> float:
    a = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
    if not math.isfinite(a):
        a = MIN_EFFORT
    return min(MAX_EFFORT, max(MIN_EFFORT, a))
