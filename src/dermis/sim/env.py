"""Grasp-lift-place with a Franka Panda parallel-jaw gripper in PyBullet.

Frames and units
    World frame: z up, origin on the table surface under the robot base, metres, seconds, newtons.
    The gripper is held pointing straight down (fixed orientation); only its position moves.
    "ee" is the Panda grasp target (link 11), the point midway between the fingertips.

Timing
    One call to step() is one policy/control tick (control_hz, default 30 Hz). Inside it the
    physics engine runs physics_hz / control_hz substeps (default 240 / 30 = 8). Contact forces
    are averaged over the substeps of a tick; peaks are taken over per-tick averages.

Action (5 floats, clipped to ACTION_LOW..ACTION_HIGH before use)
    vx, vy, vz   desired gripper velocity, m/s, world frame (|v| per axis <= max_speed)
    opening      desired total finger opening, m (0 closed .. 0.08 fully open)
    effort       finger motor force limit, N; with the fingers closed on the object the
                 squeeze force is roughly this value

    The velocity is integrated into a position target (target += v * dt), clamped to the
    workspace and to max_lead metres ahead of the measured gripper, then tracked with IK.

Contact
    Real contact physics: the object is never attached. Friction is PyBullet's product of the
    two bodies' coefficients; finger pads are 1.0, so the configured object friction is the
    effective coefficient.

    EXCESS_FORCE_N is a *simulated benchmark threshold* on the per-finger normal force on the
    object. It is not a biological, clinical or tissue-damage measurement.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pybullet as p
import pybullet_data

# --- robot ---------------------------------------------------------------
ARM_JOINTS = list(range(7))
FINGER_JOINTS = (9, 10)
EE_LINK = 11
HOME_JOINTS = (0.0, -0.3, 0.0, -2.2, 0.0, 1.9, math.pi / 4)
EE_DOWN = p.getQuaternionFromEuler((math.pi, 0.0, 0.0))          # gripper pointing down, fingers close along world y
MAX_OPENING = 0.08

# --- action space ----------------------------------------------------------
ACTION_NAMES = ("vx", "vy", "vz", "opening", "effort")
MAX_SPEED = 0.20
MIN_EFFORT, MAX_EFFORT = 2.0, 40.0
ACTION_LOW = np.array([-MAX_SPEED, -MAX_SPEED, -MAX_SPEED, 0.0, MIN_EFFORT], dtype=np.float32)
ACTION_HIGH = np.array([MAX_SPEED, MAX_SPEED, MAX_SPEED, MAX_OPENING, MAX_EFFORT], dtype=np.float32)

# --- observation -----------------------------------------------------------
# Every feature the policy may see, in order. Group names let configs drop whole groups
# (the "no contact" baseline drops "contact").
OBS_GROUPS = {
    "proprio": ["ee_x", "ee_y", "ee_z", "ee_vx", "ee_vy", "ee_vz", "opening", "opening_vel"],
    "object": ["obj_rel_x", "obj_rel_y", "obj_rel_z", "obj_vx", "obj_vy", "obj_vz"],
    "target": ["tgt_rel_ee_x", "tgt_rel_ee_y", "tgt_rel_obj_x", "tgt_rel_obj_y"],
    "contact": ["force_l", "force_r", "contact_l", "contact_r"],
}
OBS_NAMES = tuple(n for g in OBS_GROUPS.values() for n in g)

EXCESS_FORCE_N = 30.0          # simulated benchmark threshold, see module docstring
CONTACT_EPS_N = 0.05           # normal force above this counts as touching


@dataclass
class TaskConfig:
    control_hz: int = 30
    physics_hz: int = 240
    timeout_s: float = 15.0
    max_speed: float = MAX_SPEED
    max_lead: float = 0.03                       # m the position target may run ahead of the gripper
    workspace: tuple = ((0.30, 0.75), (-0.30, 0.30), (0.005, 0.40))
    object_half: float = 0.02                    # cube half-size, m
    object_mass: float = 0.2                     # kg
    friction_range: tuple = (0.4, 1.0)
    object_x_range: tuple = (0.45, 0.62)
    object_y_range: tuple = (-0.18, -0.02)
    target_x_range: tuple = (0.45, 0.62)
    target_y_range: tuple = (0.06, 0.18)
    target_radius: float = 0.035                 # object centre within this of the target centre
    dwell_ticks: int = 15                        # released, resting in target this long = success
    start_ee: tuple = (0.50, 0.0, 0.25)
    excess_force_n: float = EXCESS_FORCE_N
    drop_height: float = 0.03                    # released this far above rest height = drop

    @property
    def substeps(self) -> int:
        assert self.physics_hz % self.control_hz == 0, "physics_hz must be a multiple of control_hz"
        return self.physics_hz // self.control_hz

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def max_ticks(self) -> int:
        return int(round(self.timeout_s * self.control_hz))

    @classmethod
    def from_dict(cls, d: dict | None) -> "TaskConfig":
        d = dict(d or {})
        for k in ("workspace",):
            if k in d:
                d[k] = tuple(tuple(v) for v in d[k])
        for k, v in list(d.items()):
            if isinstance(v, list):
                d[k] = tuple(v)
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scenario:
    """Everything that varies between episodes. Sampled from a seed, or given explicitly (eval suites)."""
    seed: int
    object_xy: tuple
    target_xy: tuple
    friction: float

    @classmethod
    def sample(cls, seed: int, cfg: TaskConfig) -> "Scenario":
        rng = np.random.default_rng(seed)
        return cls(
            seed=int(seed),
            object_xy=(float(rng.uniform(*cfg.object_x_range)), float(rng.uniform(*cfg.object_y_range))),
            target_xy=(float(rng.uniform(*cfg.target_x_range)), float(rng.uniform(*cfg.target_y_range))),
            friction=float(rng.uniform(*cfg.friction_range)),
        )

    def to_dict(self) -> dict:
        return {"seed": self.seed, "object_xy": list(self.object_xy),
                "target_xy": list(self.target_xy), "friction": self.friction}

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(int(d["seed"]), tuple(d["object_xy"]), tuple(d["target_xy"]), float(d["friction"]))


@dataclass
class EpisodeStats:
    ticks: int = 0
    drops: int = 0
    excess_force_ticks: int = 0
    peak_force_n: float = 0.0
    peak_table_force_n: float = 0.0
    was_lifted: bool = False
    success: bool = False
    completion_time_s: float | None = None
    termination: str | None = None             # success | timeout | object_out_of_bounds | aborted
    events: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class GraspLiftPlaceEnv:
    def __init__(self, cfg: TaskConfig | None = None, gui: bool = False):
        self.cfg = cfg or TaskConfig()
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        self._p = dict(physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), **self._p)
        p.setGravity(0, 0, -9.81, **self._p)
        p.setTimeStep(1.0 / self.cfg.physics_hz, **self._p)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, **self._p)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, **self._p)
            p.resetDebugVisualizerCamera(0.75, 60, -35, (0.55, 0.0, 0.05), **self._p)
        self.plane = p.loadURDF("plane.urdf", **self._p)
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, **self._p)
        for j in FINGER_JOINTS:
            p.changeDynamics(self.robot, j, lateralFriction=1.0, spinningFriction=0.001, **self._p)
        h = self.cfg.object_half
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[h] * 3, **self._p)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[h] * 3, rgbaColor=(0.2, 0.45, 0.9, 1), **self._p)
        self.obj = p.createMultiBody(self.cfg.object_mass, col, vis, (0.5, -0.1, h), **self._p)
        tvis = p.createVisualShape(p.GEOM_CYLINDER, radius=self.cfg.target_radius, length=0.001,
                                   rgbaColor=(0.1, 0.8, 0.3, 0.5), **self._p)
        self.target_body = p.createMultiBody(0, -1, tvis, (0.5, 0.1, 0.0005), **self._p)
        self._label_id = None
        self.scenario: Scenario | None = None
        self.label = ""

    # -- episode lifecycle -----------------------------------------------------

    def reset(self, scenario: Scenario | int) -> np.ndarray:
        if not isinstance(scenario, Scenario):
            scenario = Scenario.sample(int(scenario), self.cfg)
        self.scenario = scenario
        cfg = self.cfg
        h = cfg.object_half
        p.changeDynamics(self.obj, -1, lateralFriction=scenario.friction, spinningFriction=0.001,
                         rollingFriction=0.0001, **self._p)
        p.resetBasePositionAndOrientation(self.obj, (*scenario.object_xy, h), (0, 0, 0, 1), **self._p)
        p.resetBaseVelocity(self.obj, (0, 0, 0), (0, 0, 0), **self._p)
        p.resetBasePositionAndOrientation(self.target_body, (*scenario.target_xy, 0.0005), (0, 0, 0, 1), **self._p)
        for j, q in zip(ARM_JOINTS, HOME_JOINTS):
            p.resetJointState(self.robot, j, q, **self._p)
        self.target_pos = np.array(cfg.start_ee, dtype=float)
        for j, q in zip(ARM_JOINTS, self._ik(self.target_pos)):
            p.resetJointState(self.robot, j, q, **self._p)
        for j in FINGER_JOINTS:
            p.resetJointState(self.robot, j, MAX_OPENING / 2, **self._p)
        self.cmd_opening, self.cmd_effort = MAX_OPENING, 10.0
        for _ in range(cfg.physics_hz // 4):                       # settle
            self._apply_motors()
            p.stepSimulation(**self._p)
        self.stats = EpisodeStats()
        self._held = False
        self._dwell = 0
        self._contact = (0.0, 0.0)
        self._table_force = 0.0
        self._prev_ee, self._prev_open = self._ee_pos(), self._opening()
        self.last_action = np.array([0, 0, 0, MAX_OPENING, 10.0], dtype=np.float32)
        self.done = False
        self._draw_label()
        return self.observe()

    def step(self, action) -> tuple[np.ndarray, bool, dict]:
        """Apply one tick of `action` (clipped). Returns (obs, done, info)."""
        if self.done:
            raise RuntimeError("step() after episode end; call reset()")
        cfg = self.cfg
        a = clip_action(action, cfg.max_speed)
        self.last_action = a
        ee = self._ee_pos()
        self.target_pos = self.target_pos + a[:3] * cfg.dt
        self.target_pos = np.clip(self.target_pos, ee - cfg.max_lead, ee + cfg.max_lead)
        for i, (lo, hi) in enumerate(cfg.workspace):
            self.target_pos[i] = min(hi, max(lo, self.target_pos[i]))
        self.cmd_opening, self.cmd_effort = float(a[3]), float(a[4])
        self._prev_ee, self._prev_open = ee, self._opening()

        self._apply_motors()
        fl = fr = table = 0.0
        for _ in range(cfg.substeps):
            p.stepSimulation(**self._p)
            l, r, t = self._contact_forces()
            fl, fr, table = fl + l, fr + r, table + t
        n = cfg.substeps
        self._contact = (fl / n, fr / n)
        self._table_force = table / n
        info = self._update_outcome()
        obs = self.observe()
        return obs, self.done, info

    def _update_outcome(self) -> dict:
        cfg, st = self.cfg, self.stats
        st.ticks += 1
        fl, fr = self._contact
        peak = max(fl, fr)
        st.peak_force_n = max(st.peak_force_n, peak)
        st.peak_table_force_n = max(st.peak_table_force_n, self._table_force)
        excess = peak > cfg.excess_force_n
        if excess:
            st.excess_force_ticks += 1
        obj, vel = self._obj_state()
        height = obj[2] - cfg.object_half
        in_target = math.dist(obj[:2], self.scenario.target_xy) <= cfg.target_radius
        touching = fl > CONTACT_EPS_N or fr > CONTACT_EPS_N
        held = fl > CONTACT_EPS_N and fr > CONTACT_EPS_N and height > 0.01
        if held:
            st.was_lifted = True
        if self._held and not held and not touching:
            if height > cfg.drop_height or not in_target:
                st.drops += 1
                st.events.append({"tick": st.ticks, "event": "drop", "height_m": round(height, 4)})
        self._held = held

        resting = height < 0.005 and np.linalg.norm(vel) < 0.02
        if st.was_lifted and in_target and resting and not touching:
            self._dwell += 1
        else:
            self._dwell = 0

        if self._dwell >= cfg.dwell_ticks:
            st.success, st.termination = True, "success"
            st.completion_time_s = st.ticks * cfg.dt
        elif obj[2] < -0.05 or not (0.2 < obj[0] < 0.9 and -0.45 < obj[1] < 0.45):
            st.termination = "object_out_of_bounds"
        elif st.ticks >= cfg.max_ticks:
            st.termination = "timeout"
        self.done = st.termination is not None
        return {"force_l": fl, "force_r": fr, "table_force": self._table_force, "excess_force": excess,
                "held": held, "in_target": in_target, "obj_pos": obj.tolist(), "termination": st.termination}

    # -- observation ------------------------------------------------------------

    def observe(self) -> np.ndarray:
        """Observation available *now*, before the next action. Order: OBS_NAMES."""
        ee, opening = self._ee_pos(), self._opening()
        dt = self.cfg.dt
        ee_vel = (ee - self._prev_ee) / dt
        open_vel = (opening - self._prev_open) / dt
        obj, obj_vel = self._obj_state()
        tgt = np.array(self.scenario.target_xy)
        fl, fr = self._contact
        obs = np.concatenate([
            ee, ee_vel, [opening, open_vel],
            obj - ee, obj_vel,
            tgt - ee[:2], tgt - obj[:2],
            [fl, fr, float(fl > CONTACT_EPS_N), float(fr > CONTACT_EPS_N)],
        ]).astype(np.float32)
        return obs

    def privileged_state(self) -> dict:
        """Ground truth for the scripted expert and for logging. Never fed to learned policies."""
        obj, vel = self._obj_state()
        return {"ee": self._ee_pos(), "opening": self._opening(), "obj": obj, "obj_vel": vel,
                "target_xy": np.array(self.scenario.target_xy), "force": self._contact,
                "friction": self.scenario.friction, "held": self._held}

    # -- internals ----------------------------------------------------------------

    def _ik(self, pos) -> list:
        q = p.calculateInverseKinematics(self.robot, EE_LINK, list(pos), EE_DOWN, maxNumIterations=100,
                                         residualThreshold=1e-5, **self._p)
        return list(q[:len(ARM_JOINTS)])

    def _apply_motors(self) -> None:
        p.setJointMotorControlArray(self.robot, ARM_JOINTS, p.POSITION_CONTROL,
                                    targetPositions=self._ik(self.target_pos), forces=[240.0] * 7, **self._p)
        for j in FINGER_JOINTS:
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=self.cmd_opening / 2,
                                    force=self.cmd_effort, maxVelocity=0.2, **self._p)

    def _ee_pos(self) -> np.ndarray:
        return np.array(p.getLinkState(self.robot, EE_LINK, computeForwardKinematics=True, **self._p)[4])

    def _opening(self) -> float:
        return float(sum(p.getJointState(self.robot, j, **self._p)[0] for j in FINGER_JOINTS))

    def _obj_state(self):
        pos, _ = p.getBasePositionAndOrientation(self.obj, **self._p)
        vel, _ = p.getBaseVelocity(self.obj, **self._p)
        return np.array(pos), np.array(vel)

    def _contact_forces(self) -> tuple[float, float, float]:
        f = [0.0, 0.0]
        for c in p.getContactPoints(self.robot, self.obj, **self._p):
            if c[3] in FINGER_JOINTS:
                f[FINGER_JOINTS.index(c[3])] += c[9]
        table = sum(c[9] for c in p.getContactPoints(self.robot, self.plane, **self._p))
        return f[0], f[1], table

    def set_label(self, text: str) -> None:
        """Mode banner in the GUI (e.g. AUTONOMOUS, REPLAY, KEYBOARD)."""
        self.label = text
        self._draw_label()

    def _draw_label(self) -> None:
        if not self.gui:
            return
        kw = dict(textColorRGB=(0.9, 0.1, 0.1), textSize=1.6, **self._p)
        if self._label_id is not None:
            kw["replaceItemUniqueId"] = self._label_id
        self._label_id = p.addUserDebugText(self.label or " ", (0.35, -0.35, 0.45), **kw)

    def close(self) -> None:
        if p.isConnected(self.client):
            p.disconnect(self.client)


def clip_action(action, max_speed: float = MAX_SPEED) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).reshape(5).copy()
    a = np.nan_to_num(a, nan=0.0)                       # a NaN command becomes "stop / closed / min effort"
    hi = ACTION_HIGH.copy()
    lo = ACTION_LOW.copy()
    hi[:3], lo[:3] = max_speed, -max_speed
    return np.clip(a, lo, hi)


def obs_indices(groups) -> list[int]:
    names = [n for g in groups for n in OBS_GROUPS[g]]
    return [OBS_NAMES.index(n) for n in names]
