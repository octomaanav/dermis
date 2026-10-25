"""Real-time pick-up: a judge wearing the glove grasps a real cup on a real table, and the simulated
three-finger hand picks up a simulated cup from a simulated table.

States:
    OPEN -> grasp detected (grip >= on_n for on_frames) -> CLOSING -> GRIPPED (cup still on the table)
    GRIPPED: the first settle_s record the IMU at rest; then any IMU motion (|acc| leaves its settled value
             by imu_acc_dev, or gyro by imu_gyro) means the real hand is moving the cup -> LIFTING to lift_m
             -> HELD. With no valid IMU the lift starts after auto_lift_s instead.
    HELD: a motion that starts with |acc| BELOW its settled value (the dip at the start of a set-down)
          -> SETTING_DOWN -> GRIPPED. Other motion keeps the cup up.
    grasp released (strongest pod below max(off_ratio, release_frac x this grasp's peak, capped at 0.8 on_ratio)
    for off_frames) -> LOWERING (if lifted) -> OPENING -> OPEN.
A grasp (combined mode) needs the strongest pod >= on_ratio x the calibrated normal hold with a second pod
>= second_ratio, or one pod >= solo_ratio on its own (a loose glove often loads only one pad firmly).
The IMU is never integrated (no velocity or position): only "moving or still" and the sign of the first
acceleration deviation. The lift height is therefore fixed, not tracked.

Grip (grip_mode):
    'combined' (default): grip = mean of the two strongest pods (any of thumb, index, middle, palm) and all
               three simulated digits close with grip_effort(grip): a curve that gives any recognised grasp
               a safe hold, grows gently above the calibrated normal hold, and reaches the simulated crack
               level only beyond ~4x it. Works when a loose glove loads the pods unevenly.
    'per_finger': each pod drives its own digit (thumb_tip -> thumb, index_tip -> index, middle_tip -> middle).
Either way the simulated cup is held only by friction from those efforts. In combined mode a grip too light
to be recognised never closes the hand, and a crushing one passes the SIMULATED crack threshold. While a release is being confirmed and while LOWERING,
the efforts are held at the strongest grip of the last 0.3 s, because the judge lets go of the real cup
only after it is back on the real table.

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
    grip_gain: float = 0.027             # linear effort per N of grip: a 2 N grip = effort_ref (venue calibration overrides)
    effort_min: float = 0.005
    effort_max: float = 0.4
    open_effort: float = 0.05
    on_n: float = 1.0                    # per_finger mode thresholds (N)
    off_n: float = 0.5
    on_frames: int = 15                  # a grasp must hold for 0.3 s (one glove frame is 0.1 s)
    off_frames: int = 5
    crack_force_n: float | None = 16.0   # SIMULATED threshold; None disables
    drop_m: float = 0.03
    reset_after_s: float = 1.5
    grip_mode: str = "combined"          # 'combined' | 'per_finger'
    # combined mode thresholds, as fractions of the calibrated normal hold (effort_ref / grip_gain), so they
    # mean the same on every glove and calibration: a grasp starts when the strongest pod passes on_ratio with
    # a second pod past second_ratio, or ONE pod passes solo_ratio on its own
    on_ratio: float = 0.12
    second_ratio: float = 0.05
    solo_ratio: float = 0.5
    off_ratio: float = 0.06
    release_frac: float = 0.3            # combined: released below this fraction of the grasp's peak grip
    # combined: grip effort follows a curve of r = grip_gain * grip / effort_ref (r = 1 is the calibrated
    # normal hold), measured window for this cup: <= 0.02 N*m drops, 0.025-0.08 holds, >= 0.09 can crack
    effort_ref: float = 0.054            # linear effort the venue calibration assigns to a normal hold
    effort_floor: float = 0.04           # any recognised grasp holds the cup, however light
    effort_normal: float = 0.05          # r = 1
    effort_safe: float = 0.075           # r = strong_ratio: still below the crack level
    strong_ratio: float = 3.5
    crack_slope: float = 0.05            # N*m per unit r beyond strong_ratio: cracks at about 4x the normal hold
    lift_mode: str = "imu"               # 'imu' (motion starts the lift) | 'auto' (lift after auto_lift_s)
    settle_s: float = 0.5                # IMU rest reading after the grip closes
    imu_acc_dev: float = 0.8             # m/s^2 change of |acc| from rest that counts as motion
    imu_gyro: float = 0.5                # rad/s change of gyro from rest that counts as motion
    motion_s: float = 0.15               # motion must last this long (filters single noisy frames)
    still_s: float = 0.4                 # quiet this long ends a motion
    auto_lift_s: float = 1.0

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


def combined_grip(forces: dict) -> float:
    """Mean of the two strongest pods: robust to a glove that loads some pods poorly."""
    v = sorted((x or 0.0 for x in forces.values()), reverse=True)
    return float(np.mean(v[:2])) if v else 0.0


class ImuMotion:
    """Moving / still from the IMU relative to its reading at rest (no integration)."""

    def __init__(self, cfg: PickupConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.rest, self.moving, self.fast, self.quiet, self.first_sign = [], False, 0, 0, 0
        self.acc0 = self.gyr0 = None

    def learn(self, imu: dict | None):
        if imu is not None:
            self.rest.append((float(np.linalg.norm(imu["acc"])), np.asarray(imu["gyr"], float)))

    def freeze(self) -> bool:
        if len(self.rest) < 3:
            return False
        self.acc0 = float(np.median([r[0] for r in self.rest]))
        self.gyr0 = np.median([r[1] for r in self.rest], axis=0)
        return True

    def update(self, imu: dict | None) -> tuple[bool, int]:
        """(moving, sign of the first |acc| deviation of this motion: +1 above rest, -1 below, 0 unknown)."""
        c, n = self.cfg, self.cfg.control_hz
        if imu is None or self.acc0 is None:
            return self.moving, self.first_sign
        dev = float(np.linalg.norm(imu["acc"])) - self.acc0
        rot = float(np.linalg.norm(np.asarray(imu["gyr"], float) - self.gyr0))
        hot = abs(dev) > c.imu_acc_dev or rot > c.imu_gyro
        if hot:
            if self.fast == 0 and not self.moving:
                self.first_sign = int(np.sign(dev)) if abs(dev) > c.imu_acc_dev else 0
            self.fast, self.quiet = self.fast + 1, 0
            if self.fast >= max(1, int(c.motion_s * n)):
                self.moving = True
        else:
            self.quiet += 1
            if not self.moving:
                self.fast = 0
            elif self.quiet >= int(c.still_s * n):
                self.moving, self.fast = False, 0
        return self.moving, self.first_sign


class GripTrigger:
    """Pick-up state machine. update() takes per-pod glove forces (N; None = pod invalid; forces=None =
    stream stale) and the IMU ({'acc', 'gyr'} or None) and returns (digit efforts or None, lift fraction)."""

    GRIPPING = ("CLOSING", "GRIPPED", "LIFTING", "HELD", "SETTING_DOWN")

    def __init__(self, cfg: PickupConfig):
        self.cfg = cfg
        self.events = []                                     # kept across resets
        self.imu = ImuMotion(cfg)
        self.reset()

    def reset(self):
        self.state, self.t_state, self.lift, self.on, self.off = "OPEN", 0.0, 0.0, 0, 0
        self.frozen = None
        self.recent = deque(maxlen=max(1, int(0.3 * self.cfg.control_hz)))
        self.efforts = None
        self.grip = self.peak = 0.0
        self.imu.reset()
        self.armed = False                                   # IMU rest reading taken (None: no IMU)
        self.need_still = False

    def _go(self, s: str, t: float, **kw):
        self.state, self.t_state = s, t
        self.events.append({"t": round(t, 3), "event": s, **kw})

    def grip_effort(self, grip: float) -> float:
        """Combined mode: wide window. Light grasps still hold, strong ones stay safe, only a squeeze several
        times the calibrated normal hold reaches the simulated crack level."""
        c = self.cfg
        r = c.grip_gain * grip / c.effort_ref
        if r <= 1.0:
            e = c.effort_floor + (c.effort_normal - c.effort_floor) * r
        elif r <= c.strong_ratio:
            e = c.effort_normal + (c.effort_safe - c.effort_normal) * (r - 1.0) / (c.strong_ratio - 1.0)
        else:
            e = c.effort_safe + c.crack_slope * (r - c.strong_ratio)
        return float(np.clip(e, c.effort_min, c.effort_max))

    def live_efforts(self, forces: dict) -> dict:
        c = self.cfg
        if c.grip_mode == "combined":
            e = self.grip_effort(combined_grip(forces))
            return {d: e for d in DIGIT_POD}
        return {d: float(np.clip(c.grip_gain * (forces.get(pod) or 0.0), c.effort_min, c.effort_max)) for d, pod in DIGIT_POD.items()}

    def _grasp_released(self, f: dict) -> tuple[bool, bool]:
        c = self.cfg
        if c.grip_mode == "combined":
            normal = c.effort_ref / c.grip_gain                  # the calibrated normal hold, in grip units
            v = sorted((x or 0.0 for x in f.values()), reverse=True) + [0.0, 0.0]
            top, second = v[0] / normal, v[1] / normal
            # release relative to this grasp's peak (a glove that does not spring fully back still lets go),
            # but never demand more than 0.8 * on_ratio to stay gripped (relaxing from a hard squeeze is not a release)
            off = max(c.off_ratio, min(c.release_frac * self.peak / normal, 0.8 * c.on_ratio))
            return (top >= c.on_ratio and second >= c.second_ratio) or top >= c.solo_ratio, top < off
        fingers = max(f.get("index_tip", 0), f.get("middle_tip", 0))
        return (f.get("thumb_tip", 0) >= c.on_n and fingers >= c.on_n), (f.get("thumb_tip", 0) < c.off_n or fingers < c.off_n)

    def update(self, forces: dict | None, t: float, outcome: str | None = None, imu: dict | None = None):
        c, dtc = self.cfg, 1.0 / self.cfg.control_hz
        if forces is None:                                   # stale stream: keep the last command, change nothing
            return self.efforts, self.lift
        f = {k: (v or 0.0) for k, v in forces.items()}
        self.grip = combined_grip(f)
        top = max((v or 0.0 for v in f.values()), default=0.0)
        self.peak = max(self.peak, top) if self.state in self.GRIPPING else 0.0
        grasp, released = self._grasp_released(f)
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
        elif s in self.GRIPPING:
            self.recent.append(self.live_efforts(f))
            if self.off >= c.off_frames:
                self.frozen = {d: max(e[d] for e in self.recent) for d in DIGIT_POD}
                self._go("LOWERING" if self.lift > 0 else "OPENING", t)
            elif s == "CLOSING" and el >= c.close_s:
                self._go("GRIPPED", t)
                self.imu.reset()
                self.armed = False
            elif s == "GRIPPED":
                if self.armed is False:                   # still recording the IMU rest reading
                    self.imu.learn(imu)
                    if el >= c.settle_s:
                        self.armed = self.imu.freeze()
                        self.events.append({"t": round(t, 3), "event": "imu_rest" if self.armed else "imu_unavailable"})
                        if not self.armed:
                            self.armed = None            # no IMU: fall back to the timed lift
                elif self.armed is None or c.lift_mode == "auto":
                    if el >= c.settle_s + c.auto_lift_s:
                        self._go("LIFTING", t, trigger="timer")
                else:
                    moving, _ = self.imu.update(imu)
                    if self.need_still:
                        self.need_still = moving
                    elif moving:
                        self._go("LIFTING", t, trigger="imu_motion")
            elif s == "LIFTING":
                self.lift = min(1.0, self.lift + dtc / c.lift_s)
                if self.armed:
                    self.imu.update(imu)
                if self.lift >= 1.0:
                    self._go("HELD", t)
            elif s == "HELD":
                if self.armed:
                    was = self.imu.moving
                    moving, sign = self.imu.update(imu)
                    if moving and not was and sign < 0:
                        self._go("SETTING_DOWN", t, trigger="imu_motion_down")
            elif s == "SETTING_DOWN":
                self.lift = max(0.0, self.lift - dtc / c.lower_s)
                if self.armed:
                    self.imu.update(imu)
                if self.lift <= 0.0:
                    self._go("GRIPPED", t)                # keeps the IMU rest reading (armed)
                    self.need_still = True                # the set-down's own motion must end first
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
        if s in self.GRIPPING:
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

    def tick(self, forces: dict | None, imu: dict | None = None) -> dict:
        sim, trig = self.sim, self.trig
        outcome = None
        if trig.state in GripTrigger.GRIPPING + ("LOWERING",):
            if sim.cracked:
                outcome = "CRACKED"
            elif sim.lift > 0.2 and sim.cup_lag_m() > self.cfg.drop_m:
                outcome = "DROPPED"
        efforts, lift = trig.update(forces, self.t, outcome, imu)
        if trig.state == "RESET":
            sim.reset()
            trig.reset()
            self.resets += 1
            trig.events.append({"t": round(self.t, 3), "event": "reset"})
            efforts, lift = None, 0.0
        sim.step(efforts, lift)
        self.t += 1.0 / self.cfg.control_hz
        return {"t": round(self.t, 3), "state": trig.state, "efforts": efforts, "forces": forces, "grip": round(trig.grip, 3),
                "imu_moving": trig.imu.moving, **sim.state()}
