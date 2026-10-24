"""Human-like 3-finger hand: a real-time picture of the glove's sensors (TELEOPERATION test, EXPERIMENTAL).

Each finger follows its own pod, so you can see every sensor move independently:

    thumb        <- thumb_tip  mag   baseline-subtracted change along one chip axis -> flex 0..1
    index finger <- index_tip  mag   (same gain / deadband / saturation as the arm mapping)
    middle finger<- middle_tip mag
    hand tilt    <- wrist IMU gravity direction (first usable of attitude_pods: wrist, then index_tip)
                    relative to its calibrated rest direction -> roll / pitch about the wrist.
                    Quasi-static only (|a| within tolerance of 1 g); no yaw.
    hand motion  <- MotionEstimator (motion.py): short IMU motion bursts such as a punch, with
                    spring-back to home. EXPERIMENTAL; IMUs cannot track position.
    finger colour = which pod (thumb blue, index orange, middle purple);
    fingertip pad colour = that pod's magnetometer state: green fresh, yellow held/STALE, red unusable.

Per-finger validity: a finger whose pod is unusable holds its last flex; the others keep moving.
The whole hand pauses only when no finger is usable or the watchdog fires (TeleopController).

Kinematics are human-like: palm down, fingers forward (+x). Index and middle have three
phalanges whose joints curl together (MCP 0-80 deg, PIP 0-95 deg, DIP 0-60 deg at full flex)
and never bend backwards; the thumb sits on the side and opposes toward the index.
The fingers are posed kinematically (joint angles set exactly from the smoothed command, no
contact forces that could shove a joint past its limit). The thumb is checked against the index
and middle finger geometry every tick and stops where it touches them, so digits never pass
through each other.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .controller import Command, TeleopController
from .mapping import LABEL, G, gravity_tilt

FINGERS = ("thumb_tip", "index_tip", "middle_tip")
# per-joint angle at flex = 1 (rad); flex 0 = straight. Index/middle: MCP, PIP, DIP. Thumb: CMC, MCP, IP.
FINGER_FLEX = (1.40, 1.65, 1.05)
THUMB_FLEX = (0.75, 0.80, 1.00)
FINGER_LEN = (0.046, 0.028, 0.022)
THUMB_LEN = (0.040, 0.030, 0.024)
FINGER_R = 0.0085
PALM_HALF = (0.045, 0.040, 0.013)    # x (length), y (width), z (thickness)
FINGER_BASE = {"index_tip": (0.045, 0.019, 0.0), "middle_tip": (0.045, -0.019, 0.0)}
THUMB_BASE = (-0.005, 0.040, -0.006)
THUMB_YAW, THUMB_DROOP = 0.75, 0.35  # thumb points forward-left and slightly down
FOREARM_R, FOREARM_L = 0.026, 0.11
WRIST_LOCAL = (-PALM_HALF[0] - 0.012, 0.0, 0.0)
HOME = (0.0, 0.0, 0.22)              # wrist position with zero motion offset
WRIST = HOME
SKIN, SKIN_DARK = (0.93, 0.78, 0.66, 1), (0.80, 0.62, 0.50, 1)
SLEEVE = (0.25, 0.27, 0.32, 1)
CLAW_COLOR = {"thumb_tip": (0.25, 0.45, 0.85, 1), "index_tip": (0.95, 0.55, 0.15, 1), "middle_tip": (0.6, 0.3, 0.75, 1)}
LINKS_PER_FINGER = 4                 # 3 revolute phalanges + fixed fingertip pad
CAMERA = (0.50, 200, -18, (0.13, 0.0, 0.21))     # distance, yaw, pitch, target: palm side, whole punch reach


def flex_limits(pod: str) -> tuple:
    return THUMB_FLEX if pod == "thumb_tip" else FINGER_FLEX


@dataclass
class HandConfig:
    grip_pods: tuple = FINGERS            # one claw per pod, in this order (name kept for Calibration)
    grip_axis: int = 2
    grip_sign: float = 1.0
    grip_gain_per_ut: float = 1.0 / 6000.0
    grip_deadband_ut: float = 150.0
    grip_saturation_ut: float = 8000.0
    attitude_pods: tuple = ("wrist", "index_tip")
    optional_imu_pods: tuple = ("wrist", "index_tip")   # baselines captured at calibration when usable
    max_tilt_rad: float = 0.6
    gravity_tolerance: float = 0.15

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = LABEL
        return d


@dataclass
class HandOutput:
    usable: bool
    reason: str | None
    flex: list                            # per claw 0..1, or None = hold (that pod unusable)
    attitude: tuple | None                # (roll, pitch) rad, or None = hold
    signals: dict


class HandMapping:
    label = f"{LABEL}: 3-finger hand (per-pod magnetic change -> finger flex, IMU -> hand tilt + motion bursts)"

    def __init__(self, cfg: HandConfig | None = None):
        self.cfg = cfg or HandConfig()

    def required_channels(self) -> list[tuple[str, str]]:
        return [("mag", p) for p in self.cfg.grip_pods]          # all needed to calibrate

    def _mag_ok(self, snap, pod, fresh: bool) -> str | None:
        if pod not in snap.pods:
            return f"{pod} not in pod order"
        i = snap.pods.index(pod)
        if not snap.present[i]:
            return f"{pod} missing"
        if fresh and not snap.mag_fresh[i]:
            return f"{pod} mag not fresh"
        if not snap.mag_held_mask[i] or not np.all(np.isfinite(snap.mag_held[i])):
            return f"{pod} mag unusable ({'STALE too long' if snap.mag_stale[i] else 'invalid/null'})"
        return None

    def check_channels(self, snap, require_fresh: bool = False) -> str | None:
        for pod in self.cfg.grip_pods:
            why = self._mag_ok(snap, pod, require_fresh)
            if why:
                return why
        return None

    def evaluate(self, snap, calib) -> HandOutput:
        c = self.cfg
        if calib is None:
            return HandOutput(False, "not calibrated", [None] * len(c.grip_pods), None, {})
        flex, status, held = [], {}, []
        for pod in c.grip_pods:
            why = self._mag_ok(snap, pod, False)
            if why or pod not in calib.mag_baseline_ut:
                flex.append(None)
                status[pod] = "unusable"
                held.append(pod)
                continue
            i = snap.pods.index(pod)
            delta = float(snap.mag_held[i][c.grip_axis] - calib.mag_baseline_ut[pod][c.grip_axis])
            drive = max(0.0, min(c.grip_sign * delta - c.grip_deadband_ut, c.grip_saturation_ut))
            flex.append(max(0.0, min(1.0, c.grip_gain_per_ut * drive)))
            status[pod] = "fresh" if snap.mag_fresh[i] else "held"
        signals = {"claw": {p: (None if f is None else round(f, 2)) for p, f in zip(c.grip_pods, flex)},
                   "status": status}
        if held:
            signals["held"] = held
        attitude = None
        for pod in c.attitude_pods:
            if pod not in snap.pods or pod not in calib.acc_baseline_mps2:
                continue
            i = snap.pods.index(pod)
            if not snap.imu_held_mask[i] or not np.all(np.isfinite(snap.acc_held[i])):
                continue
            acc = snap.acc_held[i]
            if abs(float(np.linalg.norm(acc)) - G) > c.gravity_tolerance * G:
                signals["tilt"] = f"{pod}: not quasi-static, palm held"
                break
            pitch, roll = gravity_tilt(acc, np.asarray(calib.acc_baseline_mps2[pod]))
            lim = c.max_tilt_rad
            attitude = (max(-lim, min(lim, roll)), max(-lim, min(lim, pitch)))
            signals["palm_from"] = pod
            signals["roll_pitch"] = [round(attitude[0], 3), round(attitude[1], 3)]
            break
        if all(f is None for f in flex):
            return HandOutput(False, "no claw pod usable: " + ", ".join(held), flex, attitude, signals)
        return HandOutput(True, None, flex, attitude, signals)


class HandController(TeleopController):
    """Same arm/disarm, watchdog, calibration and reconnect rules; the command is finger flex,
    hand tilt and (with a MotionEstimator) a bounded motion offset. Not ARMED: everything holds.
    Flex, tilt and offset are low-passed and rate-limited."""

    max_flex_rate = 2.5        # flex units (0..1) per second
    flex_tau_s = 0.06
    max_tilt_rate = 1.5        # rad/s
    tilt_tau_s = 0.10
    max_hand_speed = 3.0       # m/s: fast enough for a punch, still bounded
    offset_tau_s = 0.02

    def __init__(self, mapping: HandMapping, *a, motion=None, **kw):
        super().__init__(mapping, *a, **kw)
        self.flex = np.zeros(len(mapping.cfg.grip_pods))
        self.att = np.zeros(2)
        self.offset = np.zeros(3)
        self.motion = motion

    @staticmethod
    def _approach(cur, want, tau, rate, dt):
        k = min(1.0, dt / max(tau, dt))
        return cur + np.clip(k * (want - cur), -rate * dt, rate * dt)

    def tick(self, *a, **kw) -> Command:
        if self.motion is not None and self.calib is not None and self.motion.reference() is not self.calib:
            self.motion.set_reference(self.calib)       # new calibration = new 'up' and home
        return super().tick(*a, **kw)

    def _drive(self, out: HandOutput, dt: float, z_input: float) -> None:
        for i, f in enumerate(out.flex):
            if f is not None:
                self.flex[i] = float(np.clip(self._approach(self.flex[i], f, self.flex_tau_s, self.max_flex_rate, dt), 0, 1))
        if out.attitude is not None:
            lim = self.mapping.cfg.max_tilt_rad
            self.att = np.clip(self._approach(self.att, np.array(out.attitude), self.tilt_tau_s, self.max_tilt_rate, dt),
                               -lim, lim)
        if self.motion is not None:
            want = np.clip(self.motion.offset(), self.motion.limits[:, 0], self.motion.limits[:, 1])
            self.offset = self._approach(self.offset, want, self.offset_tau_s, self.max_hand_speed, dt)

    def _hold(self) -> None:
        pass                                              # keep flex, tilt and position exactly where they are

    def _command(self) -> Command:
        return Command(np.zeros(3), 0.0, self.state, self.reason,
                       {"flex": self.flex.tolist(), "attitude": self.att.tolist(), "offset": self.offset.tolist()})




class HandSim:
    """PyBullet human-like hand + forearm, posed kinematically. apply(cmd) sets the pose; measure() reads
    back joint angles, hand tilt and wrist position from PyBullet."""

    def __init__(self, gui: bool = False, physics_hz: int = 240, control_hz: int = 30, pods=FINGERS):
        import pybullet as p
        import pybullet_data
        self.p = p
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        k = dict(physicsClientId=self.client)
        self._k = k
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), **k)
        p.setGravity(0, 0, -9.81, **k)
        p.setTimeStep(1.0 / physics_hz, **k)
        self.substeps = physics_hz // control_hz
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, **k)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, **k)
            p.resetDebugVisualizerCamera(*CAMERA, **k)
        p.loadURDF("plane.urdf", **k)
        self.pods = tuple(pods)
        self.body = self._build()
        self.joints = [LINKS_PER_FINGER * n + j for n in range(len(self.pods)) for j in range(3)]
        self.limits = [lim for pod in self.pods for lim in flex_limits(pod)]
        self._tip_color = {}
        self._thumb_links = self._links("thumb_tip") if "thumb_tip" in self.pods else []
        self._finger_links = [l for pod in self.pods if pod != "thumb_tip" for l in self._links(pod)]
        self.thumb_scale = 1.0              # < 1 while the thumb is stopped by a finger
        self._set_joints([0.0] * len(self.joints))
        self._home_marker()
        self._pose(np.zeros(3), 0.0, 0.0)

    def _home_marker(self) -> None:
        p, k = self.p, self._k
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.002, rgbaColor=(0.2, 0.6, 1.0, 0.35), **k)
        p.createMultiBody(0, -1, vis, (HOME[0], HOME[1], 0.001), **k)        # floor dot under home

    def _build(self) -> int:
        p, k = self.p, self._k
        qx = p.getQuaternionFromEuler((0, math.pi / 2, 0))                    # capsule/cylinder z -> x
        palm_c = p.createCollisionShape(p.GEOM_BOX, halfExtents=PALM_HALF, **k)
        palm_v = p.createVisualShapeArray(
            shapeTypes=[p.GEOM_BOX, p.GEOM_CYLINDER, p.GEOM_SPHERE],
            halfExtents=[PALM_HALF, (0, 0, 0), (0, 0, 0)], radii=[0, FOREARM_R, FOREARM_R * 1.15],
            lengths=[0, FOREARM_L, 0],
            visualFramePositions=[(0, 0, 0), (WRIST_LOCAL[0] - FOREARM_L / 2, 0, 0), WRIST_LOCAL],
            visualFrameOrientations=[(0, 0, 0, 1), qx, (0, 0, 0, 1)], **k)
        masses, cols, vis, pos, orn, inert_p, inert_o, parents, jtypes, axes = ([] for _ in range(10))
        for pod in self.pods:
            thumb = pod == "thumb_tip"
            lens = THUMB_LEN if thumb else FINGER_LEN
            if thumb:
                base = THUMB_BASE
                base_orn = p.getQuaternionFromEuler((0.0, THUMB_DROOP, THUMB_YAW))
                axis = (0.0, 0.35, -1.0)            # curls across the palm toward the index, slightly down
            else:
                base = FINGER_BASE.get(pod, (0.045, 0.0, 0.0))
                base_orn = (0, 0, 0, 1)
                axis = (0.0, 1.0, 0.0)              # +angle curls the finger down toward the palm
            for seg, L in enumerate(lens):
                r = FINGER_R * (1.1 if thumb else 1.0) * (1.0 - 0.08 * seg)
                cols.append(p.createCollisionShape(p.GEOM_CAPSULE, radius=r, height=L - r,
                                                   collisionFramePosition=(L / 2, 0, 0),
                                                   collisionFrameOrientation=qx, **k))
                vis.append(p.createVisualShapeArray(
                    shapeTypes=[p.GEOM_SPHERE, p.GEOM_CAPSULE], radii=[r * 1.12, r], lengths=[0, L],
                    visualFramePositions=[(0, 0, 0), (L / 2, 0, 0)],
                    visualFrameOrientations=[(0, 0, 0, 1), qx], **k))
                masses.append(0.02)
                pos.append(base if seg == 0 else (lens[seg - 1], 0, 0))
                orn.append(base_orn if seg == 0 else (0, 0, 0, 1))
                parents.append(0 if seg == 0 else len(masses) - 1)
                inert_p.append((L / 2, 0, 0))
                inert_o.append((0, 0, 0, 1))
                jtypes.append(p.JOINT_REVOLUTE)
                axes.append(axis)
            cols.append(-1)                                              # fingertip pad: status colour
            vis.append(p.createVisualShape(p.GEOM_SPHERE, radius=FINGER_R * 0.85, **k))
            masses.append(0.001)
            pos.append((lens[-1], 0, 0))
            orn.append((0, 0, 0, 1))
            parents.append(len(masses) - 1)
            inert_p.append((0, 0, 0))
            inert_o.append((0, 0, 0, 1))
            jtypes.append(p.JOINT_FIXED)
            axes.append((0, 0, 1))
        body = p.createMultiBody(0, palm_c, palm_v, HOME, (0, 0, 0, 1), linkMasses=masses,
                                 linkCollisionShapeIndices=cols, linkVisualShapeIndices=vis, linkPositions=pos,
                                 linkOrientations=orn, linkInertialFramePositions=inert_p,
                                 linkInertialFrameOrientations=inert_o, linkParentIndices=parents,
                                 linkJointTypes=jtypes, linkJointAxis=axes, **k)
        for i, c in enumerate((SKIN, SLEEVE, SKIN_DARK)):
            p.changeVisualShape(body, -1, shapeIndex=i, rgbaColor=c, **k)
        for n, pod in enumerate(self.pods):
            col = CLAW_COLOR.get(pod, SKIN)
            for seg in range(3):
                link = LINKS_PER_FINGER * n + seg
                p.changeVisualShape(body, link, shapeIndex=0, rgbaColor=SKIN_DARK, **k)      # knuckle
                shade = tuple(c + (1 - c) * 0.25 * seg for c in col[:3]) + (1,)
                p.changeVisualShape(body, link, shapeIndex=1, rgbaColor=shade, **k)
                p.changeDynamics(body, link, jointLowerLimit=0.0, jointUpperLimit=flex_limits(pod)[seg] + 0.05,
                                 lateralFriction=1.0, **k)
            p.changeVisualShape(body, LINKS_PER_FINGER * n + 3, rgbaColor=(0.6, 0.6, 0.6, 1), **k)
        return body

    def _links(self, pod) -> list[int]:
        n = self.pods.index(pod)
        return [LINKS_PER_FINGER * n + j for j in range(3)]

    def _penetration(self) -> float:
        """Deepest overlap (m, negative) between the thumb and the other fingers in the current pose."""
        p, k = self.p, self._k
        p.performCollisionDetection(**k)
        worst = 0.0
        for a in self._thumb_links:
            for b in self._finger_links:
                for c in p.getClosestPoints(self.body, self.body, 0.0, linkIndexA=a, linkIndexB=b, **k):
                    worst = min(worst, c[8])
        return worst

    def _set_joints(self, targets) -> None:
        for j, q in zip(self.joints, targets):
            self.p.resetJointState(self.body, j, q, **self._k)

    def _joint_targets(self, flex) -> list[float]:
        return [f * lim for pod, f in zip(self.pods, flex) for lim in flex_limits(pod)]

    def _pose(self, offset, roll: float, pitch: float) -> None:
        """Hand tilt (roll about the hand axis x, pitch about y) is applied about the wrist; the wrist
        sits at HOME + motion offset."""
        p, k = self.p, self._k
        q = p.getQuaternionFromEuler((roll, pitch, 0.0))
        R = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
        wrist = np.array(HOME) + np.asarray(offset)
        pos = wrist - R @ np.array(WRIST_LOCAL)
        p.resetBasePositionAndOrientation(self.body, pos.tolist(), q, **k)
        self._wrist = wrist

    def apply(self, cmd: Command) -> None:
        p, k = self.p, self._k
        flex = cmd.extra.get("flex", [0.0] * len(self.pods))
        roll, pitch = cmd.extra.get("attitude", [0.0, 0.0])
        self._pose(cmd.extra.get("offset", [0.0, 0.0, 0.0]), roll, pitch)
        flex = list(flex)
        self._set_joints(self._joint_targets(flex))
        self.thumb_scale = 1.0
        if self._thumb_links and self._finger_links and self._penetration() < -0.0005:
            ti = self.pods.index("thumb_tip")
            lo, hi, want = 0.0, 1.0, flex[ti]
            for _ in range(7):                      # largest thumb flex that only touches the fingers
                mid = 0.5 * (lo + hi)
                flex[ti] = want * mid
                self._set_joints(self._joint_targets(flex))
                lo, hi = (mid, hi) if self._penetration() >= -0.0005 else (lo, mid)
            flex[ti] = want * lo
            self.thumb_scale = lo
            self._set_joints(self._joint_targets(flex))

    def measure(self) -> dict:
        p, k = self.p, self._k
        q = [p.getJointState(self.body, j, **k)[0] for j in self.joints]
        pos, orn = p.getBasePositionAndOrientation(self.body, **k)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        wrist = np.array(pos) + R @ np.array(WRIST_LOCAL)
        return {"joints": q, "palm_rp": [roll, pitch], "wrist": wrist.tolist(),
                "offset": (wrist - np.array(HOME)).tolist()}

    def show_status(self, status: dict) -> None:
        """Colour each fingertip pad by its pod's magnetometer state (cheap; GUI only)."""
        if not self.gui:
            return
        colors = {"fresh": (0.1, 0.85, 0.2, 1), "held": (1.0, 0.8, 0.0, 1), "unusable": (0.9, 0.1, 0.1, 1)}
        for n, pod in enumerate(self.pods):
            c = colors.get(status.get(pod), (0.6, 0.6, 0.6, 1))
            if self._tip_color.get(pod) != c:
                self.p.changeVisualShape(self.body, LINKS_PER_FINGER * n + 3, rgbaColor=c, **self._k)
                self._tip_color[pod] = c

    def close(self) -> None:
        if self.p.isConnected(self.client):
            self.p.disconnect(self.client)


ClawHandSim = HandSim          # earlier name
