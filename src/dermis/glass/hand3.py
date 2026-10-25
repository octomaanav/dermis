"""Three-finger hand for the glass demo (SIMULATED). Thumb, index and middle each have three phalanges that
curl in the horizontal plane around a vertical glass; the thumb opposes the fingers from the other side,
and a thenar pad on the palm (where the glove's thenar_apb pod sits) presses on the glass.

Built in this package (not the teleop HandSim) because that hand's thumb moves along a cup's axis and
cannot oppose the fingers radially (docs/glass_demo.md). Joints are driven by force-limited position
motors (no explicit joint damping: the phalanges' tiny inertia makes explicit damping unstable at 240 Hz;
speed capped at MAX_JOINT_SPEED).

Frame: palm plate in the x-z plane, palm face toward +y; glass centred at (0, glass_y, Z0) against the
thenar pad. All digits leave the palm pointing +y, along the sides of the glass (index upper and middle
lower at +x, thumb at -x); flexion about +z (fingers) / -z (thumb) curls them inward, so the proximal
phalanges press the sides and the distal phalanges wrap behind the glass and push it back into the palm:
an enveloping grasp rather than a pinch at the equator (a pinch let the glass walk out whenever the grip
effort changed). Link order: thumb 0-2, thumb tip pad 3, index 4-6, pad 7, middle 8-10, pad 11,
thenar pad 12 (fixed, on the palm).
"""
from __future__ import annotations

import pybullet as p

DIGITS = ("thumb_tip", "index_tip", "middle_tip")
PODS = DIGITS + ("thenar_apb",)
Z0 = 0.20
PALM_HALF = (0.045, 0.006, 0.05)
THENAR_HALF = (0.020, 0.004, 0.028)
THENAR_POS = (-0.020, PALM_HALF[1] + THENAR_HALF[1], -0.004)      # pad proud of the palm surface
FINGER_LEN = (0.050, 0.030, 0.024)
THUMB_LEN = (0.050, 0.028, 0.022)
SEG_R = 0.0085
ROOTS = {"thumb_tip": (-0.044, 0.006, 0.0), "index_tip": (0.044, 0.006, 0.020), "middle_tip": (0.044, 0.006, -0.020)}
FLEX_LIMIT = 1.45                     # rad per joint
MAX_JOINT_SPEED = 4.0
LINKS_PER_DIGIT = 4
THENAR_LINK = 12
COLORS = {"thumb_tip": (0.25, 0.45, 0.85, 1), "index_tip": (0.95, 0.55, 0.15, 1), "middle_tip": (0.6, 0.3, 0.75, 1)}


def build(client: int, base_pos=(0.0, 0.0, Z0), visual: bool = True, base_mass: float = 0.0) -> tuple[int, list, list]:
    """Create the hand in `client`. Returns (body, revolute joint indices, joint limits). base_mass > 0 makes the
    palm dynamic so it can be carried by a force-limited constraint (pick-up demo); 0 keeps it fixed."""
    P = dict(physicsClientId=client)
    qx = p.getQuaternionFromEuler((0, 1.5707963, 0))               # capsule z -> x
    q_root = p.getQuaternionFromEuler((0, 0, 1.5707963))           # digit x-axis -> +y (out of the palm)
    palm_c = p.createCollisionShape(p.GEOM_BOX, halfExtents=PALM_HALF, **P)
    palm_v = p.createVisualShape(p.GEOM_BOX, halfExtents=PALM_HALF, rgbaColor=(0.93, 0.78, 0.66, 1), **P) if visual else -1
    masses, cols, vis, pos, orn, ip, io, parents, jt, axes = ([] for _ in range(10))
    for d in DIGITS:
        thumb = d == "thumb_tip"
        lens = THUMB_LEN if thumb else FINGER_LEN
        for s, L in enumerate(lens):
            cols.append(p.createCollisionShape(p.GEOM_CAPSULE, radius=SEG_R, height=L - SEG_R,
                                               collisionFramePosition=(L / 2, 0, 0), collisionFrameOrientation=qx, **P))
            vis.append(p.createVisualShape(p.GEOM_CAPSULE, radius=SEG_R, length=L - SEG_R, visualFramePosition=(L / 2, 0, 0),
                                           visualFrameOrientation=qx, rgbaColor=COLORS[d], **P) if visual else -1)
            masses.append(0.02)
            pos.append(ROOTS[d] if s == 0 else (lens[s - 1], 0, 0))
            orn.append(q_root if s == 0 else (0, 0, 0, 1))
            parents.append(0 if s == 0 else len(masses) - 1)
            ip.append((L / 2, 0, 0))
            io.append((0, 0, 0, 1))
            jt.append(p.JOINT_REVOLUTE)
            axes.append((0, 0, -1) if thumb else (0, 0, 1))      # both curl inward, behind the glass
        cols.append(-1)                                           # fingertip pad marker (visual only)
        vis.append(p.createVisualShape(p.GEOM_SPHERE, radius=SEG_R * 0.9, rgbaColor=(0.6, 0.6, 0.6, 1), **P) if visual else -1)
        masses.append(0.001)
        pos.append((lens[-1], 0, 0))
        orn.append((0, 0, 0, 1))
        parents.append(len(masses) - 1)
        ip.append((0, 0, 0))
        io.append((0, 0, 0, 1))
        jt.append(p.JOINT_FIXED)
        axes.append((0, 0, 1))
    cols.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=THENAR_HALF, **P))     # thenar pad (sensor region)
    vis.append(p.createVisualShape(p.GEOM_BOX, halfExtents=THENAR_HALF, rgbaColor=(0.8, 0.6, 0.5, 1), **P) if visual else -1)
    masses.append(0.005 if base_mass > 0 else 0.0)                 # massless links destabilise a moving base
    pos.append(THENAR_POS)
    orn.append((0, 0, 0, 1))
    parents.append(0)
    ip.append((0, 0, 0))
    io.append((0, 0, 0, 1))
    jt.append(p.JOINT_FIXED)
    axes.append((0, 0, 1))
    body = p.createMultiBody(base_mass, palm_c, palm_v, base_pos, (0, 0, 0, 1), linkMasses=masses, linkCollisionShapeIndices=cols,
                             linkVisualShapeIndices=vis, linkPositions=pos, linkOrientations=orn,
                             linkInertialFramePositions=ip, linkInertialFrameOrientations=io, linkParentIndices=parents,
                             linkJointTypes=jt, linkJointAxis=axes, **P)
    joints = [n * LINKS_PER_DIGIT + s for n in range(3) for s in range(3)]
    for j in range(p.getNumJoints(body, **P)):
        p.changeDynamics(body, j, jointDamping=0.0, lateralFriction=1.0, **P)
    for j in joints:
        p.changeDynamics(body, j, jointLowerLimit=0.0, jointUpperLimit=FLEX_LIMIT, **P)
    p.changeDynamics(body, -1, lateralFriction=1.0, **P)
    return body, joints, [FLEX_LIMIT] * len(joints)


def digit_of_link(link: int) -> str | None:
    if link == THENAR_LINK:
        return "thenar_apb"
    if 0 <= link < 3 * LINKS_PER_DIGIT:
        return DIGITS[link // LINKS_PER_DIGIT]
    return None                                                    # palm plate (-1)
