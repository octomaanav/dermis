"""Simulator-state stream for external visualizers (Mana): schema docs/mana/dermis_sim_stream.schema.json.

Everything is read back from PyBullet, never re-derived: world link poses (getLinkState), base pose,
measured joint angles, and the visual geometry (getVisualShapeData). A viewer therefore needs no
kinematics of its own and cannot diverge from the simulator. Commanded values are sent separately
so "actual vs commanded" can be compared.

Conventions (also in the schema): metres, radians, newtons, seconds; world frame right-handed with
+z up and +x forward (the hand points along +x); quaternions in PyBullet order [x, y, z, w];
cylinders / capsules along their local +z; a capsule's length is the straight segment between the
two hemisphere centres. t_sim_s is simulation time (tick x control period), the timestamp to compare
against simulator logs. No host clock is ever mixed in.
"""
from __future__ import annotations

import math

import numpy as np

SCHEMA = "dermis.sim_stream/1"
SHAPES = {2: "sphere", 3: "box", 4: "cylinder", 7: "capsule"}
JOINT_ROLES = {"thumb_tip": ("CMC", "MCP", "IP"), "index_tip": ("MCP", "PIP", "DIP"), "middle_tip": ("MCP", "PIP", "DIP")}


def _r(v, nd=5):
    return [round(float(x), nd) for x in v]


def _visual(v) -> dict:
    kind = SHAPES.get(v[2], f"type{v[2]}")
    d = v[3]
    size = ({"radius": round(d[0], 5)} if kind == "sphere" else
            {"x": round(d[0], 5), "y": round(d[1], 5), "z": round(d[2], 5)} if kind == "box" else
            {"length": round(d[0], 5), "radius": round(d[1], 5)})
    return {"shape": kind, "size": size, "pos": _r(v[5]), "quat": _r(v[6]), "rgba": _r(v[7], 3)}


def hand_model(sim) -> dict:
    """Static description of the simulated hand, sent once in the hello message."""
    import pybullet as p
    from ..server.teleop.hand import LINKS_PER_FINGER, flex_limits
    c, b = sim.client, sim.body
    vis = p.getVisualShapeData(b, physicsClientId=c)
    links = [{"index": -1, "name": "palm", "parent": None, "joint": None,
              "visuals": [_visual(v) for v in vis if v[1] == -1]}]
    joints = []
    for j in range(p.getNumJoints(b, physicsClientId=c)):
        info = p.getJointInfo(b, j, physicsClientId=c)
        n, k = divmod(j, LINKS_PER_FINGER)
        pod = sim.pods[n]
        revolute = info[2] == p.JOINT_REVOLUTE
        name = f"{pod.replace('_tip', '')}_{JOINT_ROLES.get(pod, ('J1', 'J2', 'J3'))[k]}" if revolute else f"{pod.replace('_tip', '')}_tip_pad"
        jd = None
        if revolute:
            lo, hi = 0.0, flex_limits(pod)[k]
            jd = {"name": name, "type": "revolute", "axis": _r(info[13], 4), "lower": 0.0, "upper": round(hi, 4)}
            joints.append({"name": name, "link_index": j, "pod": pod, "lower": lo, "upper": round(hi, 4)})
        links.append({"index": j, "name": info[12].decode(), "parent": info[16], "pod": pod,
                      "joint": jd or {"name": name, "type": "fixed"},
                      "visuals": [_visual(v) for v in vis if v[1] == j]})
    return {"robot": "dermis_hand_3f", "units": {"length": "m", "angle": "rad", "force": "N", "time": "s"},
            "frame": {"up": "+z", "forward": "+x", "handedness": "right", "quaternion_order": "xyzw"},
            "links": links, "joints": joints, "objects": [],
            "note": "Link poses in each frame are world poses of the link frame (getLinkState worldLinkFrame); "
                    "visuals are relative to that link frame. Joint axis is in the link frame."}


def hello(sim, source: dict, control_hz: float) -> dict:
    return {"type": "hello", "schema": SCHEMA, "model": hand_model(sim), "source": source,
            "rates": {"frames_hz": control_hz, "note": "one frame per control tick"}}


def frame(sim, tick: int, dt: float, cmd_extra: dict, control: dict, source: dict, stale: bool,
          objects=(), contacts=(), task=None) -> dict:
    """One control tick of measured simulator state (+ the commands that produced it)."""
    import pybullet as p
    c, b = sim.client, sim.body
    pos, orn = p.getBasePositionAndOrientation(b, physicsClientId=c)
    links = []
    for j in range(p.getNumJoints(b, physicsClientId=c)):
        ls = p.getLinkState(b, j, computeForwardKinematics=True, physicsClientId=c)
        links.append({"index": j, "pos": _r(ls[4]), "quat": _r(ls[5])})
    q = [p.getJointState(b, j, physicsClientId=c)[0] for j in sim.joints]
    q_cmd = sim._joint_targets(cmd_extra.get("flex", [0.0] * len(sim.pods)))
    names = getattr(sim, "_mana_joint_names", None)
    if names is None:
        names = sim._mana_joint_names = [jn["name"] for jn in hand_model(sim)["joints"]]
    return {
        "type": "frame", "schema": SCHEMA, "seq": tick, "tick": tick, "t_sim_s": round(tick * dt, 4),
        "source": source, "control": control, "stale": bool(stale),
        "base": {"pos": _r(pos), "quat": _r(orn)},
        "links": links,
        "joints": [{"name": n, "q": round(float(a), 5), "q_cmd": round(float(t), 5)} for n, a, t in zip(names, q, q_cmd)],
        "commands": {"flex": {pod: round(float(f), 4) for pod, f in zip(sim.pods, cmd_extra.get("flex", []))},
                     "tilt_rad": _r(cmd_extra.get("attitude", [0.0, 0.0]), 4),
                     "offset_m": _r(cmd_extra.get("offset", [0.0, 0.0, 0.0]), 4),
                     "note": "EXPERIMENTAL mapping output (or a scripted test pattern); not measured"},
        "objects": list(objects), "contacts": list(contacts), "task": task,
    }


# --------------------------------------------------------------------------------------
# Scripted check sequence (example stream for Mana's simulator checklist)
# --------------------------------------------------------------------------------------

CHECKS = [  # (start s, end s, what moves)
    (0.0, 1.0, "rest"),
    (1.0, 3.0, "palm +x then back"),
    (3.0, 5.0, "palm +y then back"),
    (5.0, 7.0, "palm +z then back"),
    (7.0, 9.0, "wrist roll (about the hand's x axis) +0.4 rad then back"),
    (9.0, 11.0, "index finger only: flex 0 -> 1 -> 0"),
    (11.0, 13.0, "all fingers: fist then open"),
    (13.0, 14.0, "telemetry stale: frames continue, pose held, stale=true"),
]


def _bump(t, a, b):
    """0 -> 1 -> 0 smoothly over [a, b]."""
    if t <= a or t >= b:
        return 0.0
    return 0.5 - 0.5 * math.cos(2 * math.pi * (t - a) / (b - a))


def check_sequence(hz: float = 30.0):
    """Yield (t, extra-command) for the scripted checks. SCRIPTED test pattern, not telemetry."""
    n = int(CHECKS[-1][1] * hz)
    held = None
    for i in range(n):
        t = i / hz
        off = [0.10 * _bump(t, 1, 3), 0.08 * _bump(t, 3, 5), 0.08 * _bump(t, 5, 7)]
        roll = 0.4 * _bump(t, 7, 9)
        flex = [0.0, _bump(t, 9, 11), 0.0]
        fist = _bump(t, 11, 13)
        flex = [max(f, fist) for f in flex]
        cmd = {"flex": flex, "attitude": [roll, 0.0], "offset": off}
        stale = t >= 13.0
        if stale:
            cmd = held                                    # pose held while telemetry is stale
        else:
            held = cmd
        yield t, cmd, stale


def export_check_stream(path: str, hz: float = 30.0) -> int:
    """Run the scripted checks through the real HandSim and write hello + frames as JSON Lines."""
    import json
    from pathlib import Path
    from .teleop.controller import Command
    from .teleop.hand import HandSim
    sim = HandSim()
    source = {"label": "SCRIPTED", "device_id": None, "telemetry_frame": None, "telemetry_age_ms": None,
              "note": "scripted test pattern for viewer checks; not glove data"}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    k = 0
    with open(path, "w") as fh:
        fh.write(json.dumps(hello(sim, source, hz)) + "\n")
        for i, (t, cmd, stale) in enumerate(check_sequence(hz)):
            sim.apply(Command(np.zeros(3), 0.0, "SCRIPTED", None, cmd))
            phase = next(w for a, b, w in CHECKS if a <= t < b)
            fr = frame(sim, i, 1.0 / hz, cmd, {"mode": "SCRIPTED_TEST", "state": "RUNNING", "reason": phase},
                       {**source, "telemetry_age_ms": round((t - 13.0) * 1000, 1) if stale else 0.0}, stale,
                       task={"phase": phase, "outcome": None})
            fh.write(json.dumps(fr) + "\n")
            k += 1
    sim.close()
    return k
