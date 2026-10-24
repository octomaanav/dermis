"""Deterministic scripted expert for grasp-lift-place. Uses privileged simulator state.

This is an environment sanity check and a source of SCRIPTED demonstrations. It is never used
by learned-policy evaluation, and its episodes are always labelled source="scripted".

Design rule: every phase change depends on quantities that are also in the policy's
observation (relative object position, opening, opening velocity, contact). Hidden timers
make demonstrations impossible to clone, because two identical observations would need
different actions. The only timers left are safety fallbacks for stuck phases.
"""
from __future__ import annotations

import numpy as np

from .env import MAX_OPENING, GraspLiftPlaceEnv

APPROACH_DZ = 0.08           # m above the object centre before descending
LIFT_Z = 0.12                # m, object centre height while carrying
PLACE_DZ = 0.004             # release with the object bottom this far above the table
RELEASE_HEIGHT = 0.008       # m; object bottom below this (and over the target) -> open
GAIN = 4.0                   # 1/s, velocity = GAIN * position error (clipped)
GRIP_EFFORT = 15.0           # N; enough for friction >= 0.4 with a 0.2 kg cube, below the 30 N benchmark threshold
OPEN_EFFORT = 10.0
XY_TOL = 0.004


class ScriptedExpert:
    def __init__(self, max_speed: float = 0.15):
        self.max_speed = max_speed
        self.reset()

    def reset(self) -> None:
        self.phase, self.ticks = "approach", 0

    def _vel(self, ee, goal) -> np.ndarray:
        return np.clip(GAIN * (np.asarray(goal) - ee), -self.max_speed, self.max_speed)

    def act(self, env: GraspLiftPlaceEnv) -> np.ndarray:
        s = env.privileged_state()
        ee, obj, tgt, opening_now = s["ee"], s["obj"], s["target_xy"], s["opening"]
        h = env.cfg.object_half
        fl, fr = s["force"]
        gripping = fl > 0.5 and fr > 0.5
        rel = obj - ee
        self.ticks += 1
        ph = self.phase
        opening, effort = 0.0, GRIP_EFFORT

        if ph == "approach":
            opening, effort = MAX_OPENING, OPEN_EFFORT
            goal = [obj[0], obj[1], obj[2] + APPROACH_DZ]
            if np.all(np.abs(rel[:2]) < XY_TOL) and abs(goal[2] - ee[2]) < 0.01:
                self._next("descend")
        elif ph == "descend":
            opening, effort = MAX_OPENING, OPEN_EFFORT
            goal = [obj[0], obj[1], obj[2]]
            if abs(rel[2]) < 0.004 or self.ticks > 90:
                self._next("close")
        elif ph == "close":
            goal = ee                                    # hold still while the fingers close
            if (gripping and opening_now < 2 * h + 0.002) or self.ticks > 45:
                self._next("lift")
        elif ph == "lift":
            goal = [ee[0], ee[1], LIFT_Z - rel[2]]
            if obj[2] > LIFT_Z - 0.01:
                self._next("carry")
        elif ph == "carry":
            goal = [tgt[0] - rel[0], tgt[1] - rel[1], LIFT_Z - rel[2]]   # put the object, not the gripper, over the target
            if np.linalg.norm(obj[:2] - tgt) < XY_TOL:
                self._next("lower")
        elif ph == "lower":
            goal = [tgt[0] - rel[0], tgt[1] - rel[1], h + PLACE_DZ - rel[2]]
            if obj[2] - h < RELEASE_HEIGHT or self.ticks > 120:
                self._next("release")
        elif ph == "release":
            opening, effort = MAX_OPENING, OPEN_EFFORT
            goal = ee
            if opening_now > MAX_OPENING - 0.01:
                self._next("retreat")
        else:                                            # retreat upward and wait for the dwell to finish
            opening, effort = MAX_OPENING, OPEN_EFFORT
            goal = [ee[0], ee[1], obj[2] + APPROACH_DZ]

        if ph in ("approach", "descend") and gripping:  # bumped the object with open fingers: back off
            goal = [ee[0], ee[1], ee[2] + 0.02]
        v = self._vel(ee, goal)
        return np.array([*v, opening, effort], dtype=np.float32)

    def _next(self, phase: str) -> None:
        self.phase, self.ticks = phase, 0
