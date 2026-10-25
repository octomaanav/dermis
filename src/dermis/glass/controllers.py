"""Controllers for glass_hold. Each returns a grip effort (N*m) per control tick.

SCRIPTED (simulator ground truth allowed, labelled as such):
  too_weak       fixed 0.02 N*m: holds the empty glass, too little once water is added -> drop
  over_squeeze   squeezes harder for every gram poured (0.035 + 1.0 N*m/kg * water): -> simulated crack
  load_expert    grip-load coupling from the TRUE weight and friction (privileged) with a floor, a cap and
                 a slip reflex on the true glass velocity; the demonstrator for the neural network
  fixed_<x>      fixed effort x N*m (baseline)
AUTONOMOUS:
  PolicyController  the trained network; reads ONLY the SYNTHETIC glove frames (4 pods) and its own
                    previous effort command (dermis.glass.features). No simulator ground truth.
"""
from __future__ import annotations

import numpy as np
import pybullet as p

from .env import EFFORT_HIGH, EFFORT_LOW, GlassEnv

HOLD_CAPACITY = 200.0         # weight held per (N*m effort * friction coefficient); from the fixed-effort sweep
EXPERT_MARGIN, EXPERT_FLOOR, EXPERT_CAP, EXPERT_SLEW = 1.5, 0.04, 0.075, 0.003


class Scripted:
    label = "SCRIPTED"

    def reset(self, env: GlassEnv):
        pass


class Fixed(Scripted):
    def __init__(self, effort: float, name: str | None = None):
        self.e, self.name = effort, name or f"fixed_{effort:g}"

    def act(self, env: GlassEnv) -> float:
        return self.e


class OverSqueeze(Scripted):
    name = "over_squeeze"

    def act(self, env: GlassEnv) -> float:
        return 0.035 + 1.0 * env.sc.water_at(env.t)


class LoadExpert(Scripted):
    name = "load_expert"

    def reset(self, env: GlassEnv):
        self.e = env.cfg.setup_effort

    def act(self, env: GlassEnv) -> float:
        w = (env.sc.glass_mass + env.sc.water_at(env.t + 1.0 / env.cfg.control_hz)) * 9.81
        target = EXPERT_MARGIN * w / (HOLD_CAPACITY * env.sc.friction)
        vz = p.getBaseVelocity(env.glass, **env._p)[0][2]
        if vz < -0.003:                                         # slip reflex (true velocity)
            target *= 1.25
        target = float(np.clip(target, EXPERT_FLOOR, EXPERT_CAP))
        self.e += float(np.clip(target - self.e, -EXPERT_SLEW, EXPERT_SLEW))
        return self.e


class PolicyController:
    label = "AUTONOMOUS"

    def __init__(self, policy):
        self.policy = policy
        self.name = policy.meta["name"]

    def reset(self, env: GlassEnv):
        self.policy.reset(env.baseline_frame)

    def act(self, env: GlassEnv) -> float:
        return float(np.clip(self.policy.act(env.frame, env.effort), EFFORT_LOW, EFFORT_HIGH))


def scripted(name: str):
    if name == "too_weak":
        return Fixed(0.02, "too_weak")
    if name == "over_squeeze":
        return OverSqueeze()
    if name == "load_expert":
        return LoadExpert()
    if name.startswith("fixed_"):
        return Fixed(float(name[6:]))
    raise KeyError(name)
