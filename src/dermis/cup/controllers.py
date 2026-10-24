"""Grip controllers for fragile_cup. Each returns (effort_issued, expert_label_or_None, tag).

Scripted (SCRIPTED label, task checks and demonstrations):
    FixedEffort        constant effort; used to check whether the task needs adaptation at all
    GripLoadExpert     the scripted feedback expert. Uses only the observation vector (gripper,
                       object and *contact* groups): grip normal force is scaled with the measured
                       vertical shear (load) for an assumed worst-case friction, plus a slip
                       correction. Grip-load coupling of this kind is how human grasping behaves.
    HybridExpert       grip-load coupling with a mid-range friction assumption + slip reflex
    SlipReflex         non-tactile feedback: only gripper + object observations. Raises effort
                       when the cup slides relative to the gripper, otherwise relaxes slowly.
    Oracle             privileged upper bound: true friction and applied load. Analysis only.
Learned (AUTONOMOUS): PolicyController wraps a checkpoint and sees only the observation.
Human (TEACH): KeyboardGrip now; GloveGrip later (EXPERIMENTAL, explicit enable).
"""
from __future__ import annotations

import numpy as np

from .env import MAX_EFFORT, MIN_EFFORT, OBS_NAMES, SETUP_EFFORT_N, CupHoldEnv

I = {n: i for i, n in enumerate(OBS_NAMES)}


class FixedEffort:
    label = "SCRIPTED"

    def __init__(self, effort: float):
        self.effort = float(effort)
        self.name = f"fixed_{self.effort:g}N"

    def reset(self):
        pass

    def act(self, obs, env=None):
        return self.effort, None, "SCRIPTED"


class GripLoadExpert:
    """effort = (|shear_z_l| + |shear_z_r|) / (2 * mu_assumed) * (1 + margin) + floor  (+ slip boost)."""
    label = "SCRIPTED"
    name = "grip_load_expert"

    def __init__(self, mu_assumed: float = 0.35, margin: float = 0.25, floor_n: float = 1.0,
                 slip_vel: float = 0.004, slip_boost_n: float = 4.0):
        self.mu, self.margin, self.floor = mu_assumed, margin, floor_n
        self.slip_vel, self.slip_boost = slip_vel, slip_boost_n

    def reset(self):
        pass

    def act(self, obs, env=None):
        load = abs(obs[I["shear_z_l"]]) + abs(obs[I["shear_z_r"]])
        f = load / (2 * self.mu) * (1 + self.margin) + self.floor
        if obs[I["obj_vz"]] < -self.slip_vel:
            f = max(f, obs[I["effort_prev"]]) + self.slip_boost
        f = float(np.clip(f, MIN_EFFORT, MAX_EFFORT))
        return f, f, "SCRIPTED"


class SlipReflex:
    """No contact sensing. effort_t = effort_{t-1} + up if sliding else - down (to a floor)."""
    label = "SCRIPTED"
    name = "slip_reflex"

    def __init__(self, slip_vel: float = 0.003, up_n: float = 2.0, down_n_per_s: float = 1.5, floor_n: float = 2.0):
        self.slip_vel, self.up, self.down, self.floor = slip_vel, up_n, down_n_per_s, floor_n

    def reset(self):
        pass

    def act(self, obs, env=None):
        prev = obs[I["effort_prev"]]
        if obs[I["obj_vz"]] < -self.slip_vel:
            f = prev + self.up
        else:
            f = prev - self.down / 30.0
        f = float(np.clip(max(f, self.floor), MIN_EFFORT, MAX_EFFORT))
        return f, f, "SCRIPTED"


class HybridExpert:
    """Grip-load coupling (contact shear, mid-range friction assumption) + slip reflex (object
    motion). effort = max(load term, previous effort - relax) + boost while sliding."""
    label = "SCRIPTED"
    name = "hybrid_expert"

    def __init__(self, mu_assumed: float = 0.6, margin: float = 0.2, floor_n: float = 1.0,
                 slip_vel: float = 0.003, slip_boost_n: float = 2.0, relax_n_per_s: float = 1.5):
        self.mu, self.margin, self.floor = mu_assumed, margin, floor_n
        self.slip_vel, self.boost, self.relax = slip_vel, slip_boost_n, relax_n_per_s

    def reset(self):
        pass

    def act(self, obs, env=None):
        load = abs(obs[I["shear_z_l"]]) + abs(obs[I["shear_z_r"]])
        coupled = load / (2 * self.mu) * (1 + self.margin) + self.floor
        f = max(coupled, obs[I["effort_prev"]] - self.relax / 30.0)
        if obs[I["obj_vz"]] < -self.slip_vel:
            f += self.boost
        f = float(np.clip(f, MIN_EFFORT, MAX_EFFORT))
        return f, f, "SCRIPTED"


class Oracle:
    label = "SCRIPTED"
    name = "oracle_privileged"

    def __init__(self, margin: float = 0.15, floor_n: float = 0.8):
        self.margin, self.floor = margin, floor_n

    def reset(self):
        pass

    def act(self, obs, env: CupHoldEnv):
        s = env.privileged()
        need = (s["mass"] * 9.81 + s["load_n"]) / (2 * s["friction"])
        f = float(np.clip(need * (1 + self.margin) + self.floor, MIN_EFFORT, MAX_EFFORT))
        return f, None, "SCRIPTED"


class PolicyController:
    """AUTONOMOUS: a trained checkpoint acting on observations only. Accepts no human input."""
    label = "AUTONOMOUS"

    def __init__(self, policy):
        self.policy = policy
        self.name = policy.meta.get("name", "policy")

    def reset(self):
        self.policy.reset()

    def act(self, obs, env=None):
        return float(self.policy.act(obs)[0]), None, "AUTONOMOUS"


class KeyboardGrip:
    """TEACH with keyboard: Up/Down (or W/S) raise/lower effort while held; 1..9 jump to presets."""
    label = "KEYBOARD"
    name = "keyboard"
    RATE_N_PER_S = 12.0

    def __init__(self):
        self.effort = SETUP_EFFORT_N
        self.down: set = set()

    def reset(self):
        self.effort = SETUP_EFFORT_N

    def handle_keys(self, down: set, pressed: set) -> None:
        self.down = down
        for k in pressed:
            if k in "123456789":
                self.effort = float(int(k) * 4)

    def act(self, obs, env=None):
        step = self.RATE_N_PER_S / 30.0
        if self.down & {"up", "w"}:
            self.effort += step
        if self.down & {"down", "s"}:
            self.effort -= step
        self.effort = float(np.clip(self.effort, MIN_EFFORT, MAX_EFFORT))
        return self.effort, None, "KEYBOARD"


class GloveGrip:
    """TEACH with the glove (EXPERIMENTAL): the armed, two-point-calibrated grip mapping of
    dermis.glove.grip drives the fingers' effort through the same act() interface as KeyboardGrip.

    Operator keys (TEACH only): A = arm, X = disarm. Until armed, while PAUSED (stale/unusable
    required channel, watchdog) or after DISARMED (connection lost, calibration mismatch) the effort
    is HELD at its last value (bounded): neither dropping nor squeezing the cup. Each recorded tick is
    tagged GLOVE_<state>. The glove reading is never a policy observation."""
    label = "GLOVE"
    name = "glove_experimental"
    EXPERIMENTAL = True

    def __init__(self, runner, control, enabled: bool = False):
        if not enabled:
            raise PermissionError("GloveGrip is experimental; enable it deliberately (--experimental-glove-teleop)")
        self.runner, self.control = runner, control
        self.control.value = SETUP_EFFORT_N if control.state != "ARMED" else control.value
        self.status = control.state
        self.cal = control.cal
        self.last_message = ""

    def reset(self):
        pass                                    # arming survives scenario resets; effort is held, not reset

    def handle_keys(self, down: set, pressed: set) -> None:
        if "a" in pressed:
            ok, msg = self.control.arm(self.runner.latest(), self.runner.source_stale())
            self.last_message = msg
        if "x" in pressed:
            self.control.disarm()
            self.last_message = "disarmed by operator"

    def act(self, obs, env=None):
        cmd = self.control.tick(self.runner.latest())
        self.status = cmd.state
        return float(np.clip(cmd.value, MIN_EFFORT, MAX_EFFORT)), None, f"GLOVE_{cmd.state}"


class ReplayEffort:
    """REPLAY: plays a recorded episode's issued efforts open loop. Not learned, not autonomous."""
    label = "REPLAY"

    def __init__(self, actions, name: str):
        self.actions, self.name, self.k = np.asarray(actions).reshape(-1), f"replay:{name}", 0

    def reset(self):
        self.k = 0

    def act(self, obs, env=None):
        a = float(self.actions[min(self.k, len(self.actions) - 1)])
        self.k += 1
        return (a if self.k <= len(self.actions) else None), None, "REPLAY"


def scripted(name: str):
    if name.startswith("fixed_"):
        return FixedEffort(float(name[len("fixed_"):].rstrip("N")))
    return {"grip_load_expert": GripLoadExpert, "slip_reflex": SlipReflex, "hybrid_expert": HybridExpert,
            "oracle_privileged": Oracle}[name]()
