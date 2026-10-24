"""Control modes for the assistive-hand demo. Every mode produces the same action [closure, effort]
for the same physics; the mode only decides who computes it.

Human grip request g in [0, 1] (glove or keyboard):
    g <= RELEASE_G      -> OPEN: closure 0 (fingers extend; deliberate release)
    g >  RELEASE_G      -> CLOSE: closure 1, effort = EFFORT_LO + (g - RELEASE_G)/(1 - RELEASE_G) * (EFFORT_HI - EFFORT_LO)

MANUAL              final = human request.
ASSISTED - SCRIPTED final effort = clip(max(human effort, assist effort), EFFORT_LO, ASSIST_MAX) while the human
                    asks to hold; the assist effort is a bounded feedback law on SIMULATED contact (support load)
                    and object motion relative to the hand. When the human asks to OPEN, the human command is
                    used unchanged (release always wins). When the human input is PAUSED/DISARMED (tracking
                    lost), the held human command is used, and assistance still never exceeds ASSIST_MAX.
                    Every tick where assistance changed or limited the command is logged.
AUTONOMOUS - LEARNED a compatible checkpoint acts on simulator observations only. No human motor input
                    object exists in this mode (environment controls such as Reset / Add load remain).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .env import ACTION_NAMES, EFFORT_MAX, EFFORT_MIN, OBS_NAMES

MANUAL, ASSISTED, AUTONOMOUS = "MANUAL", "ASSISTED - SCRIPTED", "AUTONOMOUS - LEARNED"
MODES = (MANUAL, ASSISTED, AUTONOMOUS)
RELEASE_G = 0.10
EFFORT_LO, EFFORT_HI = 0.03, 0.50
EFFORT_OPEN = 0.15                 # torque used to extend the fingers when opening
ASSIST_MAX = 0.45                  # assistance never commands more than this
TASK = "assist_hand"


def human_to_action(g: float) -> np.ndarray:
    g = float(np.clip(g, 0.0, 1.0))
    if g <= RELEASE_G:
        return np.array([0.0, EFFORT_OPEN], np.float32)
    e = EFFORT_LO + (g - RELEASE_G) / (1 - RELEASE_G) * (EFFORT_HI - EFFORT_LO)
    return np.array([1.0, e], np.float32)


class ScriptedAssist:
    """Bounded feedback assistance (adapted from fragile_cup's HybridExpert: grip-load coupling + slip
    reflex). Inputs: SIMULATED vertical support load and the mug's motion relative to the hand."""

    def __init__(self, k_load: float = 0.025, floor: float = 0.06, slip_speed: float = 0.004,
                 boost: float = 0.04, relax_per_s: float = 0.03):
        self.k_load, self.floor, self.slip_speed, self.boost, self.relax = k_load, floor, slip_speed, boost, relax_per_s
        self.reset()

    def reset(self):
        self.level = 0.0

    def effort(self, obs: np.ndarray, dt: float) -> float:
        i = {n: k for k, n in enumerate(OBS_NAMES)}
        load = max(0.0, float(obs[i["support_z"]]))
        coupled = self.floor + self.k_load * load
        self.level = max(coupled, self.level - self.relax * dt)
        if obs[i["obj_vz"]] < -self.slip_speed:              # mug moving down relative to the hand
            self.level += self.boost
        self.level = float(min(self.level, ASSIST_MAX))
        return self.level


class ModeController:
    """Routes commands. Switching is explicit (switch()) and logged by the caller."""

    def __init__(self, dt: float):
        self.mode, self.dt = MANUAL, dt
        self.assist = ScriptedAssist()
        self.policy = None
        self.policy_info = None
        self.last = {}

    def switch(self, mode: str) -> tuple[bool, str]:
        if mode not in MODES:
            return False, f"unknown mode {mode}"
        if mode == AUTONOMOUS and self.policy is None:
            return False, "no compatible learned checkpoint is active: AUTONOMOUS disabled"
        self.mode = mode
        self.assist.reset()
        if self.policy is not None:
            self.policy.reset()
        return True, f"mode -> {mode}"

    def set_policy(self, policy, info: dict) -> None:
        self.policy, self.policy_info = policy, info

    def reset(self) -> None:
        self.assist.reset()
        if self.policy is not None:
            self.policy.reset()

    def act(self, obs: np.ndarray, human_g: float | None) -> tuple[np.ndarray, dict]:
        """human_g is None in AUTONOMOUS (never consulted). Returns (final action, trace for logging)."""
        if self.mode == AUTONOMOUS:
            a = np.asarray(self.policy.act(obs), np.float32).reshape(2)
            self.last = {"mode": self.mode, "human_g": None, "human_action": None, "assist_effort": None,
                         "final": a.tolist(), "assist_changed": False, "limited": False}
            return a, self.last
        human = human_to_action(human_g if human_g is not None else 0.0)
        final = human.copy()
        assist_e, changed, limited = None, False, False
        if self.mode == ASSISTED and human[0] > 0.5:                # human asks to hold
            assist_e = self.assist.effort(obs, self.dt)
            want = max(float(human[1]), assist_e)
            final[1] = float(np.clip(want, EFFORT_LO, max(ASSIST_MAX, float(human[1]))))
            changed = abs(final[1] - human[1]) > 1e-4
            limited = assist_e >= ASSIST_MAX - 1e-9
        elif self.mode == ASSISTED:
            self.assist.reset()                                       # release wins; assistance stands down
        self.last = {"mode": self.mode, "human_g": float(human_g) if human_g is not None else None,
                     "human_action": human.tolist(), "assist_effort": assist_e, "final": final.tolist(),
                     "assist_changed": bool(changed), "limited": bool(limited)}
        return final, self.last


def checkpoint_compatible(path: str) -> tuple[bool, str, dict | None]:
    """A checkpoint is usable in AUTONOMOUS only if it was trained for this exact task and schema."""
    mp = Path(path) / "metadata.json"
    if not (Path(path) / "model.pt").exists() or not mp.exists():
        return False, f"{path}: no checkpoint files", None
    meta = json.loads(mp.read_text())
    if meta.get("task") != TASK:
        return False, f"{path}: trained for task {meta.get('task')!r}, not {TASK!r}", meta
    if meta.get("obs_names_all") != list(OBS_NAMES):
        return False, f"{path}: observation schema differs", meta
    if meta.get("action_names") != list(ACTION_NAMES):
        return False, f"{path}: action schema differs", meta
    return True, "compatible", meta
