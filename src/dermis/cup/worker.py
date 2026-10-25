"""Headless fragile_cup worker: runs one controller on one scenario and writes a per-tick JSONL trace of
the actual physics state, for later visualization or checking.

    python -m dermis cup worker --checkpoint runs/cup/autonomous_demo/cup_autonomous_demo_v1 --seed 2000003 \
        --out runs/cup/autonomous_demo/trace.jsonl [--plot trace.png]
    python -m dermis cup worker --scripted fixed_4N --seed 2000003 --out insufficient.jsonl

Each line: t, tick, mode/label, controller (checkpoint id or scripted name), robot joint angles (arm 1-7,
fingers 9-10), gripper position, cup pose (pos + quaternion), per-finger contact (normal / vertical shear /
shear magnitude, N, SIMULATED; total = left + right), applied load (N) and remaining ramp time, schedule
id, phase, effort command, HoldStatus (movement / load status / slip / timer / outcome) and new events.
The first line is a header with the scenario (incl. the full load schedule) and its id.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pybullet as p

from .env import ARM_JOINTS, FINGER_JOINTS, CupConfig, CupHoldEnv, CupScenario
from .status import HoldStatus, outcome_text


def schedule_id(sc: CupScenario) -> str:
    return hashlib.sha256(json.dumps(sc.schedule.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def tick_state(env: CupHoldEnv, status: dict, effort: float, events: list, ctrl_name: str, label: str) -> dict:
    P = env._p
    q = {f"j{j}": round(p.getJointState(env.robot, j, **P)[0], 5) for j in list(ARM_JOINTS) + list(FINGER_JOINTS)}
    pos, orn = p.getBasePositionAndOrientation(env.cup, **P)
    c = env._contact
    sched = env.scenario.schedule
    st = env.schedule_t
    ramp_left = None if st is None else max(0.0, round(sched.duration - st, 2))
    return {"t": round(env.t, 4), "tick": env.stats.ticks, "label": label, "controller": ctrl_name,
            "robot_joints": q, "gripper_pos": [round(v, 5) for v in env._ee_pos()],
            "cup_pos": [round(v, 5) for v in pos], "cup_quat": [round(v, 5) for v in orn],
            "contact": {"normal_l": round(float(c["normal"][0]), 3), "normal_r": round(float(c["normal"][1]), 3),
                        "normal_total": round(float(c["normal"].sum()), 3),
                        "shear_z_l": round(float(c["shear_z"][0]), 3), "shear_z_r": round(float(c["shear_z"][1]), 3),
                        "shear_l": round(float(c["shear"][0]), 3), "shear_r": round(float(c["shear"][1]), 3),
                        "convention": "per finger, N, SIMULATED; total = left + right"},
            "applied_load_n": round(env.load_n, 3), "ramp_remaining_s": ramp_left, "phase": env.phase(),
            "effort_cmd_n": round(float(effort), 3), "status": status, "events": events}


def run_trace(spec: dict, seed: int, out: str, pour_at_s: float = 1.0, cfg: CupConfig | None = None) -> dict:
    from .evaluate import _make_controller
    from ..config import load_config
    cfg = cfg or CupConfig.from_dict(load_config("configs/cup/task.yaml"))
    env = CupHoldEnv(cfg)
    ctrl = _make_controller(spec)
    name = spec.get("name") or Path(spec["checkpoint"]).name
    label = "SCRIPTED" if spec["kind"] == "scripted" else "AUTONOMOUS"
    sc = CupScenario.sample(seed, cfg)
    hs = HoldStatus()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    try:
        obs = env.reset(sc, pour_at_s=pour_at_s)
        ctrl.reset()
        with open(out, "w") as fh:
            fh.write(json.dumps({"header": True, "task_config": cfg.to_dict(), "scenario": sc.to_dict(),
                                 "schedule_id": schedule_id(sc), "controller": name, "label": label,
                                 "pour_at_s": pour_at_s}) + "\n")
            seen = 0
            while not env.done:
                effort, _, _ = ctrl.act(obs, env)
                obs, done, info = env.step(effort)
                new = env.stats.events[seen:]
                seen = len(env.stats.events)
                fh.write(json.dumps(tick_state(env, hs.update(env), env.effort, new, name, label)) + "\n")
                rows += 1
        res = {"trace": out, "ticks": rows, "termination": env.stats.termination, "outcome_text": outcome_text(env.stats.termination),
               "final_status": hs.update(env), "schedule_id": schedule_id(sc), "seed": seed,
               "peak_normal_n": round(env.stats.peak_normal_n, 2), "damage_ns": round(env.stats.damage_ns, 2)}
        Path(out).with_suffix(".summary.json").write_text(json.dumps(res, indent=1))
        return res
    finally:
        env.close()


def plot_trace(trace: str, png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = [json.loads(l) for l in Path(trace).read_text().splitlines()]
    head, rows = lines[0], lines[1:]
    t = [r["t"] for r in rows]
    fig, ax = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(t, [r["applied_load_n"] for r in rows], color="#3b82f6")
    ax[0].set_ylabel("applied load (N)\nSIMULATED")
    ax[1].plot(t, [r["contact"]["normal_l"] for r in rows], label="left finger")
    ax[1].plot(t, [r["contact"]["normal_r"] for r in rows], label="right finger")
    ax[1].plot(t, [r["effort_cmd_n"] for r in rows], ls=":", color="k", label="effort command")
    ax[1].axhline(head["task_config"]["break_force_n"], color="r", lw=0.8, label="instant limit (simulated)")
    if "sustained" in head["task_config"]["damage_model"]:
        ax[1].axhline(head["task_config"]["sustained_safe_n"], color="r", lw=0.8, ls="--", label="sustained-safe level")
    ax[1].set_ylabel("fingertip normal (N)\nper finger")
    ax[1].legend(fontsize=7, loc="upper right")
    ax[2].plot(t, [r["status"]["slip_cm"] for r in rows], color="#f59e0b")
    ax[2].axhline(head["task_config"]["drop_slip_m"] * 100, color="r", lw=0.8, label="drop threshold")
    ax[2].axhline(head["task_config"]["success_slip_m"] * 100, color="r", lw=0.8, ls="--", label="success limit")
    ax[2].set_ylabel("cup down vs hand (cm)")
    ax[2].legend(fontsize=7, loc="upper left")
    states = ["HOLDING", "SLIPPING", "DROPPED"]
    ax[3].step(t, [states.index(r["status"]["movement"]) for r in rows], where="post", label="movement")
    ax[3].step(t, [1.0 if r["status"]["load_status"] == "EXCESSIVE LOAD" else 0.5 if r["status"]["load_status"] != "OK" else 0.0
                   for r in rows], where="post", color="r", alpha=0.6, label="load status (0 ok, .5 warn, 1 excessive)")
    ax[3].set_yticks([0, 1, 2], states)
    ax[3].legend(fontsize=7, loc="upper left")
    ax[3].set_xlabel("time (s)")
    last = rows[-1]
    fig.suptitle(f"{head['label']} {head['controller']} | seed {head['scenario']['seed']} schedule {head['schedule_id']} | "
                 f"outcome: {last['status']['outcome']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)
