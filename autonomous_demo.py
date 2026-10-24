#!/usr/bin/env python3
"""Autonomous rollout of surgeon_model.pth with a hard safety stop and a Claude-written diagnostic.

    export ANTHROPIC_API_KEY="sk-ant-your-key-here"     # placeholder: use your own key
    python autonomous_demo.py                  # press F in the window to inject a fault
    python autonomous_demo.py --fault-at 5     # inject the fault automatically after 5 s

Every tick the policy predicts (grip, move_x, move_y, move_z) from the current state and the
gripper is commanded to ee + move. Before a command is executed, the safety monitor checks the
measured gripper z and the commanded z against SAFETY_Z_MIN (the table plane). On a violation
the robot is frozen in place, then Claude is asked for an Emergency Diagnostic Report, which is
printed to the terminal. If the API is unreachable, a local report is printed instead.

The fault is simulated: a growing downward bias on the policy's z output, standing in for sensor
drift or a bad prediction, so the stop can be shown on demand.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone

import anthropic
import pybullet as p
import torch

from sim_env import ARM_JOINTS, CONTROL_HZ, FINGER_JOINTS, SUBSTEPS, WORKSPACE, SurgerySim, triggered
from train_model import INPUTS, MODEL_PATH, OUTPUTS, SurgeonMLP, state_inputs

SAFETY_Z_MIN = 0.0            # m; the table plane
MAX_STEP = 0.01               # m per tick; clamp on predicted moves
FAULT_DRIFT = 0.002           # m per tick^2; how fast the injected downward bias grows
GRIP_THRESHOLD = 0.5          # predicted grip above this closes the gripper fully, else fully open
CLAUDE_MODEL = "claude-haiku-5-5"

SYSTEM_PROMPT = """You are the safety-monitoring assistant for an autonomous surgical robotics \
research prototype. The robot is a simulated Franka Panda arm performing a foreign-body \
extraction task on a simulated patient; no real patient is involved. The control system has \
already frozen the robot. You will receive the telemetry captured at the moment of the stop.

Write a brief Emergency Diagnostic Report for the surgical team, as plain text for a terminal \
(no Markdown), under 250 words, with these sections:
SUMMARY, WHAT HAPPENED, LIKELY CAUSE, WHY THE HALT PROTECTS THE PATIENT, BEFORE RESUMING.
Base every statement on the telemetry. Where the data cannot establish a cause, say so rather \
than guessing."""


def load_policy(path=MODEL_PATH):
    ckpt = torch.load(path, map_location="cpu")
    if ckpt["inputs"] != INPUTS:
        sys.exit(f"{path} was trained on inputs {ckpt['inputs']}; retrain with train_model.py")
    model = SurgeonMLP(len(ckpt["inputs"]), len(ckpt["outputs"]), ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def policy(state: dict) -> dict:
        x = torch.tensor([state_inputs(state)])
        with torch.no_grad():
            y = model((x - ckpt["x_mean"]) / ckpt["x_std"]) * ckpt["y_std"] + ckpt["y_mean"]
        return dict(zip(OUTPUTS, y[0].tolist()))

    return policy


def freeze(sim: SurgerySim) -> None:
    """Hold every joint where it is right now."""
    joints = ARM_JOINTS + list(FINGER_JOINTS)
    here = [p.getJointState(sim.robot, j)[0] for j in joints]
    p.setJointMotorControlArray(sim.robot, joints, p.POSITION_CONTROL, targetPositions=here,
                                targetVelocities=[0.0] * len(joints), forces=[500.0] * len(joints))


def local_report(error_state: dict, why: str) -> str:
    s = error_state
    return (f"EMERGENCY DIAGNOSTIC REPORT (offline: {why})\n"
            f"SUMMARY: Autonomous run halted at t={s['sim_time_s']:.2f}s by the safety monitor.\n"
            f"WHAT HAPPENED: {s['trigger']}. Measured gripper z={s['measured_gripper_xyz'][2]:.4f} m, "
            f"commanded z={s['commanded_gripper_xyz'][2]:.4f} m, limit {s['safety_z_min_m']} m.\n"
            f"LIKELY CAUSE: fault injected={s['fault_injected']}; see the last policy actions.\n"
            "WHY THE HALT PROTECTS THE PATIENT: executing the command would drive the instrument "
            "below the table plane, through the patient.\n"
            "BEFORE RESUMING: review the policy outputs and sensor data, then reset the scene.")


def diagnostic_report(error_state: dict) -> str:
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            output_config={"effort": "low"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content":
                       "Telemetry at the safety stop:\n" + json.dumps(error_state, indent=2)}],
        )
    except anthropic.AuthenticationError:
        return local_report(error_state, "ANTHROPIC_API_KEY missing or invalid")
    except anthropic.RateLimitError:
        return local_report(error_state, "Claude API rate limit")
    except anthropic.APIStatusError as e:
        return local_report(error_state, f"Claude API error {e.status_code}")
    except anthropic.APIConnectionError:
        return local_report(error_state, "no connection to the Claude API")
    except anthropic.AnthropicError as e:
        return local_report(error_state, str(e).splitlines()[0])
    except TypeError as e:                             # the SDK raises this when no credentials are configured
        if "authentication" not in str(e):
            raise
        return local_report(error_state, "no ANTHROPIC_API_KEY configured")
    if response.stop_reason == "refusal":
        return local_report(error_state, "Claude declined to write the report")
    return "\n".join(b.text for b in response.content if b.type == "text").strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--fault-at", type=float, help="inject the downward fault after this many seconds")
    ap.add_argument("--direct", action="store_true", help="headless (no GUI window)")
    args = ap.parse_args()

    policy = load_policy(args.model)
    sim = SurgerySim(gui=not args.direct)
    period = 1.0 / CONTROL_HZ
    step, drift, fault, frozen = 0, 0.0, False, False
    history = deque(maxlen=10)
    print("Autonomous mode. F = inject fault, R = reset, Ctrl-C = quit.")

    try:
        while p.isConnected(sim.client):
            t0 = time.monotonic()
            keys = p.getKeyboardEvents()
            if triggered(keys, ord("r")):
                sim.reset()
                p.removeAllUserDebugItems()
                step, drift, fault, frozen = 0, 0.0, False, False
                history.clear()
                print("Reset.")
            if not fault and (triggered(keys, ord("f")) or
                              (args.fault_at is not None and step / CONTROL_HZ >= args.fault_at)):
                fault = True
                print("Fault injected: downward drift on the policy's z output.")

            if not frozen:
                state = sim.get_state()
                action = policy(state)
                move = [max(-MAX_STEP, min(MAX_STEP, action[k]))
                        for k in ("target_move_x", "target_move_y", "target_move_z")]
                if fault:
                    drift += FAULT_DRIFT
                    move[2] -= drift
                ee = [state["ee_x"], state["ee_y"], state["ee_z"]]
                command = [ee[0] + move[0], ee[1] + move[1], ee[2] + move[2]]
                command[0] = max(WORKSPACE[0][0], min(WORKSPACE[0][1], command[0]))
                command[1] = max(WORKSPACE[1][0], min(WORKSPACE[1][1], command[1]))
                command[2] = min(WORKSPACE[2][1], command[2])     # no floor clamp: the monitor owns that
                history.append({"t": round(step / CONTROL_HZ, 3),
                                "predicted": {k: round(v, 5) for k, v in action.items()},
                                "commanded_z": round(command[2], 5)})

                trigger = None
                if ee[2] < SAFETY_Z_MIN:
                    trigger = "measured gripper z below the safety floor"
                elif command[2] < SAFETY_Z_MIN:
                    trigger = "commanded gripper z below the safety floor (blocked before execution)"
                if trigger:
                    frozen = True
                    freeze(sim)
                    p.addUserDebugText("SAFETY STOP", (ee[0], ee[1], ee[2] + 0.15), (1, 0, 0), 2.0)
                    error_state = {
                        "event": "SAFETY_STOP",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "sim_time_s": step / CONTROL_HZ,
                        "trigger": trigger,
                        "safety_z_min_m": SAFETY_Z_MIN,
                        "measured_gripper_xyz": [round(v, 5) for v in ee],
                        "commanded_gripper_xyz": [round(v, 5) for v in command],
                        "tumor_xyz": [round(state[k], 5) for k in ("tumor_x", "tumor_y", "tumor_z")],
                        "finger_opening_m": [round(state["finger_l"], 5), round(state["finger_r"], 5)],
                        "fault_injected": fault,
                        "injected_z_bias_m": round(-drift, 5),
                        "last_policy_actions": list(history),
                    }
                    print(f"\n*** SAFETY STOP: {trigger}. Robot frozen. Requesting diagnostic report... ***\n")
                    print(diagnostic_report(error_state))
                    print("\nPress R in the window to reset, Ctrl-C to quit.")
                else:
                    sim.target = command
                    # on/off grip: a partly closed gripper would read as "grasping" and close itself further
                    sim.grip = 1.0 if action["target_grip_angle"] > GRIP_THRESHOLD else 0.0

            if frozen:                   # physics only; sim.step() would re-apply IK and undo the freeze
                for _ in range(SUBSTEPS):
                    p.stepSimulation()
            else:
                sim.step()
                step += 1
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except (KeyboardInterrupt, p.error):
        pass
    finally:
        if p.isConnected(sim.client):
            sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
