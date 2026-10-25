#!/usr/bin/env python3
"""Emit the actual learned three-finger assist_hand simulation as NDJSON.

This bridge deliberately lives beside the realtime project.  It imports the
existing Python simulator but neither changes its physics nor sends commands
back into it.  Each stdout line is one measured PyBullet control tick.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from dermis.assist.control import AUTONOMOUS, ModeController, checkpoint_compatible
from dermis.assist.env import AssistHandEnv, default_config
from dermis.cup.env import CupScenario
from dermis.cup.train import CupPolicy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/assist/candidates/fallback_scripted/checkpoint")
    parser.add_argument("--seed", type=int, default=2_000_000)
    parser.add_argument("--realtime", action="store_true", help="pace stdout at the simulation control rate")
    args = parser.parse_args()
    checkpoint = (ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    compatible, reason, metadata = checkpoint_compatible(str(checkpoint))
    if not compatible:
        raise RuntimeError(f"learned assist-hand checkpoint is not usable: {reason}")

    env = AssistHandEnv(default_config())
    try:
        controller = ModeController(env.cfg.dt)
        controller.set_policy(CupPolicy.load(checkpoint), {"checkpoint": str(checkpoint), "metadata": metadata})
        switched, message = controller.switch(AUTONOMOUS)
        if not switched:
            raise RuntimeError(message)
        observation = env.reset(CupScenario.sample(args.seed, env.cfg), pour_at_s=1.0)
        if env.done:
            raise RuntimeError(f"assistive-hand setup failed: {env.stats.termination}")
        print(json.dumps({"header": True, "task": "assist_hand", "checkpoint": str(checkpoint.relative_to(ROOT)),
                          "seed": args.seed, "control_hz": env.cfg.control_hz}), flush=True)
        tick_at = time.monotonic()
        while not env.done:
            action, _trace = controller.act(observation, None)
            observation, _done, info = env.step(action)
            position, orientation = env.hand.p.getBasePositionAndOrientation(env.mug, **env._p)
            joints = env.joint_state()["q"]
            outcome = "SUCCESS" if info["termination"] == "success" else (
                "DROPPED" if info["termination"] in ("drop", "released") else "FAILED" if info["termination"] else "IN_PROGRESS"
            )
            print(json.dumps({
                "t": round(env.t, 6), "robot_joints": joints, "mug_pos": position, "mug_quat": orientation,
                "contact": [info["normal_index"], info["normal_middle"], info["normal_palm"]],
                "applied_load_n": info["load_n"], "displacement_m": info["disp_m"],
                "outcome": outcome, "hold_state": info["hold_state"], "phase": info["phase"],
            }), flush=True)
            if args.realtime:
                tick_at += env.cfg.dt
                time.sleep(max(0.0, tick_at - time.monotonic()))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
