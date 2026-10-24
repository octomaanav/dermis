#!/usr/bin/env python3
"""Generate scripted expert demonstrations of the extraction task for behavioral cloning.

    python generate_demos.py                      # 50 successful episodes -> scripted_demos.csv
    python generate_demos.py --episodes 100 --gui # watch it work
    python train_model.py --csv scripted_demos.csv

Each episode resets the scene (random patient and tumor position), then the expert:
approaches above the tumor, descends into the hole, closes, lifts. The expert is closed-loop:
every tick it steps from the gripper's *current* position toward the live tumor position, so
its actions also show how to recover from being off course. The arm executes the command plus
a little noise, but the CSV records the clean command; that puts slightly-off states with the
right correction into the data, which is what keeps a cloned policy from drifting.

Only episodes that actually lift the tumor are kept. Rows use record_demo.py's columns, with
the state captured before each command, so train_model.py reads them as is.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

from record_demo import COLUMNS
from sim_env import CONTROL_HZ, SurgerySim

APPROACH_Z = 0.12             # m; line up above the tumor here before descending
GRASP_ABOVE_TUMOR = 0.003     # m; gripper target above the tumor centre when closing
LIFT_Z = 0.25
MAX_STEP = 0.004              # m per tick (0.12 m/s), a little under the keyboard's speed
SETTLE_TICKS = 15             # ticks to hold while the fingers close, and at the top
MAX_TICKS = 20 * CONTROL_HZ
SUCCESS_Z = 0.15              # tumor centre above this at the end = extracted


def step_toward(ee: list, goal: list, max_step: float = MAX_STEP) -> list:
    d = [g - e for g, e in zip(goal, ee)]
    n = math.sqrt(sum(v * v for v in d))
    scale = 1.0 if n <= max_step else max_step / n
    return [e + v * scale for e, v in zip(ee, d)]


def run_episode(sim: SurgerySim, episode: int, noise: float, rng: random.Random) -> tuple[list, bool]:
    sim.reset()
    rows, phase, phase_ticks, grip = [], "approach", 0, 0.0
    for step in range(MAX_TICKS):
        state = sim.get_state()
        ee = [state["ee_x"], state["ee_y"], state["ee_z"]]
        tumor = [state["tumor_x"], state["tumor_y"], state["tumor_z"]]
        grasp = [tumor[0], tumor[1], tumor[2] + GRASP_ABOVE_TUMOR]

        if phase == "approach":
            goal = [tumor[0], tumor[1], APPROACH_Z]
            if math.dist(ee, goal) < 0.004:
                phase, phase_ticks = "descend", 0
        elif phase == "descend":
            goal = grasp
            if math.dist(ee, goal) < 0.003 or phase_ticks > 3 * CONTROL_HZ:
                phase, phase_ticks = "close", 0
        elif phase == "close":
            goal = grasp
            grip = 1.0                                # close in one step: a clear label for the policy
            if phase_ticks > SETTLE_TICKS:
                phase, phase_ticks = "lift", 0
        elif phase == "lift":
            goal = [tumor[0], tumor[1], LIFT_Z]
            if math.dist(ee, goal) < 0.005:
                phase, phase_ticks = "hold", 0
        else:                                             # hold at the top, then stop
            goal = ee
            if phase_ticks > SETTLE_TICKS:
                break

        command = step_toward(ee, goal)
        rows.append({**state, "episode": episode, "step": step, "t": step / CONTROL_HZ,
                     "target_x": command[0], "target_y": command[1], "target_z": command[2],
                     "grip_cmd": grip})
        sim.target = [c + rng.gauss(0.0, noise) for c in command]
        sim.grip = grip
        sim.step()
        phase_ticks += 1

    return rows, sim.get_state()["tumor_z"] > SUCCESS_Z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=50, help="successful episodes to keep")
    ap.add_argument("--out", type=Path, default=Path("scripted_demos.csv"))
    ap.add_argument("--noise", type=float, default=0.0015, help="std (m) of noise added to executed commands")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gui", action="store_true", help="show the PyBullet window")
    args = ap.parse_args()

    sim = SurgerySim(gui=args.gui, randomize=True, seed=args.seed)
    rng = random.Random(args.seed + 1)
    kept, tried, rows_written = 0, 0, 0
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        while kept < args.episodes and tried < args.episodes * 2:
            tried += 1
            rows, ok = run_episode(sim, kept, args.noise, rng)
            if ok:
                writer.writerows(rows)
                kept += 1
                rows_written += len(rows)
            print(f"episode {tried:3d}: {'kept' if ok else 'failed, dropped'} ({len(rows)} ticks)  "
                  f"[{kept}/{args.episodes}]", flush=True)
    sim.close()
    print(f"Wrote {kept} successful episodes ({rows_written} rows) to {args.out}; {tried - kept} failed attempts dropped")
    return 0 if kept == args.episodes else 1


if __name__ == "__main__":
    sys.exit(main())
