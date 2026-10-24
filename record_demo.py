#!/usr/bin/env python3
"""Record teleoperated demonstrations at 30 Hz into demonstrations.csv.

    python record_demo.py                    # mock_stream.py drives the pinch (--synthetic for a fake hand)
    python record_demo.py --keyboard-grip    # keyboard only (SPACE closes)
    python record_demo.py --port /dev/cu.usbmodem1101 [--wire]

Controls are the same as sim_env.py. R resets the scene and starts a new episode.
Each row is one tick: state (ee, tumor, fingers) plus action (target xyz, grip command).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import pybullet as p

from sim_env import CONTROL_HZ, SurgerySim, control_tick, make_arg_parser, open_glove, triggered

COLUMNS = ["episode", "step", "t",
           "ee_x", "ee_y", "ee_z", "tumor_x", "tumor_y", "tumor_z", "finger_l", "finger_r",
           "target_x", "target_y", "target_z", "grip_cmd"]


def main() -> int:
    ap = make_arg_parser(__doc__)
    ap.add_argument("--out", type=Path, default=Path("demonstrations.csv"))
    args = ap.parse_args()

    glove = open_glove(args)
    sim = SurgerySim(gui=not args.direct)
    new_file = not args.out.exists() or args.out.stat().st_size == 0
    episode, step = int(time.time()), 0
    period = 1.0 / CONTROL_HZ
    print(f"Recording episode {episode} to {args.out}  (R = new episode, Ctrl-C to stop)")

    with args.out.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        try:
            while p.isConnected(sim.client):
                t0 = time.monotonic()
                keys = control_tick(sim, glove)
                if triggered(keys, ord("r")):
                    episode, step = int(time.time()), 0
                    print(f"Recording episode {episode}")
                writer.writerow({"episode": episode, "step": step, "t": step / CONTROL_HZ, **sim.get_state()})
                step += 1
                time.sleep(max(0.0, period - (time.monotonic() - t0)))
        except (KeyboardInterrupt, p.error):
            pass
        finally:
            if p.isConnected(sim.client):
                sim.close()
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
