#!/usr/bin/env python3
"""Behavioral cloning: train an MLP that maps state -> action and save it to surgeon_model.pth.

    python train_model.py              # demonstrations.csv (generated as 100 dummy rows if missing)
    python train_model.py --dummy      # train on fresh dummy data in dummy_demonstrations.csv instead

Inputs:  gripper_x, gripper_y, gripper_z, tumor_x, tumor_y, tumor_z, finger_opening
Outputs: target_grip_angle (0 open .. 1 closed), target_move_x/y/z (metres to move this tick)

finger_opening (metres per finger) is what lets the policy tell "about to grab" from "holding
it": at the bottom of the hole the positions are identical before and after the fingers close.

record_demo.py and generate_demos.py write a richer CSV (ee_*, finger_*, target_*, grip_cmd);
load_demos() converts it, so recordings, scripted demos and dummy data train the same way.
Existing CSVs are never overwritten.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

INPUTS = ["gripper_x", "gripper_y", "gripper_z", "tumor_x", "tumor_y", "tumor_z", "finger_opening"]
OUTPUTS = ["target_grip_angle", "target_move_x", "target_move_y", "target_move_z"]
HIDDEN = 64
MODEL_PATH = Path("surgeon_model.pth")


def state_inputs(s) -> list:
    """INPUTS from a SurgerySim.get_state() dict, or column-wise from a record_demo-format DataFrame."""
    return [s["ee_x"], s["ee_y"], s["ee_z"], s["tumor_x"], s["tumor_y"], s["tumor_z"],
            (s["finger_l"] + s["finger_r"]) / 2]


class SurgeonMLP(nn.Module):
    def __init__(self, n_in: int = len(INPUTS), n_out: int = len(OUTPUTS), hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_dummy_csv(path: Path, rows: int = 100, seed: int = 0) -> None:
    """Random but plausibly scaled rows: positions inside the sim workspace, small per-tick moves."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "gripper_x": rng.uniform(0.40, 0.70, rows),
        "gripper_y": rng.uniform(-0.15, 0.15, rows),
        "gripper_z": rng.uniform(0.02, 0.30, rows),
        "tumor_x": rng.uniform(0.53, 0.57, rows),
        "tumor_y": rng.uniform(-0.02, 0.02, rows),
        "tumor_z": rng.uniform(0.025, 0.030, rows),
        "finger_opening": rng.uniform(0.0, 0.04, rows),
        "target_grip_angle": rng.uniform(0.0, 1.0, rows),
        "target_move_x": rng.uniform(-0.005, 0.005, rows),
        "target_move_y": rng.uniform(-0.005, 0.005, rows),
        "target_move_z": rng.uniform(-0.005, 0.005, rows),
    })
    df.to_csv(path, index=False)
    print(f"Wrote {rows} dummy rows to {path}")


def load_demos(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if set(INPUTS + OUTPUTS) <= set(df.columns):
        return df[INPUTS + OUTPUTS]
    if {"ee_x", "finger_l", "target_x", "grip_cmd"} <= set(df.columns):   # record_demo.py format
        out = pd.DataFrame(dict(zip(INPUTS, state_inputs(df))))
        out["target_grip_angle"] = df.grip_cmd
        out["target_move_x"] = df.target_x - df.ee_x
        out["target_move_y"] = df.target_y - df.ee_y
        out["target_move_z"] = df.target_z - df.ee_z
        return out
    raise ValueError(f"{path} has neither the training columns nor record_demo.py's columns")


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("demonstrations.csv"))
    ap.add_argument("--dummy", action="store_true", help="train on 100 fresh random rows (dummy_demonstrations.csv)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    args = ap.parse_args()

    csv_path = Path("dummy_demonstrations.csv") if args.dummy else args.csv
    if args.dummy or not csv_path.exists():
        make_dummy_csv(csv_path)
    df = load_demos(csv_path)
    print(f"Training on {len(df)} rows from {csv_path}")

    x = torch.tensor(df[INPUTS].values, dtype=torch.float32)
    y = torch.tensor(df[OUTPUTS].values, dtype=torch.float32)
    # Standardize both sides: positions are ~0.5 m but moves are ~mm, so raw MSE would ignore the moves.
    # Input std is floored at 1 cm: a tumor that never moved in the demos would otherwise turn
    # sub-millimetre jitter into huge inputs at rollout.
    x_mean, x_std = x.mean(0), x.std(0).clamp_min(0.01)
    y_mean, y_std = y.mean(0), y.std(0).clamp_min(1e-6)

    device = pick_device()
    xs, ys = ((x - x_mean) / x_std).to(device), ((y - y_mean) / y_std).to(device)
    model = SurgeonMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    print(f"Device: {device}")

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(xs), device=device)
        total = 0.0
        for i in range(0, len(xs), args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = loss_fn(model(xs[idx]), ys[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if epoch == 1 or epoch % 50 == 0:
            print(f"epoch {epoch:4d}  mse {total / len(xs):.4f}")

    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
        "inputs": INPUTS, "outputs": OUTPUTS, "hidden": HIDDEN,
    }, args.out)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
