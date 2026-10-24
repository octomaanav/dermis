"""Recording-to-learning workflow for the assist demo, reusing cup/teaching.py (snapshot, background
training, status, activation) with separate directories and the assist_hand schema.

    runs/assist/candidates/<name>/     candidate checkpoints (+ eval_dev.json)
    runs/assist/active_checkpoint.json the AUTONOMOUS checkpoint (explicit activation only)
    data/datasets/assist/<name>/       immutable snapshots
The FALLBACK checkpoint ("fallback_scripted") is trained only on SCRIPTED demonstrations
(ASSISTED mode with a constant hold request) and is labelled as such everywhere.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..cup import teaching as T
from .data import collect_parallel, evaluate

ROOT = Path("runs/assist/candidates")
ACTIVE = Path("runs/assist/active_checkpoint.json")
DATASETS = Path("data/datasets/assist")
SCRIPTED_DIR = Path("data/episodes/assist_scripted")
DEMO_DIR = Path("data/episodes/assist_demo")
BC_CONFIG = {"task": "assist_hand", "feature_groups": ["gripper", "object", "contact"], "history": 4,
             "label": "expert_actions", "val_fraction": 0.2, "split_seed": 0, "seed": 0,
             "model": {"hidden": [128, 128], "dropout": 0.0},
             "train": {"epochs": 40, "batch_size": 256, "lr": 0.001, "weight_decay": 0.00001}}


def snapshot(sources: list[str], name: str, **kw) -> Path:
    return T.snapshot(sources, name, out_root=DATASETS, task="assist_hand", **kw)


def train(snap: str, name: str, init_from: str | None = None, epochs: int | None = None, foreground: bool = False) -> Path:
    return T.start_training(snap, name, dict(BC_CONFIG), init_from=init_from, epochs=epochs, foreground=foreground,
                            root=ROOT)


def status(name: str) -> dict:
    return T.status(name, ROOT, ACTIVE)


def candidates() -> list[dict]:
    return [status(d.parent.name) for d in sorted(ROOT.glob("*/candidate.json"))]


def evaluate_candidate(name: str, workers: int = 4) -> dict:
    ck = ROOT / name / "checkpoint"
    if not (ck / "model.pt").exists():
        raise RuntimeError(f"{name} has no checkpoint yet ({status(name)['state']})")
    return evaluate({"kind": "bc", "checkpoint": str(ck)}, workers=workers, out_path=str(ROOT / name / "eval_dev.json"))


def activate(name: str, force: bool = False) -> dict:
    return T.activate(name, force, ROOT, ACTIVE)


def active_checkpoint() -> str | None:
    return T.active_checkpoint(ACTIVE)


def bootstrap(episodes: int = 120, workers: int = 4, log=print) -> dict:
    """Create the clearly labelled fallback: SCRIPTED demos -> snapshot -> train -> dev eval -> activate."""
    t0 = time.time()
    if not SCRIPTED_DIR.exists() or len(list(SCRIPTED_DIR.glob("*.json"))) < episodes:
        collect_parallel(str(SCRIPTED_DIR), episodes, 0, workers)
    log(f"[assist] {episodes} SCRIPTED demonstrations in {SCRIPTED_DIR} ({time.time() - t0:.0f} s)")
    name = "fallback_scripted"
    snap = DATASETS / name
    if not snap.exists():
        snapshot([str(SCRIPTED_DIR)], name)
    if not (ROOT / name / "checkpoint" / "model.pt").exists():
        t1 = time.time()
        train(str(snap), name, foreground=True)
        log(f"[assist] trained {name} in {time.time() - t1:.1f} s wall")
    if not (ROOT / name / "eval_dev.json").exists():
        r = evaluate_candidate(name, workers)
        log(f"[assist] {name} dev: {r['summary']['successes']}/{r['summary']['attempts']} {r['summary']['terminations']}")
    if not active_checkpoint():
        activate(name)
        log(f"[assist] activated {name} (fallback, trained on SCRIPTED demonstrations only)")
    return status(name)
