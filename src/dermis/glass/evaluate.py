"""Evaluate glass_hold controllers on fixed seed suites (dev / held-out). Results are actual outcomes."""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SUITES = {"dev": range(20_000, 20_030), "heldout": range(30_000, 30_060)}


def make_controller(spec: str):
    """'scripted:<name>' or a checkpoint directory (AUTONOMOUS)."""
    from .controllers import PolicyController, scripted
    if spec.startswith("scripted:"):
        return scripted(spec.split(":", 1)[1])
    from .train import GlassPolicy
    return PolicyController(GlassPolicy.load(spec))


def _run(args):
    spec, seeds = args
    import torch
    torch.set_num_threads(1)
    from .data import run_episode
    from .env import GlassEnv, GlassScenario
    env, ctrl = GlassEnv(), make_controller(spec)
    try:
        return [run_episode(env, ctrl, GlassScenario.sample(s)) for s in seeds]
    finally:
        env.close()


def evaluate(spec: str, suite: str = "heldout", workers: int = 6, out: str | None = None) -> dict:
    seeds = list(SUITES[suite])
    chunks = [seeds[i::workers] for i in range(workers)]
    with Pool(workers) as pool:
        res = sorted((r for part in pool.map(_run, [(spec, c) for c in chunks]) for r in part), key=lambda r: r["seed"])
    terms = [r["termination"] for r in res]
    summ = {"controller": spec, "label": res[0]["label"], "suite": suite, "n": len(res),
            "outcomes": {t: terms.count(t) for t in sorted(set(terms))}, "success": terms.count("success"),
            "max_pod_normal_n_median": round(float(np.median([r["max_pod_normal_n"] for r in res])), 2),
            "max_pod_normal_n_max": round(float(np.max([r["max_pod_normal_n"] for r in res])), 2),
            "mean_effort_nm": round(float(np.mean([r["mean_effort_nm"] for r in res])), 4), "episodes": res}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(summ, indent=1))
    return summ
