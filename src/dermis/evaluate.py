"""Evaluation suites, autonomous rollouts and the results format.

A suite (YAML or JSON) lists explicit scenarios, or a deterministic generator for them:
    name: heldout_v1
    task_config: {}                     # TaskConfig overrides, applied to every scenario
    generate: {seed_start: 1000000, count: 50, frictions: [0.4, 0.6, 0.8, 1.0]}
    # or: scenarios: [{seed, object_xy, target_xy, friction}, ...]

Training data uses scenario seeds from 0 upward; suites use seeds >= 1,000,000 (dev) and
>= 2,000,000 (held-out), so the scenarios never coincide. Policy runs are AUTONOMOUS: the
policy sees only env observations. The scripted expert can be run on the same suite as a
sanity check of the environment (labelled SCRIPTED, not a learned result).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import statistics
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .sim.env import GraspLiftPlaceEnv, Scenario, TaskConfig
from .sim.rollout import run_episode, scripted_controller

TRAIN_SEED_LIMIT = 1_000_000


def suite_scenarios(suite: dict) -> list[Scenario]:
    cfg = TaskConfig.from_dict(suite.get("task_config"))
    if "scenarios" in suite:
        scen = [Scenario.from_dict(s) for s in suite["scenarios"]]
    else:
        g = suite["generate"]
        frictions = g.get("frictions")
        scen = []
        for i in range(int(g["count"])):
            s = Scenario.sample(int(g["seed_start"]) + i, cfg)
            if frictions:
                s.friction = float(frictions[i % len(frictions)])
            scen.append(s)
    bad = [s.seed for s in scen if s.seed < TRAIN_SEED_LIMIT]
    if bad:
        raise ValueError(f"suite {suite.get('name')} uses training-range seeds {bad[:5]}")
    return scen


def materialize_suite(spec_path: str, out_path: str) -> dict:
    suite = load_config(spec_path)
    scen = suite_scenarios(suite)
    out = {k: v for k, v in suite.items() if k != "generate"}
    out["generated_from"] = {"spec": str(spec_path), **suite.get("generate", {})}
    out["scenarios"] = [s.to_dict() for s in scen]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=1))
    return out


# -- workers -------------------------------------------------------------------

def _run_chunk(args) -> list[dict]:
    controller_spec, task_cfg, scenarios, gui = args
    import torch
    torch.set_num_threads(1)
    env = GraspLiftPlaceEnv(TaskConfig.from_dict(task_cfg), gui=gui)
    out = []
    try:
        if controller_spec["kind"] == "scripted":
            env.set_label("SCRIPTED (sanity check)")
        else:
            from .bc.model import BCPolicy
            policy = BCPolicy.load(controller_spec["checkpoint"])
            env.set_label("AUTONOMOUS")
        for sd in scenarios:
            sc = Scenario.from_dict(sd)
            if controller_spec["kind"] == "scripted":
                ctrl = scripted_controller(0.0, np.random.default_rng(0))
            else:
                policy.reset()
                ctrl = lambda obs, _env: (policy.act(obs), None)      # obs only: no privileged state
            outcome, _ = run_episode(env, sc, ctrl, realtime=gui)
            outcome.pop("events", None)
            out.append({"scenario": sd, **outcome})
    finally:
        env.close()
    return out


def run_suite(controller_spec: dict, suite: dict, workers: int = 1, gui: bool = False) -> list[dict]:
    scen = [s.to_dict() for s in suite_scenarios(suite)]
    task_cfg = suite.get("task_config") or {}
    if gui or workers <= 1:
        return _run_chunk((controller_spec, task_cfg, scen, gui))
    chunks = [scen[i::workers] for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_run_chunk, [(controller_spec, task_cfg, c, False) for c in chunks if c])
    by_seed = {r["scenario"]["seed"]: r for part in parts for r in part}
    return [by_seed[s["seed"]] for s in scen]


# -- metrics ---------------------------------------------------------------------

def summarize(episodes: list[dict], excess_force_n: float) -> dict:
    n = len(episodes)
    succ = [e for e in episodes if e["success"]]
    terms: dict[str, int] = {}
    for e in episodes:
        terms[e["termination"]] = terms.get(e["termination"], 0) + 1
    peaks = [e["peak_force_n"] for e in episodes]
    times = [e["completion_time_s"] for e in succ]
    by_friction: dict[str, list] = {}
    for e in episodes:
        by_friction.setdefault(f"{e['scenario']['friction']:.2f}", []).append(e["success"])
    return {
        "attempts": n,
        "successes": len(succ),
        "success_rate": len(succ) / n if n else 0.0,
        "episodes_with_drop": sum(1 for e in episodes if e["drops"] > 0),
        "total_drops": sum(e["drops"] for e in episodes),
        "excess_force_threshold_n": excess_force_n,
        "excess_force_threshold_note": "simulated benchmark threshold on per-finger normal force; not a clinical measure",
        "episodes_with_excess_force": sum(1 for e in episodes if e["excess_force_ticks"] > 0),
        "peak_contact_force_n_max": max(peaks) if peaks else None,
        "peak_contact_force_n_median": statistics.median(peaks) if peaks else None,
        "completion_time_s_mean": statistics.mean(times) if times else None,
        "completion_time_s_median": statistics.median(times) if times else None,
        "terminations": terms,
        "timeouts": terms.get("timeout", 0),
        "never_lifted": sum(1 for e in episodes if not e["was_lifted"]),
        "success_by_friction": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_friction.items())},
    }


def evaluate(checkpoint: str | None, suite_path: str, out_path: str | None = None, workers: int = 1,
             gui: bool = False, scripted: bool = False, log=print) -> dict:
    suite = load_config(suite_path)
    task_cfg = TaskConfig.from_dict(suite.get("task_config"))
    if scripted:
        spec = {"kind": "scripted"}
        policy_info = {"kind": "scripted_expert", "label": "SCRIPTED", "note": "environment sanity check, not learned"}
    else:
        if not checkpoint:
            raise SystemExit("evaluate needs --checkpoint (or --scripted for the expert sanity check)")
        meta = json.loads((Path(checkpoint) / "metadata.json").read_text())
        spec = {"kind": "bc", "checkpoint": str(checkpoint)}
        prov = meta["provenance"]
        policy_info = {
            "kind": "behavioral_cloning", "label": "AUTONOMOUS", "name": meta["name"], "checkpoint": str(checkpoint),
            "feature_groups": meta["feature_groups"], "history": meta["history"],
            "demonstrations": {"train_episodes": len(prov["train_episode_ids"]),
                               "val_episodes": len(prov["val_episode_ids"]), "sources": prov["sources"],
                               "data_dir": prov["data_dir"], "data_digest": prov["data_digest"]},
        }
        overlap = set(prov.get("train_seeds") or []) & {s.seed for s in suite_scenarios(suite)}
        if overlap:
            raise SystemExit(f"evaluation seeds overlap training seeds: {sorted(overlap)[:5]}")
    t0 = time.time()
    log(f"[{policy_info['label']}] evaluating on suite {suite.get('name')} ({len(suite_scenarios(suite))} scenarios)")
    episodes = run_suite(spec, suite, workers=workers, gui=gui)
    result = {
        "suite": suite.get("name"), "suite_path": str(suite_path),
        "experiment_note": "Simulated contact observations only. Says nothing about the value of real glove data.",
        "policy": policy_info,
        "summary": summarize(episodes, task_cfg.excess_force_n),
        "episodes": episodes,
        "wall_seconds": round(time.time() - t0, 1),
        "created_unix": time.time(),
    }
    s = result["summary"]
    log(f"[{policy_info['label']}] success {s['successes']}/{s['attempts']}  drops(ep) {s['episodes_with_drop']}  "
        f"excess-force(ep) {s['episodes_with_excess_force']}  peak {s['peak_contact_force_n_max']:.1f} N  "
        f"terminations {s['terminations']}")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=1))
        log(f"wrote {out_path}")
    return result


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) // 2))
