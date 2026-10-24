"""fragile_cup evaluation: suites, parallel rollouts, outcome summaries, task-meaningfulness check.

Results are written per controller as JSON with every episode's scenario (incl. the full load
schedule), its sha256 `scenario_hash`, termination, failure phase and contact statistics, so it
can be checked that every policy saw identical conditions. These metrics are separate from the
original grasp-lift-place benchmark and are never merged with it.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path

import numpy as np

from ..config import load_config
from .env import CupConfig, CupHoldEnv, CupScenario
from .rollout import run_episode

TRAIN_SEED_LIMIT = 1_000_000
TERMINATIONS = ("success", "drop", "damage_break", "damage_sustained", "excess_slip", "setup_failed", "timeout", "aborted")


def scenario_hash(sc: dict) -> str:
    return hashlib.sha256(json.dumps(sc, sort_keys=True).encode()).hexdigest()[:16]


def suite_scenarios(suite: dict) -> tuple[CupConfig, list[CupScenario]]:
    cfg = CupConfig.from_dict(suite.get("task_config"))
    g = suite["generate"]
    scen = [CupScenario.sample(int(g["seed_start"]) + i, cfg) for i in range(int(g["count"]))]
    if any(s.seed < TRAIN_SEED_LIMIT for s in scen):
        raise ValueError("evaluation suites must use seeds >= 1,000,000 (training uses 0..)")
    return cfg, scen


def _make_controller(spec: dict):
    from .controllers import PolicyController, scripted
    if spec["kind"] == "scripted":
        return scripted(spec["name"])
    from .train import CupPolicy
    return PolicyController(CupPolicy.load(spec["checkpoint"]))


def _run_chunk(args) -> list[dict]:
    spec, cfg_dict, scen_dicts, gui = args
    import torch
    torch.set_num_threads(1)
    cfg = CupConfig.from_dict(cfg_dict)
    env = CupHoldEnv(cfg)
    viewer = None
    if gui:                                           # display only; physics identical to headless runs
        from .viewer import CupViewer
        viewer = CupViewer(env, controls=False)
    ctrl = _make_controller(spec)
    out = []
    tick = (lambda e, *_: (viewer.sync(e), viewer.status(e))) if viewer else None
    try:
        for sd in scen_dicts:
            sc = CupScenario.from_dict(sd)
            if viewer:
                viewer.text("mode", f"{ctrl.label}: {ctrl.name}   scenario seed {sc.seed}", 0, (0.75, 0.1, 0.1))
            outcome, _ = run_episode(env, sc, ctrl, pour_at_s=cfg.pre_pour_s, realtime=gui, on_tick=tick)
            out.append({"scenario": sd, "scenario_hash": scenario_hash(sd), "peak_schedule_load_n": sc.schedule.peak,
                        **outcome})
    finally:
        env.close()
        if viewer:
            viewer.close()
    return out


def run_controller(spec: dict, cfg: CupConfig, scen: list[CupScenario], workers: int = 1, gui: bool = False) -> list[dict]:
    sds = [s.to_dict() for s in scen]
    if gui or workers <= 1:
        return _run_chunk((spec, cfg.to_dict(), sds, gui))
    chunks = [sds[i::workers] for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_run_chunk, [(spec, cfg.to_dict(), c, False) for c in chunks if c])
    by = {r["scenario"]["seed"]: r for part in parts for r in part}
    return [by[s["seed"]] for s in sds]


def summarize(eps: list[dict]) -> dict:
    n = len(eps)
    term = {t: sum(1 for e in eps if e["termination"] == t) for t in TERMINATIONS}
    term = {k: v for k, v in term.items() if v}
    fail_phase: dict = {}
    for e in eps:
        if not e["success"]:
            key = f"{e['termination']}@{e['failure_phase']}"
            fail_phase[key] = fail_phase.get(key, 0) + 1
    by_mu: dict = {}
    for e in eps:
        b = "low (<0.53)" if e["scenario"]["friction"] < 0.53 else "mid (0.53-0.72)" if e["scenario"]["friction"] < 0.72 else "high (>=0.72)"
        by_mu.setdefault(b, []).append(e["success"])
    fails = [e for e in eps if not e["success"]]
    return {
        "attempts": n, "successes": term.get("success", 0), "success_rate": term.get("success", 0) / n if n else 0,
        "terminations": term, "failure_state_distribution": dict(sorted(fail_phase.items())),
        "success_by_friction": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_mu.items())},
        "peak_normal_n_median": statistics.median(e["peak_normal_n"] for e in eps),
        "peak_normal_n_max": max(e["peak_normal_n"] for e in eps),
        "mean_effort_n_median": statistics.median(e["mean_effort_n"] for e in eps),
        "damage_ns_median": statistics.median(e["damage_ns"] for e in eps),
        "final_slip_mm_median": 1000 * statistics.median(e["final_slip_m"] for e in eps),
        "load_at_failure_n_median": statistics.median(e["load_at_end_n"] for e in fails) if fails else None,
    }


def evaluate(spec: dict, suite_path: str, out_path: str | None, workers: int, gui: bool = False,
             damage_model: str | None = None, log=print) -> dict:
    suite = load_config(suite_path)
    if damage_model:
        suite = {**suite, "task_config": {**(suite.get("task_config") or {}), "damage_model": damage_model}}
    cfg, scen = suite_scenarios(suite)
    t0 = time.time()
    eps = run_controller(spec, cfg, scen, workers, gui)
    name = spec.get("name") or Path(spec["checkpoint"]).name
    label = "SCRIPTED" if spec["kind"] == "scripted" else "AUTONOMOUS"
    res = {"task": "fragile_cup", "suite": suite.get("name"), "suite_path": str(suite_path),
           "damage_model": cfg.damage_model, "controller": {"name": name, "label": label, **spec},
           "note": "Simulation benchmark: rigid cup, external-force load proxy, simulated damage threshold.",
           "summary": summarize(eps), "episodes": eps, "wall_seconds": round(time.time() - t0, 1)}
    if spec["kind"] == "bc":
        meta = json.loads((Path(spec["checkpoint"]) / "metadata.json").read_text())
        res["controller"]["training"] = {k: meta[k] for k in ("feature_groups", "history", "provenance")}
        overlap = set(meta["provenance"]["train_seeds"]) & {s.seed for s in scen}
        assert not overlap, f"evaluation seeds overlap training seeds {sorted(overlap)[:5]}"
    s = res["summary"]
    log(f"[{label}] {name} on {suite.get('name')} ({cfg.damage_model}): {s['successes']}/{s['attempts']}  "
        f"{s['terminations']}  peak normal median {s['peak_normal_n_median']:.1f} N")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(res, indent=1))
    return res


def check_task(suite_path: str, efforts: list[float], out_dir: str, workers: int, damage_models: list[str],
               log=print) -> dict:
    """Is adaptation needed? Fixed efforts vs scripted feedback controllers vs privileged oracle."""
    names = [f"fixed_{e:g}N" for e in efforts] + ["slip_reflex", "grip_load_expert", "oracle_privileged"]
    table = {}
    for dm in damage_models:
        rows = []
        for n in names:
            r = evaluate({"kind": "scripted", "name": n}, suite_path, f"{out_dir}/{dm}/{n}.json", workers,
                         damage_model=dm, log=log)
            s = r["summary"]
            rows.append({"controller": n, "success": f"{s['successes']}/{s['attempts']}",
                         "success_rate": s["success_rate"], "terminations": s["terminations"],
                         "peak_normal_n_median": round(s["peak_normal_n_median"], 2),
                         "mean_effort_n_median": round(s["mean_effort_n_median"], 2)})
        fixed_ok = [r["controller"] for r in rows if r["controller"].startswith("fixed_") and r["success_rate"] == 1.0]
        table[dm] = {"rows": rows, "fixed_efforts_solving_all": fixed_ok}
        log(f"== {dm}: fixed efforts that solve every scenario: {fixed_ok or 'none'}")
    lines = ["# fragile_cup task check", "",
             f"Suite `{suite_path}`. Simulation; damage = simulated contact-load threshold.", ""]
    for dm, t in table.items():
        lines += [f"## damage model: {dm}", "", "| Controller | Success | Terminations | Peak normal N (median) | Mean effort N (median) |",
                  "|---|---|---|---|---|"]
        lines += [f"| {r['controller']} | {r['success']} | {r['terminations']} | {r['peak_normal_n_median']} | "
                  f"{r['mean_effort_n_median']} |" for r in t["rows"]]
        lines += ["", f"Fixed efforts that succeed in every scenario: {', '.join(t['fixed_efforts_solving_all']) or 'none'}", ""]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "task_check.md").write_text("\n".join(lines))
    Path(out_dir, "task_check.json").write_text(json.dumps(table, indent=1))
    return table
