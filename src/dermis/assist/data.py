"""assist_hand recording, scripted demonstrations and development evaluation.

Episode (shared EpisodeStore format, meta["task"] == "assist_hand"):
    obs[t]            simulator observation before the action (OBS_NAMES); never glove readings
    actions[t]        FINAL [closure, effort] applied to the hand at t (what the robot actually did)
    human_g[t]        the human grip request (NaN in AUTONOMOUS / SCRIPTED)
    human_actions[t]  the action the human request alone maps to (NaN when no human)
    assist_effort[t]  scripted assistance effort (NaN when assistance was not computing)
    tags[t]           control mode; host_ns[t] host monotonic time (links to raw glove recordings)
    load_n, normal_index/middle/palm, support_z, disp_m, hold_state, phase: SIMULATED, for inspection
Glove sensor data is never stored here; it is recorded separately with host receive times.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import statistics
import time
import uuid
from pathlib import Path

import numpy as np

from ..cup.env import CupConfig, CupScenario
from ..data.episodes import Episode, EpisodeStore
from .control import ASSISTED, AUTONOMOUS, MANUAL, TASK, ModeController
from .env import ACTION_NAMES, OBS_NAMES, AssistHandEnv, default_config

SOURCES = ("keyboard", "glove", "scripted", "autonomous")


class AssistRecorder:
    def __init__(self, source: str, mode: str, env: AssistHandEnv, controller: str, extra: dict | None = None):
        assert source in SOURCES
        self.meta = {"format_version": 1, "task": TASK, "source": source, "label": source.upper(), "mode": mode,
                     "controller": controller,
                     "id": f"assist-{source}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}",
                     "created_unix": time.time(), "dt": env.cfg.dt, "task_config": env.cfg.to_dict(),
                     "scenario": env.scenario.to_dict(), "seed": env.scenario.seed, "obs_names": list(OBS_NAMES),
                     "action_names": list(ACTION_NAMES), "simulated_fields": ["load_n", "normal_index", "normal_middle",
                     "normal_palm", "support_z", "disp_m"], **(extra or {})}
        self.r = {k: [] for k in ("obs", "actions", "human_g", "human_actions", "assist_effort", "tags", "host_ns",
                                  "load_n", "normal_index", "normal_middle", "normal_palm", "support_z", "disp_m",
                                  "hold_state", "phase", "closure_meas")}

    def record(self, obs, final, trace: dict, mode: str) -> None:
        r = self.r
        r["obs"].append(np.asarray(obs, np.float32))
        r["actions"].append(np.asarray(final, np.float32))
        r["human_g"].append(np.nan if trace.get("human_g") is None else trace["human_g"])
        r["human_actions"].append([np.nan, np.nan] if trace.get("human_action") is None else trace["human_action"])
        r["assist_effort"].append(np.nan if trace.get("assist_effort") is None else trace["assist_effort"])
        r["tags"].append(mode)
        r["host_ns"].append(time.monotonic_ns())

    def log(self, info: dict) -> None:
        for k in ("load_n", "normal_index", "normal_middle", "normal_palm", "support_z", "disp_m", "hold_state",
                  "phase", "closure_meas"):
            self.r[k].append(info[k])

    def finish(self, final_obs, outcome: dict) -> Episode:
        r = self.r
        arr = {"obs": np.array(r["obs"], np.float32).reshape(-1, len(OBS_NAMES)),
               "actions": np.array(r["actions"], np.float32).reshape(-1, 2),
               "human_g": np.array(r["human_g"], np.float32), "human_actions": np.array(r["human_actions"], np.float32).reshape(-1, 2),
               "assist_effort": np.array(r["assist_effort"], np.float32), "tags": np.array(r["tags"], dtype="U24"),
               "host_ns": np.array(r["host_ns"], np.int64), "final_obs": np.asarray(final_obs, np.float32),
               **{k: np.array(r[k], np.float32) for k in ("load_n", "normal_index", "normal_middle", "normal_palm",
                                                           "support_z", "disp_m", "closure_meas")},
               "hold_state": np.array(r["hold_state"], dtype="U28"), "phase": np.array(r["phase"], dtype="U16")}
        if self.meta["source"] == "scripted":
            arr["expert_actions"] = arr["actions"].copy()
        meta = {**self.meta, "length": len(r["obs"]), "outcome": outcome, "success": bool(outcome.get("success")),
                "termination": outcome.get("termination")}
        return Episode(meta, arr)


def run_episode(env: AssistHandEnv, scenario, ctrl: ModeController, human_g=None, recorder=None,
                pour_at_s: float | None = 1.0) -> dict:
    """Headless episode. human_g: float, or callable(env, t) -> float, or None (AUTONOMOUS)."""
    obs = env.reset(scenario, pour_at_s=pour_at_s)
    ctrl.reset()
    while not env.done:
        g = human_g(env, env.t) if callable(human_g) else human_g
        a, trace = ctrl.act(obs, None if ctrl.mode == AUTONOMOUS else g)
        if recorder:
            recorder.record(obs, a, trace, ctrl.mode)
        obs, _, info = env.step(a)
        if recorder:
            recorder.log(info)
    out = {**env.stats.to_dict(), "pour_t_s": env.pour_t0}
    return out, (recorder.finish(obs, out) if recorder else None)


SCRIPTED_G = 0.35       # the scripted demonstrator: ASSISTED mode with a constant moderate hold request


def collect(out_dir: str, seeds: list[int], cfg_dict: dict | None = None) -> list[dict]:
    cfg = CupConfig.from_dict(cfg_dict) if cfg_dict else default_config()
    env = AssistHandEnv(cfg)
    store = EpisodeStore(out_dir)
    res = []
    try:
        for s in seeds:
            ctrl = ModeController(cfg.dt)
            ctrl.switch(ASSISTED)
            sc = CupScenario.sample(s, cfg)
            env.reset(sc, pour_at_s=1.0)
            rec = AssistRecorder("scripted", ASSISTED, env, f"scripted_assist_hold_g{SCRIPTED_G}")
            out, ep = run_episode(env, sc, ctrl, SCRIPTED_G, rec)
            ep.meta["id"] = f"assist-scripted-s{s:07d}"
            store.save(ep)
            res.append({"id": ep.meta["id"], "termination": out["termination"]})
    finally:
        env.close()
    return res


def collect_parallel(out_dir: str, n: int, seed_start: int = 0, workers: int = 4) -> list[dict]:
    seeds = list(range(seed_start, seed_start + n))
    if workers <= 1:
        return collect(out_dir, seeds)
    chunks = [seeds[i::workers] for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.starmap(collect, [(out_dir, c, None) for c in chunks if c])
    return [r for part in parts for r in part]


DEV_SEEDS = list(range(1_000_000, 1_000_020))


def _eval_chunk(spec: dict, seeds: list[int]) -> list[dict]:
    import torch
    torch.set_num_threads(1)
    cfg = default_config()
    env = AssistHandEnv(cfg)
    ctrl = ModeController(cfg.dt)
    g = None
    if spec["kind"] == "bc":
        from ..cup.train import CupPolicy
        ctrl.set_policy(CupPolicy.load(spec["checkpoint"]), {"checkpoint": spec["checkpoint"]})
        ctrl.switch(AUTONOMOUS)
    else:
        ctrl.switch(ASSISTED if spec["kind"] == "assisted" else MANUAL)
        g = spec["g"]
    out = []
    try:
        for s in seeds:
            sc = CupScenario.sample(s, cfg)
            r, _ = run_episode(env, sc, ctrl, g)
            out.append({"seed": s, **r})
    finally:
        env.close()
    return out


def evaluate(spec: dict, seeds: list[int] = DEV_SEEDS, workers: int = 4, out_path: str | None = None) -> dict:
    t0 = time.time()
    if workers <= 1:
        eps = _eval_chunk(spec, seeds)
    else:
        chunks = [seeds[i::workers] for i in range(workers)]
        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.starmap(_eval_chunk, [(spec, c) for c in chunks if c])
        eps = sorted((e for part in parts for e in part), key=lambda e: e["seed"])
    terms: dict = {}
    for e in eps:
        terms[e["termination"]] = terms.get(e["termination"], 0) + 1
    res = {"task": TASK, "suite": "assist_dev (seeds 1,000,000-1,000,019)", "spec": spec,
           "summary": {"attempts": len(eps), "successes": terms.get("success", 0), "terminations": terms,
                       "peak_normal_n_median": statistics.median(e["peak_normal_n"] for e in eps),
                       "mean_effort_median": statistics.median(e["mean_effort"] for e in eps)},
           "episodes": eps, "wall_seconds": round(time.time() - t0, 1)}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(res, indent=1, default=float))
    return res
