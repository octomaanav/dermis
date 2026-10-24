"""fragile_cup episode loop, recorder and scripted collection.

Same contract as the original benchmark: at each tick obs_t is read, the controller turns it
into an effort, that exact (clipped) effort is recorded with obs_t, then the env advances.
Episodes are saved with the shared EpisodeStore (npz + json) under a cup-specific schema
(meta["task"] == "fragile_cup"), with source scripted | keyboard | glove.
"""
from __future__ import annotations

import time
import uuid

import numpy as np

from ..data.episodes import Episode, EpisodeStore
from .env import ACTION_NAMES, OBS_NAMES, CupConfig, CupHoldEnv, CupScenario, clip_effort

TASK = "fragile_cup"
SOURCES = ("scripted", "keyboard", "glove")


class CupRecorder:
    def __init__(self, source: str, cfg: CupConfig, scenario: CupScenario, controller_name: str,
                 episode_id: str | None = None, extra: dict | None = None):
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        self.meta = {"format_version": 1, "task": TASK,
                     "id": episode_id or f"cup-{source}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}",
                     "source": source, "label": source.upper(), "controller": controller_name,
                     "created_unix": time.time(), "dt": cfg.dt, "task_config": cfg.to_dict(),
                     "scenario": scenario.to_dict(), "seed": scenario.seed, "obs_names": list(OBS_NAMES),
                     "action_names": list(ACTION_NAMES), "calibration_id": None, **(extra or {})}
        self.rows = {k: [] for k in ("obs", "actions", "expert_actions", "t", "tags", "host_ns", "load_n", "normal_l",
                                     "normal_r", "shear_z_l", "shear_z_r", "slip_m", "phase")}

    def record(self, obs, effort: float, t: float, label, tag: str | None) -> None:
        self.rows["obs"].append(np.asarray(obs, np.float32))
        self.rows["actions"].append([effort])
        self.rows["expert_actions"].append([np.nan if label is None else float(label)])
        self.rows["t"].append(t)
        self.rows["tags"].append(tag or "")
        self.rows["host_ns"].append(time.monotonic_ns())     # links this tick to raw sensor recordings (host clock)

    def log(self, info: dict) -> None:
        for k in ("load_n", "normal_l", "normal_r", "shear_z_l", "shear_z_r", "slip_m", "phase"):
            self.rows[k].append(info[k])

    def finish(self, final_obs, outcome: dict) -> Episode:
        r = self.rows
        arrays = {"obs": np.array(r["obs"], np.float32).reshape(-1, len(OBS_NAMES)),
                  "actions": np.array(r["actions"], np.float32).reshape(-1, 1),
                  "t": np.array(r["t"]), "tags": np.array(r["tags"], dtype="U24"),
                  "host_ns": np.array(r["host_ns"], dtype=np.int64),
                  "final_obs": np.asarray(final_obs, np.float32),
                  # logged for inspection only: load_n is privileged and never a policy input
                  **{k: np.array(r[k], np.float32) for k in ("load_n", "normal_l", "normal_r", "shear_z_l",
                                                             "shear_z_r", "slip_m")},
                  "phase": np.array(r["phase"], dtype="U16")}
        exp = np.array(r["expert_actions"], np.float32).reshape(-1, 1)
        if len(exp) and np.isfinite(exp).all():
            arrays["expert_actions"] = exp
        meta = {**self.meta, "length": len(r["obs"]), "outcome": outcome,
                "success": bool(outcome.get("success")), "termination": outcome.get("termination")}
        return Episode(meta, arrays)


def run_episode(env: CupHoldEnv, scenario, controller, recorder: CupRecorder | None = None,
                pour_at_s: float | None = 1.0, realtime: bool = False, on_tick=None,
                noise=None) -> tuple[dict, Episode | None]:
    obs = env.reset(scenario, pour_at_s=pour_at_s)
    controller.reset()
    tick = 0
    while not env.done:
        t0 = time.monotonic()
        effort, label, tag = controller.act(obs, env)
        if effort is None:
            env.stats.termination = "aborted"
            break
        if noise is not None:
            effort = effort + noise()
        effort = clip_effort(effort)
        if recorder:
            recorder.record(obs, effort, tick * env.cfg.dt, label, tag)
        obs, done, info = env.step(effort)
        if recorder:
            recorder.log(info)
        if on_tick:
            on_tick(env, obs, effort, info)
        tick += 1
        if realtime:
            time.sleep(max(0.0, env.cfg.dt - (time.monotonic() - t0)))
    outcome = env.stats.to_dict()
    outcome.pop("events", None)
    outcome["pour_t_s"] = env.pour_t0
    return outcome, (recorder.finish(obs, outcome) if recorder else None)


def collect(out_dir: str, seeds: list[int], controller_name: str, noise_std: float, cfg_dict: dict) -> list[dict]:
    from .controllers import scripted
    cfg = CupConfig.from_dict(cfg_dict)
    env = CupHoldEnv(cfg)
    store = EpisodeStore(out_dir)
    out = []
    try:
        for seed in seeds:
            rng = np.random.default_rng([seed, 99])
            ctrl = scripted(controller_name)
            sc = CupScenario.sample(seed, cfg)
            rec = CupRecorder("scripted", cfg, sc, ctrl.name, episode_id=f"cup-scripted-s{seed:07d}",
                              extra={"execution_noise_std_n": noise_std})
            outcome, ep = run_episode(env, sc, ctrl, rec, noise=(lambda: rng.normal(0, noise_std)) if noise_std else None)
            store.save(ep)
            out.append({"id": ep.id, "termination": outcome["termination"], "ticks": len(ep)})
    finally:
        env.close()
    return out


def collect_parallel(out_dir: str, episodes: int, seed_start: int, controller_name: str, noise_std: float,
                     cfg: CupConfig, workers: int) -> list[dict]:
    seeds = list(range(seed_start, seed_start + episodes))
    if workers <= 1:
        return collect(out_dir, seeds, controller_name, noise_std, cfg.to_dict())
    import multiprocessing as mp
    chunks = [seeds[i::workers] for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.starmap(collect, [(out_dir, c, controller_name, noise_std, cfg.to_dict()) for c in chunks if c])
    return sorted((r for part in parts for r in part), key=lambda r: r["id"])
