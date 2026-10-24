"""The one control loop shared by every mode: scripted, keyboard, glove, replay and autonomous.

At each tick: obs_t is read, the controller turns it into an action, that exact (clipped) action
is recorded with obs_t, then the env advances one tick. Nothing from tick t+1 can leak into obs_t.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from ..data.episodes import Episode, EpisodeRecorder, EpisodeStore
from .env import GraspLiftPlaceEnv, Scenario, TaskConfig, clip_action
from .expert import ScriptedExpert

# controller(obs, env) -> (action, expert_label_or_None[, tag]); returning None as action aborts the episode
Controller = Callable[[np.ndarray, GraspLiftPlaceEnv], tuple]


def run_episode(env: GraspLiftPlaceEnv, scenario: Scenario | int, controller: Controller,
                recorder_factory: Callable[[GraspLiftPlaceEnv], EpisodeRecorder] | None = None,
                realtime: bool = False, on_tick: Callable | None = None) -> tuple[dict, Episode | None]:
    obs = env.reset(scenario)
    rec = recorder_factory(env) if recorder_factory else None
    period = env.cfg.dt
    tick = 0
    done = False
    while not done:
        t0 = time.monotonic()
        out = controller(obs, env)
        action, label = out[0], out[1]
        tag = out[2] if len(out) > 2 else None
        if action is None:
            env.stats.termination = "aborted"
            break
        action = clip_action(action, env.cfg.max_speed)
        if rec:
            rec.record(obs, action, tick * period, None if label is None else clip_action(label, env.cfg.max_speed), tag)
        obs, done, info = env.step(action)
        if rec:
            rec.log_step(info)
        if on_tick:
            on_tick(env, obs, action, info)
        tick += 1
        if realtime:
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    outcome = env.stats.to_dict()
    return outcome, (rec.finish(obs, outcome) if rec else None)


def scripted_controller(noise_std: float, rng: np.random.Generator) -> Controller:
    """Expert labels are clean; the *issued* action carries optional velocity noise (DART-style),
    which visits slightly-off states so the cloned policy sees corrections. Both are recorded."""
    expert = ScriptedExpert()

    def act(obs, env):
        if env.stats.ticks == 0:
            expert.reset()
        label = expert.act(env)
        issued = label.copy()
        if noise_std > 0:
            issued[:3] += rng.normal(0.0, noise_std, 3).astype(np.float32)
        return issued, label

    return act


def collect_scripted(out_dir: str, seeds: list[int], noise_std: float = 0.0,
                     cfg: TaskConfig | None = None, gui: bool = False, log=print) -> list[dict]:
    env = GraspLiftPlaceEnv(cfg, gui=gui)
    env.set_label("SCRIPTED")
    store = EpisodeStore(out_dir)
    summaries = []
    try:
        for seed in seeds:
            rng = np.random.default_rng([seed, 7919])
            factory = lambda e, seed=seed: EpisodeRecorder(
                "scripted", e.cfg.to_dict(), e.scenario.to_dict(), e.cfg.dt, episode_id=f"scripted-s{seed:07d}",
                extra={"expert": "ScriptedExpert", "execution_noise_std_mps": noise_std})
            outcome, ep = run_episode(env, seed, scripted_controller(noise_std, rng), factory, realtime=gui)
            store.save(ep)
            summaries.append({"id": ep.id, "success": ep.meta["success"], "termination": ep.meta["termination"],
                              "ticks": len(ep)})
            log(f"[SCRIPTED] {ep.id}: {ep.meta['termination']} ticks={len(ep)} drops={outcome['drops']} "
                f"peak_force={outcome['peak_force_n']:.1f}N")
    finally:
        env.close()
    return summaries


def _collect_chunk(args) -> list[dict]:
    out_dir, seeds, noise_std, cfg_dict = args
    return collect_scripted(out_dir, seeds, noise_std, TaskConfig.from_dict(cfg_dict), log=lambda *_: None)


def collect_scripted_parallel(out_dir: str, episodes: int, seed_start: int = 0, noise_std: float = 0.0,
                              cfg: TaskConfig | None = None, workers: int = 1, log=print) -> list[dict]:
    seeds = list(range(seed_start, seed_start + episodes))
    if workers <= 1:
        return collect_scripted(out_dir, seeds, noise_std, cfg, log=log)
    import multiprocessing as mp
    chunks = [seeds[i::workers] for i in range(workers)]
    cfg_dict = (cfg or TaskConfig()).to_dict()
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_collect_chunk, [(out_dir, c, noise_std, cfg_dict) for c in chunks if c])
    summaries = sorted((s for part in parts for s in part), key=lambda s: s["id"])
    for s in summaries:
        log(f"[SCRIPTED] {s['id']}: {s['termination']} ticks={s['ticks']}")
    return summaries
