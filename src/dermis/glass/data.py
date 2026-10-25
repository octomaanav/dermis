"""Run glass_hold episodes and record them.

Per episode, record() writes:
  <id>.frames.jsonl   one SYNTHETIC glove frame per control tick, exactly in the dermis.glass.schema layout
                      (line 1 is the unloaded baseline frame captured with the fingers open)
  <id>.trace.jsonl    header (config, scenario, controller, label) + per-tick simulator state
  <id>.npz            X (features from the frame seen BEFORE each action), executed effort, clean label
Demonstrations: the SCRIPTED load_expert, executed with Gaussian effort noise (DART-style) so the data
contains recoveries; the label stored is the expert's clean (noise-free) command.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .controllers import scripted
from .env import EFFORT_HIGH, EFFORT_LOW, GlassConfig, GlassEnv, GlassScenario
from .features import NAMES, frame_features


def run_episode(env: GlassEnv, ctrl, sc: GlassScenario, out_dir: str | None = None, ep_id: str | None = None,
                noise: float = 0.0, label: str | None = None) -> dict:
    rng = np.random.default_rng(sc.seed + 7)
    frame = env.reset(sc)
    ctrl.reset(env)
    label = label or ctrl.label
    X, executed, clean, rows, frames = [], [], [], [], [env.baseline_frame, frame]
    while not env.done:
        x = frame_features(env.frame, env.baseline_frame, env.effort)
        a = float(ctrl.act(env))
        e = float(np.clip(a + (rng.normal(0, noise) if noise else 0.0), EFFORT_LOW, EFFORT_HIGH))
        frame = env.step(e)
        X.append(x)
        executed.append(e)
        clean.append(a)
        frames.append(frame)
        rows.append(env.state())
    res = {"seed": sc.seed, "controller": ctrl.name, "label": label, "termination": env.termination,
           "t_end": round(env.t, 3), "peak_normal_n": {k: round(v, 2) for k, v in env.peak_normal.items()},
           "max_pod_normal_n": round(max(env.peak_normal.values()), 2),
           "mean_effort_nm": round(float(np.mean(executed)), 4), "final_drop_cm": rows[-1]["drop_cm"],
           "water_kg": sc.water_kg}
    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        ep_id = ep_id or f"{ctrl.name}_{sc.seed}"
        with open(d / f"{ep_id}.frames.jsonl", "w") as fh:
            for f in frames:
                fh.write(json.dumps(f, separators=(",", ":")) + "\n")
        with open(d / f"{ep_id}.trace.jsonl", "w") as fh:
            fh.write(json.dumps({"header": True, "task": "glass_hold", "task_config": env.cfg.to_dict(),
                                 "scenario": sc.to_dict(), "controller": ctrl.name, "label": label,
                                 "end_t": env.end_t, "noise": noise, "result": res}) + "\n")
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        np.savez_compressed(d / f"{ep_id}.npz", X=np.array(X), executed=np.array(executed, np.float32),
                            label=np.array(clean, np.float32), seed=sc.seed, feature_names=np.array(NAMES))
    return res


def collect(out_dir: str, n: int, seed0: int = 10_000, noise: float = 0.02) -> dict:
    env = GlassEnv(GlassConfig())
    out = []
    try:
        for i in range(n):
            out.append(run_episode(env, scripted("load_expert"), GlassScenario.sample(seed0 + i), out_dir,
                                   f"ep_{i:04d}", noise=noise, label="SCRIPTED (SYNTHETIC glove frames)"))
    finally:
        env.close()
    summ = {"episodes": n, "seed0": seed0, "noise_nm": noise, "demonstrator": "load_expert",
            "terminations": {t: sum(r["termination"] == t for r in out) for t in sorted({r["termination"] for r in out})},
            "data": "SYNTHETIC glove frames generated from simulator contact (dermis.glass.sensors)"}
    Path(out_dir, "collection.json").write_text(json.dumps({"summary": summ, "episodes": out}, indent=1))
    return summ
