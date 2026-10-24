"""REPLAY: re-run a recorded episode's issued actions open-loop from its saved scenario, for
inspection. This is playback, not control: nothing is predicted. AUTONOMOUS rollouts (a policy
choosing actions from observations) live in autonomous_gui() and evaluate.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data.episodes import load_episode
from .sim.env import ACTION_NAMES, OBS_NAMES, GraspLiftPlaceEnv, Scenario, TaskConfig


def replay(path: str, gui: bool = False, plot: str | None = None) -> int:
    ep = load_episode(Path(path).with_suffix(""))
    m = ep.meta
    print(f"[REPLAY] {m['id']} source={m['label']} seed={m['seed']} ticks={len(ep)} recorded outcome={m['termination']}")
    env = GraspLiftPlaceEnv(TaskConfig.from_dict(m["task_config"]), gui=gui)
    env.set_label(f"REPLAY of {m['label']} episode (open-loop actions)")
    obs = env.reset(Scenario.from_dict(m["scenario"]))
    max_dev = float(np.max(np.abs(obs - ep.obs[0])))
    import time
    for t, a in enumerate(ep.actions):
        t0 = time.monotonic()
        max_dev = max(max_dev, float(np.max(np.abs(obs[:14] - ep.obs[t][:14]))))
        obs, done, _ = env.step(a)
        if gui:
            time.sleep(max(0.0, env.cfg.dt - (time.monotonic() - t0)))
        if done:
            break
    print(f"[REPLAY] replayed outcome={env.stats.termination}; max |obs - recorded obs| over "
          f"pose/velocity features = {max_dev:.2e} (0 means the simulator reproduced the recording)")
    env.close()
    if plot:
        plot_episode(ep, plot)
        print(f"[REPLAY] wrote {plot}")
    return 0


def plot_episode(ep, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = ep.arrays["t"]
    o, a = ep.obs, ep.actions
    idx = {n: i for i, n in enumerate(OBS_NAMES)}
    fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for n in ("obj_rel_x", "obj_rel_y", "obj_rel_z"):
        ax[0].plot(t, o[:, idx[n]], label=n)
    ax[0].set_ylabel("object - gripper (m)")
    for i, n in enumerate(ACTION_NAMES[:3]):
        ax[1].plot(t, a[:, i], label=f"issued {n}")
    ax[1].set_ylabel("velocity cmd (m/s)")
    ax[2].plot(t, a[:, 3], label="issued opening")
    ax[2].plot(t, o[:, idx["opening"]], label="measured opening", ls="--")
    ax[2].set_ylabel("opening (m)")
    ax[3].plot(t, ep.arrays["force_l"], label="left finger normal force")
    ax[3].plot(t, ep.arrays["force_r"], label="right finger normal force")
    ax[3].plot(t, a[:, 4], label="issued effort limit", ls=":")
    thr = ep.meta["task_config"].get("excess_force_n")
    if thr:
        ax[3].axhline(thr, color="r", lw=0.8, label="benchmark threshold (simulated)")
    ax[3].set_ylabel("N (simulated)")
    ax[3].set_xlabel("time (s)")
    for x in ax:
        x.legend(fontsize=7, loc="upper right")
        x.grid(alpha=0.3)
    fig.suptitle(f"REPLAY inspection: {ep.id} [{ep.meta['label']}] -> {ep.meta['termination']}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def autonomous_gui(checkpoint: str, seed: int, episodes: int, gui: bool = True) -> int:
    from .bc.model import BCPolicy
    from .sim.rollout import run_episode
    policy = BCPolicy.load(checkpoint)
    env = GraspLiftPlaceEnv(gui=gui)
    env.set_label(f"AUTONOMOUS: {policy.meta['name']}")
    try:
        for k in range(episodes):
            policy.reset()
            outcome, _ = run_episode(env, seed + k, lambda obs, _e: (policy.act(obs), None), realtime=gui)
            print(f"[AUTONOMOUS] seed {seed + k}: {outcome['termination']} drops={outcome['drops']} "
                  f"peak_force={outcome['peak_force_n']:.1f}N time={outcome['ticks'] * env.cfg.dt:.1f}s")
    finally:
        env.close()
    return 0
