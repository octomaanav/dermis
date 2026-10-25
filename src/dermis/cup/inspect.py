"""Episode inspection for fragile_cup datasets: observation/action alignment and contact plausibility.

    python -m dermis cup inspect-episodes --data runs/cup/benchmark_hybrid/episodes --n 6

Checks (per episode, reported as numbers, plus a pass/fail summary):
    first_obs_effort      obs[0].effort_prev == setup effort (obs[0] is read before any action)
    alignment             obs[t+1].effort_prev == actions[t] (the issued, clipped effort reached the env)
    label_vs_issued       |expert_actions - actions| (execution noise on SCRIPTED demos; labels stay clean)
    log_vs_next_obs       logged contact after step t == contact in obs[t+1] (same measurement)
    force_balance         during 'pause' segments with the cup quasi-static: shear_z_l + shear_z_r == m g + load
    outcome_consistency   final logged slip vs the recorded termination
"""
from __future__ import annotations

import json

import numpy as np

from ..data.episodes import EpisodeStore
from .env import OBS_NAMES, SETUP_EFFORT_N

I = {n: k for k, n in enumerate(OBS_NAMES)}


def inspect_episode(ep) -> dict:
    o, a, A = ep.obs, ep.actions[:, 0], ep.arrays
    m = ep.meta
    r = {"id": m["id"], "source": m["source"], "controller": m.get("controller"), "ticks": len(ep),
         "termination": m["termination"]}
    r["first_obs_effort_n"] = float(o[0, I["effort_prev"]])
    r["first_obs_ok"] = abs(r["first_obs_effort_n"] - SETUP_EFFORT_N) < 1e-4
    r["alignment_max_err_n"] = float(np.abs(o[1:, I["effort_prev"]] - a[:-1]).max()) if len(ep) > 1 else 0.0
    if "expert_actions" in A:
        d = A["expert_actions"][:, 0] - a
        r["label_minus_issued_n"] = {"mean": float(d.mean()), "std": float(d.std()), "max_abs": float(np.abs(d).max())}
    nxt = o[1:]
    r["log_vs_next_obs_max_err"] = float(max(np.abs(A["shear_z_l"][:-1] - nxt[:, I["shear_z_l"]]).max(),
                                             np.abs(A["normal_r"][:-1] - nxt[:, I["normal_r"]]).max())) if len(ep) > 1 else 0.0
    mass = m["scenario"]["mass"]
    phase = A["phase"]
    still = np.abs(nxt[:, I["obj_vz"]]) < 0.002 if len(ep) > 1 else np.array([], bool)
    sel = (phase[:-1] == "pause") & still if len(ep) > 1 else np.array([], bool)
    if sel.any():
        support = A["shear_z_l"][:-1][sel] + A["shear_z_r"][:-1][sel]
        need = mass * 9.81 + A["load_n"][:-1][sel]
        err = support - need
        r["force_balance"] = {"ticks": int(sel.sum()), "median_err_n": float(np.median(err)), "p95_abs_err_n": float(np.percentile(np.abs(err), 95))}
    slip = float(A["slip_m"][-1]) if len(A["slip_m"]) else 0.0
    r["final_slip_mm"] = round(slip * 1000, 2)
    term = m["termination"]
    r["outcome_consistent"] = ((term == "drop") == (slip > m["task_config"]["drop_slip_m"]) or term == "drop") and \
                              (term != "success" or slip <= m["task_config"]["success_slip_m"])
    return r


def inspect(data: str, n: int = 6, seed: int = 0) -> dict:
    store = EpisodeStore(data)
    ids = store.ids()
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(ids, size=min(n, len(ids)), replace=False).tolist())
    rows = [inspect_episode(store.load(i)) for i in pick]
    fb = [r["force_balance"]["p95_abs_err_n"] for r in rows if "force_balance" in r]
    summary = {"episodes": len(rows), "first_obs_ok": all(r["first_obs_ok"] for r in rows),
               "alignment_max_err_n": max(r["alignment_max_err_n"] for r in rows),
               "log_vs_next_obs_max_err": max(r["log_vs_next_obs_max_err"] for r in rows),
               "force_balance_p95_abs_err_n_worst": max(fb) if fb else None,
               "outcome_consistent": all(r["outcome_consistent"] for r in rows)}
    summary["pass"] = (summary["first_obs_ok"] and summary["alignment_max_err_n"] < 1e-4 and
                       summary["log_vs_next_obs_max_err"] < 1e-4 and (fb == [] or max(fb) < 0.15) and summary["outcome_consistent"])
    return {"data": data, "summary": summary, "episodes": rows}


if __name__ == "__main__":
    import sys
    print(json.dumps(inspect(sys.argv[1]), indent=1))
