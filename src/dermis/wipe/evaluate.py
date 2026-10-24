"""wipe_v1 evaluation: suites, equal-budget tuning on the dev suite, frozen evaluation, stress tests,
paired comparisons, plots. Separate from the pick-and-place and fragile_cup benchmarks."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path

import numpy as np

from .controllers import ObsFilter, make
from .env import WipeConfig, WipeEnv, WipeScenario

STRESS = {
    "nominal": {},
    "obs_noise": {"noise": {"pose_m": 0.0005, "geometry_m": 0.001, "load_n": 0.3}},   # stress test, labelled
    "delay_2ticks": {"delay_ticks": 2},                                                # stress test, labelled
}

# Equal tuning budget: 8 configurations per controller, chosen before seeing dev results.
GRIDS = {
    "A": [{"kind": "A", "k_z": k, "target_n": t} for k in (300.0, 600.0, 1500.0, 3000.0) for t in (3.5, 4.5)],
    "B": [{"kind": "B", "k_z": k, "target_n": t} for k in (1500.0, 2000.0, 3000.0, 4000.0) for t in (3.5, 4.5)],
    "C": [{"kind": "C", "k_z": k, "target_n": t} for k in (150.0, 300.0, 600.0, 1000.0) for t in (3.5, 4.5)],
    "Cp": [{"kind": "Cp", "k_z": k, "kp": kp, "ki": ki} for k in (600.0, 1500.0)
           for kp, ki in ((0.0, 0.002), (0.0, 0.005), (0.0003, 0.002), (0.0003, 0.005))],
    "D": [{"kind": "D", "k_z": k, "kp": kp, "ki": ki} for k in (600.0, 1500.0)
          for kp, ki in ((0.0, 0.002), (0.0, 0.005), (0.0003, 0.002), (0.0003, 0.005))],
}
SELECTION_RULE = "max dev successes; ties -> max mean dev coverage; ties -> min mean overload time"
NAMES = {"A": "A constant height", "B": "B geometry, stiff", "C": "C geometry, compliant",
         "Cp": "Cp proprioceptive load", "D": "D contact feedback"}


def dev_suite() -> list[WipeScenario]:
    return [WipeScenario.flat(1_000_000)] + [WipeScenario.sample(1_000_000 + i) for i in range(1, 16)]


def make_eval_suite(path: str, count: int = 40) -> dict:
    scen = [WipeScenario.flat(2_000_000)] + [WipeScenario.sample(2_000_000 + i) for i in range(1, count + 1)]
    suite = {"name": "wipe_eval_v1", "task": "wipe_v1", "frozen_unix": time.time(),
             "note": "Frozen before the final comparison. Do not tune on it.",
             "scenarios": [s.to_dict() for s in scen]}
    body = json.dumps(suite["scenarios"], sort_keys=True).encode()
    suite["sha256"] = hashlib.sha256(body).hexdigest()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(suite, indent=1))
    return suite


def load_eval_suite(path: str) -> list[WipeScenario]:
    s = json.loads(Path(path).read_text())
    body = json.dumps(s["scenarios"], sort_keys=True).encode()
    assert hashlib.sha256(body).hexdigest() == s["sha256"], "frozen evaluation suite was modified"
    return [WipeScenario.from_dict(d) for d in s["scenarios"]]


def run_episode(spec: dict, sc: WipeScenario, cfg: WipeConfig, stress: dict | None = None, env=None) -> dict:
    stress = stress or {}
    own = env is None
    env = env or WipeEnv(cfg)
    try:
        ctrl = make(spec)
        obs = env.reset(sc, k_z=ctrl.k_z, delay_ticks=stress.get("delay_ticks", 0))
        ctrl.reset(env.cfg)
        filt = ObsFilter(env, ctrl.SEES, stress.get("noise"), seed=sc.seed)
        while not env.done:
            obs, _, _ = env.step(ctrl.act(filt.view(obs)))
        return {"seed": sc.seed, **env.stats.to_dict(), "duration_s": env.t}
    finally:
        if own:
            env.close()


def _chunk(args):
    specs_scen, cfg_d, stress = args
    import torch  # noqa: F401  (keeps worker imports uniform)
    env = WipeEnv(WipeConfig.from_dict(cfg_d))
    out = []
    try:
        for spec, sd in specs_scen:
            out.append({"spec": spec, **run_episode(spec, WipeScenario.from_dict(sd), env.cfg, stress, env)})
    finally:
        env.close()
    return out


def run_many(jobs: list[tuple[dict, WipeScenario]], cfg: WipeConfig, stress: dict, workers: int) -> list[dict]:
    items = [(s, sc.to_dict()) for s, sc in jobs]
    if workers <= 1:
        return _chunk((items, cfg.to_dict(), stress))
    chunks = [items[i::workers] for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_chunk, [(c, cfg.to_dict(), stress) for c in chunks if c])
    return [r for part in parts for r in part]


def summarize(rows: list[dict]) -> dict:
    cov = [r["coverage"] for r in rows]
    return {"n": len(rows), "successes": sum(r["success"] for r in rows),
            "coverage_mean": statistics.mean(cov), "coverage_min": min(cov),
            "attempted_coverage_mean": statistics.mean(r["attempted_coverage"] for r in rows),
            "contact_loss_s_mean": statistics.mean(r["contact_loss_s"] for r in rows),
            "overload_s_mean": statistics.mean(r["overload_s"] for r in rows),
            "overload_episodes": sum(r["overload_s"] > 0 for r in rows),
            "peak_load_n_max": max(r["peak_load_n"] for r in rows),
            "p95_load_n_median": statistics.median(r["p95_load_n"] for r in rows),
            "penetration_episodes": sum(r["penetration_s"] > 0 for r in rows),
            "terminations": {t: sum(r["termination"] == t for r in rows) for t in {r["termination"] for r in rows}},
            "wipe_time_s": statistics.mean(r["wipe_time_s"] for r in rows),
            "approach_time_s": statistics.mean(r["approach_time_s"] for r in rows),
            "duration_s": statistics.mean(r["duration_s"] for r in rows)}


def tune(out_dir: str, workers: int, log=print) -> dict:
    cfg = WipeConfig()
    scen = dev_suite()
    jobs = [(spec, sc) for grid in GRIDS.values() for spec in grid for sc in scen]
    t0 = time.time()
    rows = run_many(jobs, cfg, {}, workers)
    chosen, table = {}, {}
    for key, grid in GRIDS.items():
        cands = []
        for spec in grid:
            rr = [r for r in rows if r["spec"] == spec]
            s = summarize(rr)
            cands.append((s["successes"], s["coverage_mean"], -s["overload_s_mean"], spec, s))
        cands.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)
        chosen[key] = cands[0][3]
        table[key] = [{"spec": c[3], "dev_successes": c[0], "dev_coverage_mean": round(c[1], 4),
                       "dev_overload_s_mean": round(-c[2], 4)} for c in cands]
        log(f"[tune] {NAMES[key]}: chose {cands[0][3]} -> dev {cands[0][0]}/{len(scen)} successes, "
            f"coverage {cands[0][1]:.3f}")
    res = {"selection_rule": SELECTION_RULE, "dev_suite_seeds": [s.seed for s in scen], "budget_per_controller": 8,
           "chosen": chosen, "candidates": table, "wall_s": round(time.time() - t0, 1)}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "tuning.json").write_text(json.dumps(res, indent=1))
    return res


def paired(rows: dict, a: str, b: str) -> dict:
    """Paired per-scenario differences (a - b) on identical scenarios. Descriptive only."""
    ra = {r["seed"]: r for r in rows[a]}
    rb = {r["seed"]: r for r in rows[b]}
    seeds = sorted(set(ra) & set(rb))
    d = np.array([ra[s]["coverage"] - rb[s]["coverage"] for s in seeds])
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return {"pair": f"{a} - {b}", "n": len(seeds), "coverage_diff_mean": float(d.mean()),
            "coverage_diff_min": float(d.min()), "coverage_diff_max": float(d.max()),
            "bootstrap_95_interval_of_mean": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "a_better": int((d > 1e-9).sum()), "b_better": int((d < -1e-9).sum()), "equal": int((abs(d) <= 1e-9).sum()),
            "success_a_only": sum(ra[s]["success"] and not rb[s]["success"] for s in seeds),
            "success_b_only": sum(rb[s]["success"] and not ra[s]["success"] for s in seeds)}


def evaluate(tuning_path: str, suite_path: str, out_dir: str, workers: int, log=print) -> dict:
    chosen = json.loads(Path(tuning_path).read_text())["chosen"]
    scen = load_eval_suite(suite_path)
    cfg = WipeConfig()
    out = {"suite": suite_path, "chosen": chosen, "conditions": {}}
    for cond, stress in STRESS.items():
        jobs = [(spec, sc) for spec in chosen.values() for sc in scen]
        rows = run_many(jobs, cfg, stress, workers)
        by = {k: sorted([r for r in rows if r["spec"] == spec], key=lambda r: r["seed"]) for k, spec in chosen.items()}
        out["conditions"][cond] = {
            "stress": stress, "label": "NOMINAL" if cond == "nominal" else "STRESS TEST",
            "summary": {k: summarize(v) for k, v in by.items()},
            "paired_vs_D": [paired(by, "D", k) for k in by if k != "D"],
            "episodes": {k: v for k, v in by.items()}}
        for k, s in out["conditions"][cond]["summary"].items():
            log(f"[{cond}] {NAMES[k]:26s} success {s['successes']}/{s['n']}  coverage {s['coverage_mean']:.3f} "
                f"(min {s['coverage_min']:.2f})  loss {s['contact_loss_s_mean']:.2f}s  overload {s['overload_s_mean']:.3f}s "
                f"p95 {s['p95_load_n_median']:.2f}N  peak {s['peak_load_n_max']:.1f}N")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "evaluation.json").write_text(json.dumps(out, indent=1, default=float))
    plots(out, out_dir)
    report(out, out_dir)
    return out


def plots(res: dict, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = list(res["conditions"])
    keys = list(res["chosen"])
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.6), sharey=True)
    for ax, cond in zip(np.atleast_1d(axes), conds):
        eps = res["conditions"][cond]["episodes"]
        data = [[r["coverage"] for r in eps[k]] for k in keys]
        ax.boxplot(data, showfliers=False)
        for i, d in enumerate(data):
            ax.scatter(np.full(len(d), i + 1) + np.random.default_rng(i).uniform(-0.15, 0.15, len(d)), d, s=6, alpha=0.5)
        ax.set_xticks(range(1, len(keys) + 1), keys)
        ax.axhline(0.9, color="r", lw=0.7, ls="--")
        ax.set_title(f"{cond}{' (stress)' if cond != 'nominal' else ''}", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("in-band coverage (fraction of 96 cells)")
    fig.suptitle("wipe_v1 frozen eval suite: per-scenario coverage (synthetic benchmark)", fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(out_dir, "coverage.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    nom = res["conditions"]["nominal"]["episodes"]
    for k in keys:
        if k == "D":
            continue
        d = sorted(a["coverage"] - b["coverage"] for a, b in zip(nom["D"], nom[k]))
        ax.plot(d, marker=".", label=f"D - {k}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("scenario (sorted by difference)")
    ax.set_ylabel("paired coverage difference")
    ax.set_title("Paired differences on identical scenarios (nominal)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out_dir, "paired_nominal.png"), dpi=120)
    plt.close(fig)


def report(res: dict, out_dir: str) -> None:
    lines = ["# wipe_v1 frozen evaluation (synthetic manipulation benchmark)", "",
             f"Suite: `{res['suite']}`. Controllers tuned on the dev suite only ({SELECTION_RULE}).", ""]
    for cond, c in res["conditions"].items():
        lines += [f"## {cond} ({c['label']})", "",
                  "| Controller | Success | Coverage mean (min) | Attempted | Contact loss s | Overload s (episodes) | p95 N (median) | Peak N (max) | Penetration eps |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for k, s in c["summary"].items():
            lines.append(f"| {NAMES[k]} | {s['successes']}/{s['n']} | {s['coverage_mean']:.3f} ({s['coverage_min']:.2f}) | "
                         f"{s['attempted_coverage_mean']:.3f} | {s['contact_loss_s_mean']:.2f} | "
                         f"{s['overload_s_mean']:.3f} ({s['overload_episodes']}) | {s['p95_load_n_median']:.2f} | "
                         f"{s['peak_load_n_max']:.1f} | {s['penetration_episodes']} |")
        lines += ["", "| Paired (identical scenarios) | mean coverage diff | min / max | 95% bootstrap interval of mean (descriptive) | D better / other better / equal | success only D / only other |",
                  "|---|---|---|---|---|---|"]
        for p_ in c["paired_vs_D"]:
            lo, hi = p_["bootstrap_95_interval_of_mean"]
            lines.append(f"| {p_['pair']} | {p_['coverage_diff_mean']:+.3f} | {p_['coverage_diff_min']:+.2f} / {p_['coverage_diff_max']:+.2f} | "
                         f"[{lo:+.3f}, {hi:+.3f}] | {p_['a_better']} / {p_['b_better']} / {p_['equal']} | "
                         f"{p_['success_a_only']} / {p_['success_b_only']} |")
        lines.append("")
    Path(out_dir, "report.md").write_text("\n".join(lines))
