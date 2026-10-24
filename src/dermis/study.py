"""Data-scaling x training-seed study: BC with vs. without simulated contact observations.

For every (demo count N, training seed s) both variants train on the *same* first-N episodes
with the same episode split, architecture and budget; only feature_groups differ. All runs
are evaluated on the same suite. Reports per-cell success and the mean/min/max over seeds.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from .bc.train import train
from .config import load_config
from .evaluate import evaluate


def run_study(cfg_path: str, workers: int, out: str | None = None) -> dict:
    cfg = load_config(cfg_path)
    root = Path(out or cfg["out"])
    rows = []
    for n in cfg["demo_counts"]:
        for seed in cfg["train_seeds"]:
            for pol in cfg["policies"]:
                name = f"{pol['name']}_n{n}_s{seed}"
                res_path = root / "results" / f"{name}.json"
                if not res_path.exists():
                    tc = load_config(pol["config"])
                    tc.update({"data": cfg["data"], "out": str(root / "runs" / name), "name": name,
                               "max_episodes": n, "seed": seed})
                    ckpt = train(tc, log=lambda *_: None)
                    evaluate(str(ckpt), cfg["suite"], str(res_path), workers=workers, log=lambda *_: None)
                s = json.loads(res_path.read_text())["summary"]
                rows.append({"policy": pol["name"], "demos": n, "seed": seed, "successes": s["successes"],
                             "attempts": s["attempts"], "drop_eps": s["episodes_with_drop"],
                             "excess_force_eps": s["episodes_with_excess_force"], "timeouts": s["timeouts"],
                             "never_lifted": s["never_lifted"]})
                print(f"{name}: {s['successes']}/{s['attempts']} drops(ep)={s['episodes_with_drop']} "
                      f"excess(ep)={s['episodes_with_excess_force']} terminations={s['terminations']}", flush=True)
    agg = []
    for n in cfg["demo_counts"]:
        for pol in cfg["policies"]:
            cell = [r for r in rows if r["demos"] == n and r["policy"] == pol["name"]]
            rates = [r["successes"] / r["attempts"] for r in cell]
            agg.append({"policy": pol["name"], "demos": n, "seeds": len(cell),
                        "success_rate_mean": statistics.mean(rates), "success_rate_min": min(rates),
                        "success_rate_max": max(rates), "per_seed": [f"{r['successes']}/{r['attempts']}" for r in cell],
                        "drop_eps_total": sum(r["drop_eps"] for r in cell),
                        "excess_force_eps_total": sum(r["excess_force_eps"] for r in cell)})
    lines = [f"# {cfg.get('title', 'Study')}", "", f"Suite: `{cfg['suite']}`. Simulated contact observations only; "
             "this says nothing about the value of real glove data.", "",
             "| Demos (train+val) | Policy | Success per training seed | Mean | Min–max | Drop eps (all seeds) | Excess-force eps (all seeds) |",
             "|---|---|---|---|---|---|---|"]
    for a in agg:
        lines.append(f"| {a['demos']} | {a['policy']} | {', '.join(a['per_seed'])} | {a['success_rate_mean']:.2f} | "
                     f"{a['success_rate_min']:.2f}–{a['success_rate_max']:.2f} | {a['drop_eps_total']} | {a['excess_force_eps_total']} |")
    (root / "report").mkdir(parents=True, exist_ok=True)
    (root / "report" / "study.md").write_text("\n".join(lines) + "\n")
    (root / "report" / "study.json").write_text(json.dumps({"rows": rows, "aggregate": agg, "config": cfg}, indent=1))
    _chart(agg, cfg, root / "report" / "study.png")
    print("\n".join(lines))
    return {"rows": rows, "aggregate": agg}


def _chart(agg, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, pol in enumerate(cfg["policies"]):
        cell = [a for a in agg if a["policy"] == pol["name"]]
        x = [a["demos"] for a in cell]
        m = [a["success_rate_mean"] for a in cell]
        lo = [a["success_rate_mean"] - a["success_rate_min"] for a in cell]
        hi = [a["success_rate_max"] - a["success_rate_mean"] for a in cell]
        ax.errorbar(x, m, yerr=[lo, hi], marker="o", capsize=3, label=pol["name"], color=["#4a72b8", "#d8783a"][k % 2])
    ax.set_xscale("log")
    ax.set_xticks(cfg["demo_counts"], [str(n) for n in cfg["demo_counts"]])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("scripted demonstrations (train + val)")
    ax.set_ylabel("held-out success rate")
    ax.set_title("BC success vs. data (mean, min–max over training seeds; simulation)", fontsize=9)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
