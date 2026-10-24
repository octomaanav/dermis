"""Comparison table (Markdown + JSON) and bar chart from evaluation result files."""
from __future__ import annotations

import json
from pathlib import Path


def _fmt(v, nd=2):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def compare(result_paths: list[str], out_dir: str, title: str = "Evaluation comparison") -> dict:
    results = [json.loads(Path(p).read_text()) for p in result_paths]
    rows = []
    for r in results:
        s, pol = r["summary"], r["policy"]
        name = pol.get("name") or pol["kind"]
        demos = pol.get("demonstrations")
        rows.append({
            "policy": name, "label": pol["label"], "suite": r["suite"],
            "success": f"{s['successes']}/{s['attempts']}", "success_rate": s["success_rate"],
            "episodes_with_drop": s["episodes_with_drop"], "episodes_with_excess_force": s["episodes_with_excess_force"],
            "peak_force_n_max": s["peak_contact_force_n_max"], "peak_force_n_median": s["peak_contact_force_n_median"],
            "completion_s_median": s["completion_time_s_median"], "timeouts": s["timeouts"],
            "other_failures": s["attempts"] - s["successes"] - s["timeouts"],
            "never_lifted": s["never_lifted"],
            "demos": f"{demos['train_episodes']}+{demos['val_episodes']} ({', '.join(demos['sources'])})" if demos else "–",
            "features": "+".join(pol.get("feature_groups", [])) or "privileged state",
        })
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = [("policy", "Policy"), ("label", "Mode"), ("success", "Success"), ("episodes_with_drop", "Drop eps"),
            ("episodes_with_excess_force", "Excess-force eps"), ("peak_force_n_max", "Peak force N (max)"),
            ("peak_force_n_median", "Peak force N (median)"), ("completion_s_median", "Completion s (median)"),
            ("timeouts", "Timeouts"), ("other_failures", "Other failures"), ("demos", "Demos (train+val)"),
            ("features", "Observation groups")]
    lines = [f"# {title}", "", f"Suite: `{results[0]['suite']}`. " + results[0]["experiment_note"], "",
             "| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row[k], 1) for k, _ in cols) + " |")
    lines += ["", f"Excess-force threshold: {results[0]['summary']['excess_force_threshold_n']} N per finger "
              "(simulated benchmark threshold; not a biological or clinical measure).", ""]
    (out / "comparison.md").write_text("\n".join(lines))
    (out / "comparison.json").write_text(json.dumps({"title": title, "rows": rows, "sources": result_paths}, indent=1))
    _chart(rows, out / "comparison.png", title)
    return {"rows": rows, "markdown": "\n".join(lines)}


def _chart(rows: list[dict], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["policy"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    metrics = [("success_rate", "Success rate", 1.0), ("episodes_with_drop", "Episodes with a drop", None),
               ("episodes_with_excess_force", "Episodes over force threshold", None)]
    for ax, (k, label, ymax) in zip(axes, metrics):
        vals = [r[k] or 0 for r in rows]
        ax.bar(names, vals, color="#4a72b8")
        ax.set_title(label, fontsize=10)
        if ymax:
            ax.set_ylim(0, ymax)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}" if isinstance(v, float) else str(v), ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title + " (simulation)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
