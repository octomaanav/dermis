"""Inspection report for a recorded glove session (python -m dermis glove report --session DIR).

Per discovered pod with a magnetometer: |dB| relative to the UNLOADED reference (or, if none was
captured, the mean of the 'unloaded' trials), raw and exponentially filtered, per labelled trial and
repeat. Numbers: stationary variation, press plateaus per repeat, release recovery, sideways-load
response direction (NOT slip), timing (device and host clocks reported separately, not synchronised).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..telemetry.pipeline import RawMessage, TelemetryPipeline


def _load(session: Path):
    layout = json.loads((session / "layout.json").read_text())
    meta = json.loads((session / "meta.json").read_text()) if (session / "meta.json").exists() else {}
    markers = [json.loads(l) for l in (session / "markers.jsonl").read_text().splitlines() if l.strip()]
    pipe = TelemetryPipeline(pod_order=layout["pods"])
    rows = []
    for line in (session / "frames.jsonl").read_text().splitlines():
        obj = json.loads(line)
        t = obj.pop("_t_recv_ns", None)
        s = pipe.ingest(RawMessage(obj, t or 0), "REPLAY")
        if s is not None:
            rows.append((t, s))
    return layout, meta, markers, rows, pipe.stats()


def _ema(x, a=0.2):
    out, m = np.empty_like(x), np.nan
    for i, v in enumerate(x):
        m = v if not np.isfinite(m) else (m if not np.isfinite(v) else m + a * (v - m))
        out[i] = m
    return out


def report(session: str, out: str | None = None) -> dict:
    session = Path(session)
    out = Path(out or session / "report")
    out.mkdir(parents=True, exist_ok=True)
    layout, meta, markers, rows, stats = _load(session)
    t = np.array([r[0] for r in rows], float)
    pods = [p for p in layout["pods"] if layout["channels"][p]["mag"]]
    trials = [m for m in markers if m.get("kind") == "trial"]
    refs = {m["label"]: m for m in markers if m.get("kind") == "reference"}
    res = {"source_label": meta.get("source_label"), "status": "PENDING HARDWARE" if meta.get("source_label") in
           ("SYNTHETIC", None) else "RECORDED - judge against the checklist", "pods": {},
           "units": "microtesla change from the unloaded reference; not force", "pipeline": stats}
    series = {}
    for p in pods:
        i = rows[0][1].pods.index(p) if rows else 0
        B = np.array([s.mag_raw[i] for _, s in rows])
        ref = None
        u = refs.get("reference_unloaded", {}).get("stats", {}).get(p, {})
        if u and u.get("mean") is not None:
            ref = np.array(u["mean"])
        else:
            sel = np.zeros(len(t), bool)
            for m in trials:
                if m["label"] == "unloaded":
                    sel |= (t >= m["start_ns"]) & (t <= m["end_ns"])
            if sel.any():
                ref = np.nanmean(B[sel], 0)
        if ref is None:
            res["pods"][p] = {"note": "no unloaded reference or unloaded trial: dB not computed"}
            continue
        dB = B - ref
        mag = np.linalg.norm(dB, axis=1)
        series[p] = (dB, mag, _ema(mag))
        r = {"reference": "unloaded reference capture" if u else "mean of unloaded trials"}
        per = {}
        for m in trials:
            sel = (t >= m["start_ns"]) & (t <= m["end_ns"])
            x = mag[sel]
            x = x[np.isfinite(x)]
            if len(x) < 4:
                continue
            mid = x[len(x) // 4: 3 * len(x) // 4]
            per.setdefault(m["label"], []).append({"median_ut": float(np.median(mid)), "std_ut": float(np.std(mid)),
                                                   "last20pct_ut": float(np.mean(x[-max(2, len(x) // 5):])),
                                                   "peak_ut": float(x.max()),
                                                   "dB_mean": np.nanmean(dB[sel][len(x) // 4: 3 * len(x) // 4], 0).tolist()})
        r["trials"] = per
        if "unloaded" in per:
            r["stationary_std_ut"] = float(np.mean([v["std_ut"] for v in per["unloaded"]]))
        for lab in ("light_press", "stronger_press"):
            if lab in per:
                med = [v["median_ut"] for v in per[lab]]
                r[f"{lab}_repeat_cv"] = float(np.std(med) / max(1e-9, np.mean(med)))
        if "release" in per:
            r["release_residual_ut"] = [v["last20pct_ut"] for v in per["release"]]
        if "sideways_load" in per and "stronger_press" in per:
            a = np.mean([v["dB_mean"] for v in per["sideways_load"]], 0)
            b = np.mean([v["dB_mean"] for v in per["stronger_press"]], 0)
            r["sideways_vs_press_direction_deg"] = float(np.degrees(np.arccos(np.clip(
                a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1))))
            r["sideways_note"] = "direction difference = response to tangential load; this is NOT slip detection"
        res["pods"][p] = r
    if rows:
        dev = np.diff([s.frame.t_device_us for _, s in rows]) / 1e3
        host = np.diff(t) / 1e6
        res["timing"] = {"device_interval_ms_median": float(np.median(dev)), "host_interval_ms_median": float(np.median(host)),
                         "host_interval_ms_p95": float(np.percentile(host, 95)),
                         "note": "device and host clocks are reported separately; they are not synchronised"}
    (out / "numbers.json").write_text(json.dumps(res, indent=1, default=float))
    _plots(series, rows, t, trials, meta, out)
    return res


def _plots(series, rows, t, trials, meta, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not rows or not series:
        return
    tag = f" [{meta.get('source_label', '?')}]"
    ts = (t - t[0]) / 1e9
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 2.4 * len(series)), sharex=True, squeeze=False)
    for ax, (p, (dB, mag, filt)) in zip(axes[:, 0], series.items()):
        ax.plot(ts, mag, lw=0.6, alpha=0.6, label="|dB| raw")
        ax.plot(ts, filt, lw=1.2, label="|dB| filtered (EMA 0.2)")
        for m in trials:
            a0, a1 = (m["start_ns"] - t[0]) / 1e9, (m["end_ns"] - t[0]) / 1e9
            ax.axvspan(a0, a1, alpha=0.08, color="gray")
            ax.text(a0, ax.get_ylim()[1] * 0.95 if np.isfinite(ax.get_ylim()[1]) else 0, m["label"], fontsize=6, va="top")
        ax.set_ylabel(f"{p}\nuT")
        ax.legend(fontsize=6, loc="upper right")
    axes[-1, 0].set_xlabel("host time (s)")
    fig.suptitle("Magnetic change from unloaded reference, raw vs filtered (not force)" + tag, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "overlay.png", dpi=110)
    plt.close(fig)
    labels = sorted({m["label"] for m in trials})
    if not labels:
        return
    p0 = next(iter(series))
    _, mag, _ = series[p0]
    fig, axes = plt.subplots(len(labels), 1, figsize=(7, 1.8 * len(labels)), sharex=True, squeeze=False)
    for ax, lab in zip(axes[:, 0], labels):
        for k, m in enumerate([m for m in trials if m["label"] == lab]):
            sel = (t >= m["start_ns"]) & (t <= m["end_ns"])
            ax.plot((t[sel] - m["start_ns"]) / 1e9, mag[sel], lw=0.9, label=f"repeat {k + 1}")
        ax.set_title(lab, fontsize=8)
        ax.legend(fontsize=6)
    axes[-1, 0].set_xlabel("s since trial start")
    fig.suptitle(f"Repeats per label, {p0}" + tag, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "repeats.png", dpi=110)
    plt.close(fig)
