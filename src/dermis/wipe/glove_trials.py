"""Glove pad-contact feasibility recording and inspection. RESULT STATUS: PENDING HARDWARE.

Physical setup (see docs/wipe_feasibility.md section 6): an instrumented fingertip presses a thin
pad (which must transmit load to the sensing region) against a stationary surface. Optional
reference: a kitchen/lab scale under the surface, logged as `host_unix_s,grams` lines (it measures
one reference load component only, not three-axis force). Otherwise record synchronized video.

Recording uses the existing ingestion path (telemetry.sources -> TelemetryPipeline). A session
directory holds:
    meta.json          pod, mount/pad notes, source label, device/config ids, clock anchors
    calibration.json   unloaded magnetic baseline captured at the start (telemetry.calibration)
    frames.jsonl       every decoded frame exactly as received + host receive time
    markers.jsonl      one line per trial: label, repeat, host start/end (monotonic ns + unix s)
Nothing here converts magnetic readings to newtons, labels shear as slip, or estimates position.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

TRIALS = [
    ("unloaded", "Hand still, pad touching nothing."),
    ("light_press", "Press the pad lightly on the surface and hold still."),
    ("strong_press", "Press firmly and hold still."),
    ("press_release", "Press, hold ~2 s, release fully, keep the hand still after release."),
    ("wipe_constant_contact", "Keep contact and wipe slowly back and forth with steady pressure."),
    ("lift_off", "Start pressing, then lift clear of the surface and hold still in the air."),
    ("sideways_no_slip", "Press and push sideways WITHOUT letting the pad slide."),
    ("observed_slip", "Press and push sideways until the pad visibly slides (optional; note what you saw)."),
]


class _Stamped:
    """Wraps a telemetry source so each recorded frame also carries its host receive time
    (`_t_recv_ns`, monotonic ns). The adapter ignores the extra key."""

    def __init__(self, src):
        self.src, self.label = src, getattr(src, "label", "UNKNOWN")
        self.device_info = None

    def __iter__(self):
        for m in self.src:
            if m.obj is not None:
                m.obj["_t_recv_ns"] = m.t_recv_ns
            self.device_info = getattr(self.src, "device_info", None)
            yield m


def record_session(a) -> int:
    from ..telemetry.calibration import BaselineCapture
    from ..telemetry.pipeline import TelemetryPipeline
    from ..telemetry.runner import TelemetryRunner
    from ..telemetry.sources import open_source
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = _Stamped(open_source(a.source, path=a.path, port=a.port, serial_port=a.serial_port, wire=a.wire, loop=True))
    runner = TelemetryRunner(src, TelemetryPipeline(), history=20000).start()
    label = runner.label
    if label == "SYNTHETIC":
        print("NOTE: SYNTHETIC source - software test only, not a glove measurement.")
    runner.start_recording(str(out / "frames.jsonl"))
    snap = runner.wait_for_frame(5.0)
    if snap is None:
        print("no telemetry received; check the source")
        return 2
    meta = {"status": "PENDING HARDWARE" if label == "SYNTHETIC" else "RECORDED (unvalidated)",
            "source": label, "pod": a.pod, "mount_note": a.mount_note, "device_id": snap.frame.device_id,
            "config_id": snap.frame.config_id, "repeats": a.repeats, "trial_s": a.trial_s,
            "clock_anchor": {"host_unix_s": time.time(), "host_monotonic_ns": time.monotonic_ns()},
            "scale_csv": a.scale_csv, "device_info": getattr(src, "device_info", None)}
    ask = (lambda msg: None) if a.auto else (lambda msg: input(msg))
    ask("Baseline: keep the pad UNLOADED and the hand still, then press Enter...")
    cap = BaselineCapture(min_samples=min(50, a.baseline_frames // 2))
    seen = 0
    t0 = time.monotonic()
    last = -1
    while seen < a.baseline_frames and time.monotonic() - t0 < 30:
        for seq, s in runner.since(last):
            last = seq
            cap.add(s)
            seen += 1
        time.sleep(0.02)
    cal, rep = cap.finish(scale_ut=500.0)
    cal.save(out / "calibration.json")
    meta["calibration_id"] = cal.calibration_id
    meta["baseline_accepted_for_pod"] = a.pod in cal.pods
    if a.pod not in cal.pods:
        print(f"WARNING: baseline for {a.pod} was REJECTED ({rep.get(a.pod)}). The pad was probably loaded or "
              "moving. Redo the session; baseline-subtracted values for this pod will be unavailable.")
    meta["baseline_report"] = rep
    with open(out / "markers.jsonl", "a") as mk:
        for name, instr in TRIALS:
            for k in range(a.repeats):
                ans = ask(f"[{name} {k + 1}/{a.repeats}] {instr}  Enter = start, 's' = skip trial type: ")
                if ans and ans.strip().lower() == "s":
                    break
                start_ns, start_unix = time.monotonic_ns(), time.time()
                print(f"  recording {a.trial_s:.0f} s ...", flush=True)
                time.sleep(a.trial_s)
                note = "" if a.auto else (input("  note (what happened? slip seen?): ") or "")
                mk.write(json.dumps({"label": name, "repeat": k, "start_ns": start_ns, "end_ns": time.monotonic_ns(),
                                     "start_unix_s": start_unix, "end_unix_s": time.time(), "note": note}) + "\n")
                mk.flush()
    runner.stop_recording()
    meta["pipeline_stats"] = runner.pipeline.stats()
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"session saved to {out}. Next: python -m dermis wipe glove-report --session {out}")
    return 0


# -- analysis -----------------------------------------------------------------------------

def _load(session: Path):
    from ..telemetry.calibration import Calibration
    from ..telemetry.pipeline import RawMessage, TelemetryPipeline
    meta = json.loads((session / "meta.json").read_text())
    cal = Calibration.load(session / "calibration.json")
    markers = [json.loads(l) for l in (session / "markers.jsonl").read_text().splitlines() if l.strip()]
    pipe = TelemetryPipeline()
    pod = meta["pod"]
    rows = []
    for line in (session / "frames.jsonl").read_text().splitlines():
        obj = json.loads(line)
        t_ns = obj.pop("_t_recv_ns", None)
        snap = pipe.ingest(RawMessage(obj, t_ns or 0), meta["source"])
        if snap is None:
            continue
        i = snap.pods.index(pod)
        d = cal.apply(snap, use_held=False)
        rows.append({"t_dev": snap.frame.t_device_us, "t_recv": t_ns, "raw": snap.mag_raw[i].copy(),
                     "delta": d["delta_ut"][i].copy(), "valid": bool(snap.mag_valid[i]), "fresh": bool(snap.mag_fresh[i]),
                     "stale": bool(snap.mag_stale[i]), "acc": snap.acc_raw[i].copy(), "gyr": snap.gyr_raw[i].copy(),
                     "lost": snap.lost_before})
    return meta, cal, markers, rows, pipe.stats()


def report_session(session: str, out: str | None) -> int:
    session = Path(session)
    out = Path(out or session / "report")
    out.mkdir(parents=True, exist_ok=True)
    meta, cal, markers, rows, stats = _load(session)
    if meta["pod"] not in cal.pods:
        print(f"{meta['pod']} has no accepted unloaded baseline in this session: baseline-subtracted analysis is "
              "not possible. Re-record with the pad unloaded during the baseline step. Raw plots are still written.")
        plot({}, rows, meta, out, markers)
        (out / "numbers.json").write_text(json.dumps({"status": "NOT CALIBRATED", "pod": meta["pod"]}, indent=1))
        return 3
    if not rows or rows[0]["t_recv"] is None:
        print("frames.jsonl lacks host receive times (_t_recv_ns); re-record with this tool")
        return 2
    t = np.array([r["t_recv"] for r in rows], float)
    dB = np.array([r["delta"] for r in rows])
    mag = np.linalg.norm(dB, axis=1)
    trials = {}
    for m in markers:
        sel = (t >= m["start_ns"]) & (t <= m["end_ns"])
        tt = (t[sel] - m["start_ns"]) / 1e9
        trials.setdefault(m["label"], []).append({"t": tt, "mag": mag[sel], "dB": dB[sel], "repeat": m["repeat"],
                                                   "fresh_frac": float(np.mean([rows[i]["fresh"] for i in np.where(sel)[0]])) if sel.any() else 0.0,
                                                   "note": m.get("note", "")})
    numbers = analyse(trials)
    numbers["status"] = "PENDING HARDWARE" if meta["source"] == "SYNTHETIC" else "MEASURED - judge against the checklist"
    numbers["source"] = meta["source"]
    numbers["pipeline"] = stats
    numbers["units"] = "magnetic change |dB| in microtesla relative to the unloaded baseline; NOT force"
    if meta.get("scale_csv") and Path(meta["scale_csv"]).exists():
        numbers["scale_reference"] = scale_compare(meta, rows, mag, out)
    (out / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    plot(trials, rows, meta, out, markers)
    print(json.dumps({k: v for k, v in numbers.items() if k != "pipeline"}, indent=1, default=float))
    return 0


def _plateau(x):
    """Middle 50 % of a trial: excludes the press/release transitions."""
    n = len(x)
    return x[n // 4: 3 * n // 4] if n >= 8 else x


def analyse(trials: dict) -> dict:
    res = {}
    unl = [r["mag"] for r in trials.get("unloaded", [])]
    if unl:
        allu = np.concatenate(unl)
        res["noise_floor_ut_std"] = float(allu.std())
        res["unloaded_offset_ut_mean"] = float(allu.mean())
    for lab in ("light_press", "strong_press"):
        reps = [float(np.median(_plateau(r["mag"]))) for r in trials.get(lab, []) if len(r["mag"])]
        if reps:
            res[f"{lab}_plateau_ut_per_repeat"] = reps
            res[f"{lab}_repeat_cv"] = float(np.std(reps) / max(1e-9, np.mean(reps)))
            res[f"{lab}_within_trial_std_ut"] = float(np.mean([np.std(_plateau(r["mag"])) for r in trials[lab]]))
    if "light_press_plateau_ut_per_repeat" in res and "strong_press_plateau_ut_per_repeat" in res:
        a, b = np.array(res["light_press_plateau_ut_per_repeat"]), np.array(res["strong_press_plateau_ut_per_repeat"])
        pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-9
        res["strong_vs_light_separation"] = float((b.mean() - a.mean()) / pooled)
    rec = []
    for lab in ("press_release", "lift_off"):
        for r in trials.get(lab, []):
            m = r["mag"]
            if len(m) > 10:
                rec.append({"label": lab, "repeat": r["repeat"], "peak_ut": float(m.max()),
                            "residual_last_20pct_ut": float(np.mean(m[-max(2, len(m) // 5):]))})
    res["release_recovery"] = rec
    sw = [r["dB"] for r in trials.get("sideways_no_slip", [])]
    pr = [r["dB"] for r in trials.get("strong_press", [])]
    if sw and pr:
        def direction(blocks):
            v = np.concatenate([_plateau(b) for b in blocks]).mean(0)
            return v / (np.linalg.norm(v) + 1e-9)
        res["sideways_vs_press_dB_direction_angle_deg"] = float(np.degrees(np.arccos(np.clip(direction(sw) @ direction(pr), -1, 1))))
        res["sideways_note"] = "a different dB direction suggests the pad responds to tangential load; it does NOT detect slip"
    res["fresh_fraction_by_label"] = {k: float(np.mean([r["fresh_frac"] for r in v])) for k, v in trials.items()}
    res["notes"] = {k: [r["note"] for r in v if r["note"]] for k, v in trials.items()}
    return res


def scale_compare(meta, rows, mag, out: Path) -> dict:
    """Pair each scale reading with the nearest frame by host time. A scale measures ONE (vertical)
    load component; this does not validate three-axis force."""
    a = meta["clock_anchor"]
    lines = [l.split(",") for l in Path(meta["scale_csv"]).read_text().splitlines() if l.strip() and not l.startswith("#")]
    ts = np.array([float(x[0]) for x in lines])
    g = np.array([float(x[1]) for x in lines])
    t_unix = a["host_unix_s"] + (np.array([r["t_recv"] for r in rows], float) - a["host_monotonic_ns"]) / 1e9
    idx = np.clip(np.searchsorted(t_unix, ts), 0, len(t_unix) - 1)
    pairs = np.column_stack([g * 9.81e-3, mag[idx]])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.scatter(pairs[:, 0], pairs[:, 1], s=8)
    ax.set_xlabel("scale reading (N, vertical reference component only)")
    ax.set_ylabel("|dB| (uT)")
    ax.set_title("Reference comparison (NOT a calibration)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "scale_vs_dB.png", dpi=120)
    plt.close(fig)
    r = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]) if len(pairs) > 2 else None
    return {"pairs": len(pairs), "pearson_r": r, "note": "monotonic association only; no newton conversion"}


def plot(trials: dict, rows: list, meta: dict, out: Path, markers: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tag = " [SYNTHETIC - software test only]" if meta["source"] == "SYNTHETIC" else ""
    labels = [l for l, _ in TRIALS if l in trials]
    if labels:
        _plot_repeats(trials, labels, tag, out)
    _plot_overview(rows, meta, tag, out, markers)


def _plot_repeats(trials, labels, tag, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(labels), 1, figsize=(8, 1.9 * len(labels)), sharex=True, squeeze=False)
    for ax, lab in zip(axes[:, 0], labels):
        for r in trials[lab]:
            ax.plot(r["t"], r["mag"], lw=1, label=f"repeat {r['repeat'] + 1}")
        ax.set_ylabel("|dB| uT", fontsize=7)
        ax.set_title(lab, fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
    axes[-1, 0].set_xlabel("s since trial start")
    fig.suptitle("Repeats overlaid: magnetic change from unloaded baseline (not force)" + tag, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "repeats.png", dpi=110)
    plt.close(fig)



def _plot_overview(rows, meta, tag, out, markers):
    import matplotlib.pyplot as plt
    raw = np.array([r["raw"] for r in rows])
    dB = np.array([r["delta"] for r in rows])
    acc = np.array([r["acc"] for r in rows])
    t = (np.array([r["t_recv"] for r in rows], float) - rows[0]["t_recv"]) / 1e9
    flags = np.array([0 if r["fresh"] else (1 if r["stale"] else 2) for r in rows])
    fig, ax = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    for k, n in enumerate("xyz"):
        ax[0].plot(t, raw[:, k], lw=0.7, label=f"B{n}")
        ax[1].plot(t, dB[:, k], lw=0.7, label=f"dB{n}")
        ax[2].plot(t, acc[:, k], lw=0.7, label=f"a{n}")
    ax[3].plot(t, flags, lw=0.7)
    ax[3].set_yticks([0, 1, 2], ["fresh", "STALE", "invalid"])
    ax[0].set_ylabel("raw B (uT)")
    ax[1].set_ylabel("baseline-subtracted (uT)")
    ax[2].set_ylabel("acc (m/s2)")
    for a_ in ax[:3]:
        a_.legend(fontsize=6, loc="upper right")
    for m in markers:
        a0 = (m["start_ns"] - rows[0]["t_recv"]) / 1e9
        a1 = (m["end_ns"] - rows[0]["t_recv"]) / 1e9
        for a_ in ax:
            a_.axvspan(a0, a1, alpha=0.08, color="gray")
        ax[0].text(a0, ax[0].get_ylim()[1], m["label"], fontsize=5, rotation=90, va="top")
    ax[3].set_xlabel("host time (s)")
    fig.suptitle(f"Session overview, pod {meta['pod']}{tag}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=110)
    plt.close(fig)

