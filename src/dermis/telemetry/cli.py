"""`stream` and `calibrate` commands. Both use the same source -> pipeline path as the dashboard and demos."""
from __future__ import annotations

import json
import time

import numpy as np

from .calibration import BaselineCapture, Calibration, mounts_from_device_info
from .pipeline import TelemetryPipeline
from .sources import WireFileSource, open_source


def _open(a, realtime=True):
    src = open_source(a.source, path=a.path, rate=a.rate, loop=a.loop, host=a.host, port=a.port,
                      serial_port=a.serial_port, wire=a.wire, faults=a.faults, realtime=realtime)
    if a.source == "tcp":
        print(f"[GLOVE] listening for decoded-frame JSONL on {a.host}:{src.port}", flush=True)
    return src, TelemetryPipeline(pod_order=[p for p in a.pods.split(",") if p])


def _status_line(snap, label: str, cal: Calibration | None) -> str:
    parts = []
    dev = cal.apply(snap) if cal else None
    for i, p in enumerate(snap.pods):
        if not snap.present[i]:
            parts.append(f"{p}: MISSING")
            continue
        m = "fresh" if snap.mag_fresh[i] else ("STALE" if snap.mag_stale[i] else
                                                ("held" if snap.mag_held_mask[i] else "INVALID"))
        s = f"{p}: mag={m}"
        if dev is not None and dev["mask"][i]:
            s += f" dB={dev['delta_norm_ut'][i]:.0f}uT def={dev['deformation'][i]:.2f}"
        elif np.isfinite(snap.mag_raw[i]).all():
            s += f" |B|={np.linalg.norm(snap.mag_raw[i]):.0f}uT"
        s += f" imu={'fresh' if snap.imu_fresh[i] else ('STALE' if snap.imu_stale[i] else 'INVALID')}"
        parts.append(s)
    f = snap.frame
    return f"[{label}] frame={f.frame} cfg={f.config_id} lost={snap.lost_before} " + " | ".join(parts)


def stream(a) -> int:
    src, pipe = _open(a)
    label = src.label
    cal = Calibration.load(a.calibration) if a.calibration else None
    rec = open(a.record, "a") if a.record else None
    n = 0
    t0 = time.monotonic()
    try:
        for msg in src:
            if rec and msg.obj is not None:
                rec.write(json.dumps(msg.obj) + "\n")
            snap = pipe.ingest(msg, label)
            if snap is not None:
                n += 1
                for e in snap.events:
                    print(f"[{label}] EVENT {e}", flush=True)
                if a.json:
                    print(json.dumps(snap.summary()), flush=True)
                elif n % a.print_every == 1 or a.print_every == 1:
                    print(_status_line(snap, label, cal), flush=True)
            if a.max_frames and pipe.counters["messages"] >= a.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if rec:
            rec.close()
    stats = pipe.stats()
    if isinstance(src, WireFileSource):
        stats["wire_frames_held_for_config"] = src.held_frames
        stats["wire_stream_errors"] = getattr(src, "stream_errors", None)
    print(f"[{label}] done in {time.monotonic() - t0:.1f}s: " + json.dumps(stats), flush=True)
    return 0


def calibrate(a) -> int:
    print("Keep the glove UNLOADED (nothing touching the pads) during capture.")
    src, pipe = _open(a)
    cap = BaselineCapture(min_samples=min(50, a.frames // 2))
    n = 0
    for msg in src:
        snap = pipe.ingest(msg, src.label)
        if snap is not None:
            cap.add(snap)
            n += 1
        if n >= a.frames:
            break
    mounts = mounts_from_device_info(src.device_info) if getattr(src, "device_info", None) else None
    cal, report = cap.finish(a.scale_ut, mounts)
    cal.save(a.out)
    print(json.dumps({"calibration_id": cal.calibration_id, "device_id": cal.device_id, "config_id": cal.config_id,
                      "source": src.label, "frames": n, "pods": report, "mount_transforms": bool(mounts)}, indent=1))
    if src.label == "SYNTHETIC":
        print("NOTE: SYNTHETIC calibration, for software testing only.")
    print(f"wrote {a.out}")
    return 0
