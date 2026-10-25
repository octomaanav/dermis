"""Build-time feature extraction: numbers from raw 350 Hz frames (needs numpy).

Runs when the snapshot is built (snapshot.py), never inside a hosted agent. Works on any
frames x columns array with dermis-sim-v1 column names; columns hardware can't measure
(ground-truth force, labels, object lift) are used when present and skipped when not.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .analysis import FAULT, GRIP
from .profiles import PODS

RAIL_FLAG = 0x08
# Field-to-force is an assumption until load-cell calibration (dataset README, FORCE_REF).
FIELD_PER_NEWTON_UT = 1500.0
# MLX90393 noise spec used by the sim, µT (x, y, z). Real noise above ~3x this is flagged.
NOISE_SPEC_UT = (5.0, 5.0, 9.0)


def _moving_avg(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) < n:
        return x.copy()
    k = np.ones(n) / n
    padded = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(padded, k, mode="valid")


def _noise(x: np.ndarray, mask: np.ndarray) -> float:
    """Robust white-noise level: MAD of first differences over the masked frames."""
    d = np.diff(x)[mask[1:] & mask[:-1]]
    return float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2)) if d.size else 0.0


def analyze_episode(meta: dict, columns: list[str], X: np.ndarray) -> dict:
    c = {name: i for i, name in enumerate(columns)}
    n = len(X)
    get = lambda name: X[:, c[name]].astype(np.float64) if name in c else None  # noqa: E731

    t_us = get("t_device_us")
    if t_us is not None and n > 1:
        dt = np.diff(t_us)
        rate = float(1e6 / np.median(dt[dt > 0])) if np.any(dt > 0) else 350.0
        t = (t_us - t_us[0]) / 1e6
        dropped = int(np.sum(dt > 1.5 * 1e6 / rate))
    else:
        rate, dropped = 350.0, 0
        t = np.arange(n) / rate

    contact_lbl = get("label_contact")
    # Rest field: frames labelled no-contact if we have labels, else the first 0.25 s.
    if contact_lbl is not None and np.any(contact_lbl == 0):
        rest_mask = contact_lbl == 0
    else:
        rest_mask = np.zeros(n, bool)
        rest_mask[: max(10, int(0.25 * rate))] = True

    pods = {}
    any_contact = np.zeros(n, bool)
    force_series, field_series, accel_series, shear_series, calibration = {}, {}, {}, {}, {}
    for p in PODS:
        bx, by, bz = get(f"{p}_bx_ut"), get(f"{p}_by_ut"), get(f"{p}_bz_ut")
        if bz is None:
            continue
        rest = np.array([np.median(v[rest_mask]) for v in (bx, by, bz)])
        noise = np.array([_noise(v, rest_mask) for v in (bx, by, bz)])
        dz = bz - rest[2]
        dshear = np.hypot(bx - rest[0], by - rest[1])
        thr = max(6 * noise[2], 150.0)
        touching = dz > thr
        any_contact |= touching

        flags = get(f"{p}_mag_flags")
        rail = int(np.sum((flags.astype(np.int64) & RAIL_FLAG) > 0)) if flags is not None else 0

        # Slip band: high-passed field energy in 0.1 s windows, relative to the rest-window level.
        hp = bz - _moving_avg(bz, 7)
        band = np.sqrt(_moving_avg(hp**2, max(5, int(0.1 * rate))))
        base = float(np.median(band[rest_mask])) or 1.0
        slip_sig = touching & (band > 4 * base)

        gt = get(f"{p}_normal_n")
        est = np.clip(dz, 0, None) / FIELD_PER_NEWTON_UT
        force_series[p] = gt if gt is not None else est
        field_series[p] = dz / 1000
        if f"{p}_shear_n" in c:
            shear_series[p] = get(f"{p}_shear_n")
        if gt is not None and np.sum(gt > 0.2) > 20:
            # How well the field tracks force: the fit a load-cell calibration would make on hardware.
            m = gt > 0.2
            calibration[p] = {"mt_per_n": round(float(np.polyfit(gt[m], dz[m] / 1000, 1)[0]), 3),
                              "r": round(float(np.corrcoef(gt[m], dz[m])[0, 1]), 4), "samples": int(m.sum())}

        acc = np.stack([get(f"{p}_a{a}_mps2") for a in "xyz"], 1) if f"{p}_ax_mps2" in c else None
        gyr = np.stack([get(f"{p}_g{a}_rads") for a in "xyz"], 1) if f"{p}_gx_rads" in c else None
        if acc is not None:
            accel_series[p] = np.linalg.norm(acc, axis=1)

        pods[p] = {
            "rest_field_ut": rest.round(0).tolist(),
            "noise_ut": noise.round(1).tolist(),
            "noise_vs_spec": round(float(np.max(noise / np.array(NOISE_SPEC_UT))), 2),
            "peak_field_delta_mt": round(float(np.max(np.hypot(dz, dshear))) / 1000, 2),
            "peak_force_est_n": round(float(est.max()), 2),
            "peak_force_gt_n": round(float(gt.max()), 2) if gt is not None else None,
            "contact_s": round(float(touching.sum() / rate), 2),
            "rail_frames": rail,
            "slip_signal_s": round(float(slip_sig.sum() / rate), 3),
            "slip_signal_first_s": round(float(t[np.argmax(slip_sig)]), 2) if slip_sig.any() else None,
            "peak_accel_mps2": round(float(np.linalg.norm(acc, axis=1).max()), 1) if acc is not None else None,
            "peak_gyro_rads": round(float(np.linalg.norm(gyr, axis=1).max()), 2) if gyr is not None else None,
        }

    facts: dict = {
        "episode": meta.get("episode"),
        "object": meta.get("object"),
        "mass_g": meta.get("mass_g"),
        "mu": meta.get("mu"),
        "scenario": meta.get("scenario"),
        "safe_grip": meta.get("safe_grip"),
        "outcome": meta.get("outcome"),
        "source": meta.get("source", "sim"),
        "seconds": round(float(t[-1]), 2) if n else 0.0,
        "frames": n,
        "rate_hz": round(rate, 1),
        "dropped_frames": dropped,
        "pods": pods,
        "has_ground_truth_force": any(v["peak_force_gt_n"] is not None for v in pods.values()),
    }

    # Signal-based touch onset vs the truth. In the sim, label_contact turns on before any pod carries
    # force (contact elsewhere on the hand), so ground-truth pod force > 0.1 N is the fairer reference.
    sig_on = float(t[np.argmax(any_contact)]) if any_contact.any() else None
    gt_any = None
    gts = [get(f"{p}_normal_n") for p in pods]
    if gts and all(g is not None for g in gts):
        pressed = np.any(np.stack(gts, 1) > 0.1, axis=1)
        gt_any = float(t[np.argmax(pressed)]) if pressed.any() else None
    if gt_any is None and contact_lbl is not None and np.any(contact_lbl > 0):
        gt_any = float(t[np.argmax(contact_lbl > 0)])
    facts["touch_onset_s"] = round(sig_on, 3) if sig_on is not None else None
    facts["touch_true_onset_s"] = round(gt_any, 3) if gt_any is not None else None
    facts["touch_detect_latency_ms"] = (
        round((sig_on - gt_any) * 1000, 1) if sig_on is not None and gt_any is not None else None
    )

    grip = get("label_grip")
    if grip is not None:
        counts = Counter(int(g) for g in grip)
        facts["grip_state_s"] = {GRIP[k]: round(v / rate, 2) for k, v in sorted(counts.items()) if k < len(GRIP)}
        hold = np.flatnonzero(grip == GRIP.index("hold"))
        facts["time_to_hold_s"] = round(float(t[hold[0]]), 2) if hold.size else None

    slip = get("label_slip")
    if slip is not None:
        facts["slip_label_s"] = round(float(slip.sum() / rate), 3)

    fault = get("label_fault")
    if fault is not None and np.any(fault > 0):
        i = int(np.argmax(fault > 0))
        facts["fault"] = FAULT[int(fault[i])]
        facts["fault_at_s"] = round(float(t[i]), 2)
    else:
        facts["fault"] = None
        facts["fault_at_s"] = None

    margin = get("friction_margin")
    if margin is not None and any_contact.any():
        m = margin[any_contact]
        facts["min_friction_margin"] = round(float(m.min()), 3)
    lift = get("object_lift_m")
    if lift is not None:
        facts["max_lift_cm"] = round(float(lift.max()) * 100, 1)

    # Downsampled series for charts, so nothing downstream needs the raw frames.
    idx = np.linspace(0, n - 1, min(n, 100)).astype(int) if n else np.array([], int)
    pick = lambda d, nd=2: {p: v[idx].round(nd).tolist() for p, v in d.items()}  # noqa: E731
    series = {"t": t[idx].round(2).tolist(), "force": pick(force_series), "field": pick(field_series, 3),
              "accel": pick(accel_series, 1), "shear": pick(shear_series)}
    if margin is not None:
        series["friction_margin"] = margin[idx].round(3).tolist()
    if lift is not None:
        series["lift_cm"] = (lift[idx] * 100).round(2).tolist()
    facts["_series"] = series
    facts["_calibration"] = calibration

    # A few raw rows from first contact, every 0.2 s, for the Data Explorer's "head()" view.
    onset = int(np.argmax(any_contact)) if any_contact.any() else 0
    step = max(1, int(0.2 * rate))
    facts["_sample"] = [
        {"t": round(float(t[i]), 2), **{p: round(float(force_series[p][i]), 2) for p in force_series},
         **({"grip": GRIP[int(grip[i])]} if grip is not None else {})}
        for i in range(onset, min(n, onset + 5 * step), step)
    ]
    return facts
