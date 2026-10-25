#!/usr/bin/env python3
"""Offline slip-risk heuristic: a port of the dashboard rules (dashboard/app.js, derivations.js).

Input: raw Dermis JSONL (one frame per line, with force_n from the force-calibrated firmware).
Output: per-patch CSV (t, force, contact, transverse rate, slip evidence, slip %) and a summary.

Rules, per MLX channel:
  1. Filter: 5-sample median + EMA (b: 0.10 s, force: 0.12 s).
  2. Zero calibration: the first --cal-seconds of the log (device still, pads unloaded) give the
     baseline field, baseline force, and rate-domain noise floors (median + 6*MAD, min 40 uT/s, 0.2 N/s).
  3. Contact: force above baseline, hysteresis 0.25 N on / 0.12 N off. Needs reliable force.
  4. Transverse rate: field velocity minus its component along the (field - baseline) direction;
     only when |field - baseline| > max(50, 3*noise).
  5. Instant score (contact held >= 0.35 s):
       field = max(0, transverse - fieldFloor) / max(500, 3*fieldFloor)
       unload = max(0, -forceRate - forceFloor) / max(1, 4*forceFloor)
       score = 100 * (1 - exp(-(0.8*field + unload)))
  6. Evidence: EMA of score, tau 0.35 s. Active after 3 frames >= 42, inactive after 5 frames <= 18.
     Reported slip = evidence while active, else 0. Losing contact clears everything.

A heuristic, not measured shear.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

CHANNEL_PATCH = {0: "palm", 1: "middle", 2: "thumb", 3: "index"}  # S1..S4
CAL_SECONDS = 5.0


def median(v):
    v = [x for x in v if math.isfinite(x)]
    return statistics.median(v) if v else None


def mad(v, c=None):
    c = median(v) if c is None else c
    m = None if c is None else median([abs(x - c) for x in v if math.isfinite(x)])
    return None if m is None else 1.4826 * m


class MedianEma:
    def __init__(self, window, tau):
        self.window, self.tau, self.hist, self.value, self.last = window, tau, [], None, None

    def update(self, x, t):
        x = [float(i) for i in (x if isinstance(x, (list, tuple)) else [x])]
        if not all(map(math.isfinite, x)):
            return None
        self.hist = (self.hist + [x])[-self.window:]
        center = [median([h[i] for h in self.hist]) for i in range(len(x))]
        if self.value is None:
            self.value = center
        else:
            dt = min(max(t - self.last, 0.0), 0.5) if self.last is not None else 0.0
            a = 1.0 if self.tau == 0 else 1 - math.exp(-dt / self.tau)
            self.value = [v + a * (c - v) for v, c in zip(self.value, center)]
        self.last = t
        return self.value[:]


def norm(v):
    return math.sqrt(sum(i * i for i in v))


def slip_score(contact, transverse, force_rate, field_floor=40.0, force_floor=0.2):
    if not contact:
        return 0.0
    field = max(0.0, transverse - field_floor) / max(500.0, 3 * field_floor)
    unload = max(0.0, -force_rate - force_floor) / max(1.0, 4 * force_floor)
    return min(100.0, max(0.0, 100 * (1 - math.exp(-(0.8 * field + unload)))))


class Channel:
    def __init__(self):
        self.fb, self.ff = MedianEma(5, 0.10), MedianEma(5, 0.12)
        self.cal = {"b": [], "f": [], "high": 0, "n": 0, "fr": [], "br": [], "pb": None, "pf": None, "pt": None}
        self.base = None
        self.p = dict(b=None, force=None, t=None, contact=False, age=0.0, ev=0.0, active=False, on=0, off=0)

    def calibrate(self, b, f, flags, t):
        c = self.cal
        c["b"].append(b)
        if f is not None:
            c["f"].append(f)
        c["n"] += 1
        c["high"] += "FORCE_CLAMPED_HIGH" in flags
        if c["pt"] is not None and 0.01 < t - c["pt"] < 0.5:
            c["br"].append(norm([x - y for x, y in zip(b, c["pb"])]) / (t - c["pt"]))
            if f is not None and c["pf"] is not None:
                c["fr"].append(abs(f - c["pf"]) / (t - c["pt"]))
        c["pb"], c["pf"], c["pt"] = b, f, t

    def finish(self):
        c = self.cal
        if len(c["b"]) < 12:
            return
        b0 = [median([v[i] for v in c["b"]]) for i in range(3)]
        noise = max(1.0, norm([mad([v[i] for v in c["b"]], b0[i]) or 0 for i in range(3)]))
        f0 = median(c["f"]) or 0.0
        hi = c["high"] / c["n"] if c["n"] else 1.0
        brc, frc = median(c["br"]) or 0.0, median(c["fr"]) or 0.0
        self.base = dict(
            b=b0, noise=noise, force=f0,
            usable=len(c["f"]) >= 12 and hi < 0.2 and f0 < 9.7,
            field_floor=max(40.0, brc + 6 * (mad(c["br"], brc) or 0)),
            force_floor=max(0.2, frc + 6 * (mad(c["fr"], frc) or 0)))

    def step(self, mag, t):
        p, base = self.p, self.base
        flags = mag.get("flags") or []
        b = self.fb.update(mag["b_ut"], t)
        fv = mag.get("force_valid") is not False and isinstance(mag.get("force_n"), (int, float))
        f = self.ff.update(float(mag["force_n"]), t)[0] if fv else None
        if self.base is None and t - self.t0 < CAL_SECONDS:
            self.calibrate(b, f, flags, t)
        elif self.base is None:
            self.finish()
            base = self.base
        if base is None:
            return None
        reliable = fv and ("FORCE_CAL_READY" in flags or base["usable"])
        force = max(0.0, f - base["force"]) if reliable and f is not None else None
        dt = min(max(t - p["t"], 0.001), 0.5) if p["t"] is not None else 0.1
        force_rate = (force - p["force"]) / dt if force is not None and p["force"] is not None else 0.0
        contact = p["contact"] if reliable else False
        if force is not None:
            if not contact and force >= 0.25:
                contact = True
            elif contact and force < 0.12:
                contact = False
        age = (p["age"] + dt if p["contact"] else 0.0) if contact else 0.0
        trans = 0.0
        if p["b"] is not None:
            disp = [x - y for x, y in zip(b, base["b"])]
            length = norm(disp)
            if length > max(50.0, 3 * base["noise"]):
                axis = [d / length for d in disp]
                speed = [(x - y) / dt for x, y in zip(b, p["b"])]
                axial = sum(s * a for s, a in zip(speed, axis))
                trans = norm([s - axial * a for s, a in zip(speed, axis)])
        inst = slip_score(contact and age >= 0.35, trans, force_rate, base["field_floor"], base["force_floor"])
        ev = p["ev"] + (1 - math.exp(-dt / 0.35)) * (inst - p["ev"])
        on = p["on"] + 1 if ev >= 42 else 0
        off = p["off"] + 1 if ev <= 18 else 0
        active = p["active"]
        if not contact:
            active, on, off = False, 0, 0
        elif not active and on >= 3:
            active = True
        elif active and off >= 5:
            active = False
        self.p = dict(b=b, force=force, t=t, contact=contact, age=age, ev=ev, active=active, on=on, off=off)
        return dict(force=force, contact=contact, transverse=trans, force_rate=force_rate,
                    evidence=ev, slip=ev if active else 0.0, base_ok=base["usable"])


def run(path, out, cal_seconds):
    global CAL_SECONDS
    CAL_SECONDS = cal_seconds
    chans, rows, t_start = {}, [], None
    for line in Path(path).read_text().splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        t = d["t_device_us"] / 1e6
        t_start = t if t_start is None else t_start
        for name, pod in d.get("pods", {}).items():
            mag = pod.get("mag") or {}
            ch = mag.get("ch")
            if ch not in CHANNEL_PATCH or not mag.get("valid") or not mag.get("b_ut"):
                continue
            c = chans.setdefault(ch, Channel())
            c.t0 = t_start
            r = c.step(mag, t)
            if r:
                rows.append(dict(t=round(t - t_start, 3), patch=CHANNEL_PATCH[ch], ch=f"S{ch + 1}", **r))
    if out and rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return chans, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl")
    ap.add_argument("--out", help="per-frame CSV")
    ap.add_argument("--cal-seconds", type=float, default=CAL_SECONDS)
    a = ap.parse_args()
    chans, rows = run(a.jsonl, a.out, a.cal_seconds)
    if not rows:
        sys.exit("no calibrated frames (log shorter than --cal-seconds or no valid MLX data)")
    print(f"{'patch':8}{'ch':4}{'frames':>7}{'contact%':>10}{'slip%time':>10}{'peak slip':>10}  force-cal")
    for ch, name in CHANNEL_PATCH.items():
        r = [x for x in rows if x["ch"] == f"S{ch + 1}"]
        if not r:
            continue
        n = len(r)
        print(f"{name:8}S{ch + 1:<3}{n:>7}{100 * sum(x['contact'] for x in r) / n:>10.1f}"
              f"{100 * sum(x['slip'] > 0 for x in r) / n:>10.1f}{max(x['slip'] for x in r):>10.1f}"
              f"  {'ok' if r[0]['base_ok'] else 'NOT usable: contact/slip disabled'}")
    if a.out:
        print("wrote", a.out)


if __name__ == "__main__":
    main()
