"""The two general-purpose agents: Data Explorer (tables) and Visualizer (charts).

Pure Python over the snapshot, so both run inside Agentverse-hosted agents (no pandas there).
Explorer answers with describe/crosstab/pivot/group-by tables and shows the pandas expression a
builder would use to reproduce each one; Visualizer picks a chart from the request and always adds
the numbers behind it, so the reply still says something if an image doesn't load.
"""

from __future__ import annotations

import asyncio
import math
import re
import statistics
from collections import Counter, defaultdict

import dermis_analysis as analysis, dermis_charts as charts
from dermis_profiles import BY_KEY, HAND_PODS, PODS, Profile
from dermis_store import DataUnavailable, snapshot

GROUPS = {"object": "object", "scenario": "scenario", "outcome": "outcome", "safegrip": "safe_grip",
          "safe grip": "safe_grip", "mass": "mass_g", "friction": "mu"}
METRICS = {  # word in the request -> fleet column, label
    "peak force": ("peak_force_n", "peak force (N)"), "force": ("peak_force_n", "peak force (N)"),
    "latency": ("latency_ms", "touch detection latency (ms)"), "contact": ("contact_s", "contact time (s)"),
    "slip": ("slip_signal_s", "slip-band time (s)"), "margin": ("min_margin", "min friction margin"),
    "lift": ("max_lift_cm", "max lift (cm)"), "hold": ("time_to_hold_s", "time to stable hold (s)"),
    "duration": ("seconds", "duration (s)"), "mass": ("mass_g", "mass (g)"), "rail": ("rail_frames", "railed frames"),
}
SIGNALS = ("force", "field", "accel", "shear", "margin", "lift")
SIGNAL_LABEL = {"force": "force (N)", "field": "field change (mT)", "accel": "acceleration (m/s²)",
                "shear": "shear force (N)", "margin": "friction margin", "lift": "object lift (cm)"}


# ---------------------------------------------------------------- small table toolkit

def fmt(v, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def md_table(header: list[str], rows: list[list], nd: int = 2) -> str:
    return "\n".join([f"| {' | '.join(header)} |", "|" + "---|" * len(header)] +
                     [f"| {' | '.join(fmt(v, nd) for v in r)} |" for r in rows])


def code(expr: str) -> str:
    return f"`{expr}`"


def describe(values: list[float]) -> list:
    v = sorted(x for x in values if x is not None)
    if not v:
        return [0, None, None, None, None, None]
    return [len(v), statistics.fmean(v), statistics.pstdev(v) if len(v) > 1 else 0.0, v[0],
            statistics.median(v), v[-1]]


def mean(values) -> float | None:
    v = [x for x in values if x is not None]
    return statistics.fmean(v) if v else None


# ---------------------------------------------------------------- data views

def episode_rows() -> tuple[list[dict], str]:
    snap = snapshot()
    return [{**e, "success": e["outcome"] == "success"} for e in snap.episodes], snap.label


def fleet_rows() -> tuple[list[dict], str]:
    """One row per episode with computed metrics, all from the snapshot."""
    snap = snapshot()
    rows = []
    for e in snap.episodes:
        f = snap.facts(e["episode"])
        pod, peak = analysis.peak_force(f)
        rows.append({
            **{k: e.get(k) for k in ("episode", "object", "scenario", "safe_grip", "outcome", "mass_g", "mu", "seconds")},
            "success": e["outcome"] == "success",
            "peak_force_n": peak, "peak_pod": pod,
            "contact_s": sum(v["contact_s"] for v in f["pods"].values()),
            "slip_signal_s": sum(v["slip_signal_s"] for v in f["pods"].values()),
            "rail_frames": sum(v["rail_frames"] for v in f["pods"].values()),
            "latency_ms": f.get("touch_detect_latency_ms"), "min_margin": f.get("min_friction_margin"),
            "max_lift_cm": f.get("max_lift_cm"), "time_to_hold_s": f.get("time_to_hold_s"),
        })
    return rows, snap.label


def filtered(rows: list[dict], parsed: dict) -> tuple[list[dict], str]:
    parts = []
    if parsed.get("object"):
        rows = [r for r in rows if r["object"] == parsed["object"]]
        parts.append(f"object = {parsed['object']}")
    if parsed.get("scenario"):
        rows = [r for r in rows if r["scenario"] == parsed["scenario"]]
        parts.append(f"scenario = {parsed['scenario']}")
    if parsed.get("safe_grip") is not None:
        rows = [r for r in rows if bool(r["safe_grip"]) == parsed["safe_grip"]]
        parts.append(f"SafeGrip {'on' if parsed['safe_grip'] else 'off'}")
    return rows, " · ".join(parts)


def find_group(text: str) -> list[str]:
    t = text.lower()
    found = []
    for m in re.finditer(r"\b(?:by|per|across|for each)\s+(safe ?grip|object|scenario|outcome|mass|friction)s?"
                         r"(?:\s+(?:and|x|×|,)\s+(safe ?grip|object|scenario|outcome|mass|friction)s?)?", t):
        for g in m.groups():
            if g:
                found.append(GROUPS["safegrip" if "safe" in g else g])
    return list(dict.fromkeys(found))


def find_metrics(text: str) -> list[tuple[str, str]]:
    t = text.lower()
    hits = []
    for word, m in METRICS.items():
        pos = t.find(word)
        if pos >= 0 and m not in [h for _, h in hits]:
            hits.append((pos, m))
    return [m for _, m in sorted(hits)]


def all_episode_ids(text: str) -> list[str]:
    return [f"episode_{int(n):04d}" for n in re.findall(r"(?:episode|session|ep|run)[\s_#-]*(\d{1,4})", text.lower())]


def _meta(text: str, parsed: dict) -> dict | None:
    snap = snapshot()
    ids = all_episode_ids(text)
    eid = ids[0] if ids else parsed.get("episode")
    if eid:
        return snap.meta(eid)
    return analysis.pick_episode(BY_KEY["humanoid_arm"], snap.episodes, parsed.get("object"),
                                 parsed.get("scenario"), parsed.get("safe_grip"))


def _reply(md: list[str], headline: str, episode: str | None = None) -> dict:
    return {"markdown": "\n".join(md), "score": None, "headline": headline, "episode": episode}


def _src_line(src: str, n: int | None = None, what: str = "episodes") -> str:
    return f"_{n:,} {what} · {src}_" if n is not None else f"_{src}_"


def _episode_header(meta: dict, src: str) -> str:
    return (f"**{meta['object']}** · `{meta['scenario']}` · SafeGrip {'on' if meta['safe_grip'] else 'off'} · "
            f"outcome **{meta['outcome']}** · _{src}_")


async def respond(p: Profile, text: str, parsed: dict) -> dict:
    try:
        if parsed.get("mode") == "help":
            return _reply([help_text(p)], "help")
        fn = explore if p.kind == "explorer" else visualize
        return await asyncio.to_thread(fn, text, parsed)
    except DataUnavailable as e:
        return _reply([f"**{p.title}**: couldn't load data ({e})."], f"data unavailable: {e}")


def help_text(p: Profile) -> str:
    ex = "\n".join(f"- “{e}”" for e in p.examples)
    return f"## {p.title}\n{p.domain[0].upper() + p.domain[1:]}.\n\nTry:\n{ex}"


# ---------------------------------------------------------------- Data Explorer

def explore(text: str, parsed: dict) -> dict:
    if parsed.get("episode") or all_episode_ids(text):
        return explore_episode(text, parsed)
    groups = find_group(text)
    metrics = find_metrics(text)
    if groups or metrics:
        return explore_groupby(text, parsed, groups or ["object"], metrics)
    return explore_overview(text, parsed)


def explore_overview(text: str, parsed: dict) -> dict:
    rows, src = episode_rows()
    rows, filt = filtered(rows, parsed)
    if not rows:
        return _reply(["No recordings match that filter."], "no match")
    objs = sorted({r["object"] for r in rows})
    outcomes = sorted({r["outcome"] for r in rows})
    scens = sorted({r["scenario"] for r in rows})
    ct = Counter((r["object"], r["outcome"]) for r in rows)

    def rate(sel):
        sel = list(sel)
        return sum(r["success"] for r in sel) / len(sel) * 100 if sel else None

    piv = {o: {s: rate(r for r in rows if r["object"] == o and r["scenario"] == s) for s in scens} for o in objs}
    sg = {o: {b: rate(r for r in rows if r["object"] == o and bool(r["safe_grip"]) == b) for b in (False, True)}
          for o in objs}

    ct_rows = [[o] + [ct[(o, oc)] for oc in outcomes] + [sum(ct[(o, oc)] for oc in outcomes)] for o in objs]
    ct_rows.append(["total"] + [sum(ct[(o, oc)] for o in objs) for oc in outcomes] + [len(rows)])
    piv_rows = [[o] + [round(piv[o][s]) if piv[o][s] is not None else None for s in scens] for o in objs]
    sg_rows = [[o, round(sg[o][False]) if sg[o][False] is not None else None,
                round(sg[o][True]) if sg[o][True] is not None else None,
                round(sg[o][True] - sg[o][False]) if None not in sg[o].values() else None] for o in objs]
    desc_rows = [[c] + describe([r[c] for r in rows])[1:] for c in ("mass_g", "mu", "seconds", "frames")]

    avg = {o: mean(piv[o].values()) for o in objs}
    best, worst = max(avg, key=avg.get), min(avg, key=avg.get)
    drop = mean(piv[o].get("drop") for o in objs) if "drop" in scens else None
    md = [
        "## Data Explorer · dataset overview" + (f" ({filt})" if filt else ""),
        _src_line(src, len(rows)) + f" · {sum(r['frames'] for r in rows):,} frames · "
        f"{sum(r['seconds'] for r in rows) / 60:.1f} min", "",
        charts.stacked_bar(objs, {oc: [ct[(o, oc)] for o in objs] for oc in outcomes}, "Outcomes by object", "episodes"), "",
        "### Outcomes by object", code("pd.crosstab(df.object, df.outcome, margins=True)"), "",
        md_table(["object"] + outcomes + ["total"], ct_rows), "",
        "### Success rate %, object × scenario", code("df.pivot_table('success', 'object', 'scenario') * 100"), "",
        md_table(["object"] + scens, piv_rows), "",
        "### SafeGrip effect", code("df.pivot_table('success', 'object', 'safe_grip') * 100"), "",
        md_table(["object", "SafeGrip off %", "SafeGrip on %", "Δ points"], sg_rows), "",
        "### Numeric columns", code("df[['mass_g','mu','seconds','frames']].describe()"), "",
        md_table(["column", "mean", "std", "min", "50%", "max"], desc_rows), "",
        f"**Read-out:** most reliable object **{best}** ({avg[best]:.0f}% avg success), least **{worst}** "
        f"({avg[worst]:.0f}%)." + (f" The `drop` scenario succeeds {drop:.0f}% of the time." if drop is not None else ""),
        "", "Ask for a breakdown, e.g. “peak force by object and scenario” or “latency by safe grip”.",
    ]
    return _reply(md, f"{len(rows)} episodes, {len(objs)} objects")


def explore_groupby(text: str, parsed: dict, groups: list[str], metrics: list[tuple[str, str]]) -> dict:
    computed = [m for m in metrics if m[0] not in ("mass_g", "seconds")]
    rows, src = fleet_rows() if computed else episode_rows()
    # Grouping by a column makes a filter on that same column meaningless.
    parsed = {k: v for k, v in parsed.items() if k not in groups}
    rows, filt = filtered(rows, parsed)
    if not rows:
        return _reply(["No recordings match that filter."], "no match")
    buckets: dict = defaultdict(list)
    for r in rows:
        buckets[tuple(fmt(r[g]) for g in groups)].append(r)
    keys = sorted(buckets)

    if not metrics:
        stat = {k: sum(r["success"] for r in buckets[k]) / len(buckets[k]) * 100 for k in keys}
        table_rows = [list(k) + [len(buckets[k]), stat[k], mean(r["seconds"] for r in buckets[k])] for k in keys]
        header = groups + ["episodes", "success %", "mean seconds"]
        expr = f"df.groupby({groups}).agg(episodes=('episode','count'), success_pct=('success','mean'))"
        label = "success %"
    else:
        col, label = metrics[0]
        label = f"mean {label}"
        stat, table_rows = {}, []
        for k in keys:
            d = describe([r[col] for r in buckets[k]])
            stat[k] = d[1]
            table_rows.append(list(k) + d)
        header = groups + ["count", "mean", "std", "min", "median", "max"]
        expr = f"df.groupby({groups})['{col}'].agg(['count','mean','std','min','median','max'])"

    if len(groups) == 1:
        chart = charts.bar_chart([k[0] for k in keys], [round(stat[k] or 0, 2) for k in keys],
                                 f"{label} by {groups[0]}", label)
    else:
        firsts = sorted({k[0] for k in keys})
        seconds_ = sorted({k[1] for k in keys})
        chart = charts.grouped_bar(firsts, {f"{groups[1]}={b}": [round(stat.get((a, b)) or 0, 2) for a in firsts]
                                            for b in seconds_}, f"{label} by {groups[0]} and {groups[1]}", label)
    valid = {k: v for k, v in stat.items() if v is not None}
    top, low = max(valid, key=valid.get), min(valid, key=valid.get)
    md = [f"## Data Explorer · {label} by {' × '.join(groups)}" + (f" ({filt})" if filt else ""),
          _src_line(src, len(rows)), "", chart, "", code(expr), "", md_table(header, table_rows), "",
          f"**Highest:** {' / '.join(top)} ({valid[top]:.2f}) · **Lowest:** {' / '.join(low)} ({valid[low]:.2f})"]
    return _reply(md, f"{label} by {', '.join(groups)}")


def explore_episode(text: str, parsed: dict) -> dict:
    meta = _meta(text, parsed)
    if meta is None:
        return _reply(["I don't have that episode."], "episode not found")
    snap = snapshot()
    f = snap.facts(meta["episode"])
    force = f["_series"]["force"]
    gt = f["has_ground_truth_force"]
    key = "peak_force_gt_n" if gt else "peak_force_est_n"
    desc_rows = [[HAND_PODS[p]] + describe(force[p])[1:] + [f["pods"][p][key]] for p in PODS if p in force]

    lines = []
    if f.get("grip_state_s"):
        lines += ["### Time in each grip state", code("df.label_grip.map(GRIP).value_counts() / 350"), "",
                  md_table(["grip state", "seconds"], sorted(f["grip_state_s"].items(), key=lambda kv: -kv[1])), ""]
    if f.get("_calibration"):
        cal_rows = [[HAND_PODS[p], c["samples"], c["mt_per_n"], c["r"]] for p, c in f["_calibration"].items()]
        lines += ["### Field vs force (calibration check)",
                  code("np.polyfit(df[f'{pod}_normal_n'], df[f'{pod}_bz_ut'] / 1000, 1)"), "",
                  md_table(["pod", "samples", "mT per N", "r"], cal_rows, nd=3), "",
                  "_A slope near 1.5 mT/N with r ≈ 1 means the field tracks force as the sim assumes. "
                  "On hardware this table is the calibration you'd fit against a load cell._", ""]
    if f.get("_sample"):
        cols = list(f["_sample"][0])
        lines += ["### Frames from first contact, every 0.2 s", code("df.iloc[onset:onset+350:70]"), "",
                  md_table([HAND_PODS.get(c, "t (s)" if c == "t" else c) for c in cols],
                           [[r[c] for c in cols] for r in f["_sample"]]), ""]

    peaks = [f["pods"][p][key] for p in PODS if p in force]
    means = [mean(x for x in force[p] if x > 0.2) or 0 for p in PODS if p in force]
    md = [f"## Data Explorer · {meta['episode']}", _episode_header(meta, snap.label), "",
          charts.grouped_bar([HAND_PODS[p] for p in PODS if p in force],
                             {"peak N": peaks, "mean in contact N": [round(m, 2) for m in means]},
                             f"{meta['episode']} · force per pod", "N"), "",
          f"### {'Normal force (N)' if gt else 'Force estimated from field (N, uncalibrated)'} per pod",
          code("df[[f'{pod}_normal_n' for pod in PODS]].describe()"), "",
          md_table(["pod", "mean", "std", "min", "median", "max (downsampled)", "true peak"], desc_rows), "",
          *lines,
          f"_{f['frames']:,} frames at {f['rate_hz']:g} Hz; stats above use the 100-point downsampled trace, "
          "the true peak comes from every frame._"]
    return _reply(md, f"{meta['episode']} tables", meta["episode"])


# ---------------------------------------------------------------- Visualizer

CHART_WORDS = [
    ("heatmap", r"heat ?map|grid|matrix"),
    ("histogram", r"histogram|distribution|spread"),
    ("pie", r"\bpie\b|donut|doughnut|share|proportion|breakdown"),
    ("radar", r"radar|spider|per[- ]pod profile|fingerprint"),
    ("compare", r"compare|overlay|twin|side by side|\bvs\b|versus|safe ?grip on (?:and|vs) off"),
    ("line", r"\bline\b|timeline|time series|signal|trace|over time"),
    ("dashboard", r"dashboard|overview|everything"),
]


def chart_kind(text: str) -> str | None:
    t = text.lower()
    if re.search(r"scatter|correlat", t) or (re.search(r"\bvs\b|versus|against", t) and len(find_metrics(t)) >= 2):
        return "scatter"
    for kind, pat in CHART_WORDS:
        if re.search(pat, t):
            return kind
    return None


def signal_of(text: str) -> str:
    t = text.lower()
    for s, pat in (("field", r"field|magnet|\bbz\b|raw"), ("accel", r"accel|imu|vibration|gyro"),
                   ("shear", r"shear"), ("margin", r"margin|friction"), ("lift", r"lift|height")):
        if re.search(pat, t):
            return s
    return "force"


def visualize(text: str, parsed: dict) -> dict:
    kind = chart_kind(text)
    wants_episode = bool(parsed.get("episode") or all_episode_ids(text))
    if kind == "heatmap":
        return viz_heatmap(parsed)
    if kind == "histogram":
        return viz_histogram(text, parsed)
    if kind == "pie":
        return viz_pie(parsed)
    if kind == "scatter":
        return viz_scatter(text, parsed)
    if kind == "radar":
        return viz_radar(text, parsed)
    if kind == "compare":
        return viz_compare(text, parsed)
    if kind == "line" or (wants_episode and signal_of(text) != "force"):
        return viz_line(text, parsed)
    if wants_episode or parsed.get("object") or parsed.get("scenario"):
        return viz_episode_dashboard(text, parsed)
    return viz_fleet_dashboard(parsed)


def _signal(f: dict, signal: str) -> dict[str, list[float]]:
    s = f["_series"]
    if signal == "margin":
        return {"friction margin": s["friction_margin"]} if "friction_margin" in s else {}
    if signal == "lift":
        return {"lift (cm)": s["lift_cm"]} if "lift_cm" in s else {}
    per_pod = s.get(signal, {})
    return {HAND_PODS[p]: v for p, v in per_pod.items() if max(map(abs, v), default=0) > 1e-3}


def _scalar(f: dict, signal: str) -> list[float]:
    """One value per sample: total force across pods, or the largest pod for other signals."""
    series = _signal(f, signal)
    if not series:
        return [0.0] * len(f["_series"]["t"])
    cols = list(series.values())
    agg = sum if signal == "force" else max
    return [round(agg(c[i] for c in cols), 2) for i in range(len(cols[0]))]


def viz_line(text: str, parsed: dict) -> dict:
    meta = _meta(text, parsed)
    if meta is None:
        return _reply(["No recording matches that."], "no match")
    snap = snapshot()
    f = snap.facts(meta["episode"])
    signal = signal_of(text)
    series, label = _signal(f, signal), SIGNAL_LABEL[signal]
    if not series:
        return _reply([f"{meta['episode']} has no `{signal}` data."], "no data", meta["episode"])
    t = f["_series"]["t"]
    md = [f"## Visualizer · {label} over time · {meta['episode']}", _episode_header(meta, snap.label), "",
          charts.line(t, series, f"{meta['episode']} · {label}", "time (s)", label), "",
          "| Series | Peak | At (s) | Trace |", "|---|---|---|---|"]
    for k, v in series.items():
        i = max(range(len(v)), key=v.__getitem__)
        md.append(f"| {k} | {v[i]:.2f} | {t[i]} | `{charts.sparkline(v, 30)}` |")
    md += ["", f"Other signals: {', '.join(s for s in SIGNALS if s != signal)} (e.g. “plot the raw field for episode 4”)."]
    return _reply(md, f"{label} line", meta["episode"])


def _radar(f: dict) -> tuple[str, list[str]]:
    force = f["_series"]["force"]
    key = "peak_force_gt_n" if f["has_ground_truth_force"] else "peak_force_est_n"
    axes, peak, contact, acc, rows = [], [], [], [], []
    for p in PODS:
        if p not in force:
            continue
        v = f["pods"][p]
        axes.append(HAND_PODS[p])
        peak.append(v[key] or 0)
        contact.append(v["contact_s"])
        acc.append(v["peak_accel_mps2"] or 0)
        rows.append(f"| {HAND_PODS[p]} | {v[key]:.2f} N | {v['contact_s']:.2f} | {v['peak_accel_mps2'] or 0:.1f} | "
                    f"`{charts.sparkline(force[p], 24)}` |")
    norm = lambda vals: [round(x / max(vals) * 100, 1) if max(vals) else 0 for x in vals]  # noqa: E731
    md = charts.radar(axes, {"peak force": norm(peak), "contact time": norm(contact), "peak accel": norm(acc)},
                      "Per-pod profile (% of the largest pod)")
    return md, rows


def viz_episode_dashboard(text: str, parsed: dict) -> dict:
    meta = _meta(text, parsed)
    if meta is None:
        return _reply(["No recording matches that."], "no match")
    snap = snapshot()
    f = snap.facts(meta["episode"])
    t = f["_series"]["t"]
    radar_md, rows = _radar(f)
    extra = {**_signal(f, "margin"), **_signal(f, "lift")}
    md = [f"## Visualizer · dashboard · {meta['episode']}", _episode_header(meta, snap.label), "",
          charts.line(t, _signal(f, "force"), f"{meta['episode']} · force per pod (N)", "time (s)", "N"), "",
          radar_md, ""]
    if _signal(f, "accel"):
        md += [charts.line(t, _signal(f, "accel"), f"{meta['episode']} · acceleration per pod", "time (s)", "m/s²"), ""]
    if extra:
        md += [charts.line(t, extra, f"{meta['episode']} · friction margin and lift (cm)", "time (s)", "value"), ""]
    md += ["| Pod | Peak force | Contact (s) | Peak accel | Trace |", "|---|---|---|---|---|", *rows]
    return _reply(md, "episode dashboard", meta["episode"])


def viz_radar(text: str, parsed: dict) -> dict:
    meta = _meta(text, parsed)
    if meta is None:
        return _reply(["No recording matches that."], "no match")
    snap = snapshot()
    radar_md, rows = _radar(snap.facts(meta["episode"]))
    md = [f"## Visualizer · per-pod profile · {meta['episode']}", _episode_header(meta, snap.label), "", radar_md, "",
          "Each axis is a pod; each line is scaled to the largest pod, so shape shows which fingers did the work.", "",
          "| Pod | Peak force | Contact (s) | Peak accel | Trace |", "|---|---|---|---|---|", *rows]
    return _reply(md, "radar", meta["episode"])


def _interp(grid: list[float], t: list[float], y: list[float]) -> list[float | None]:
    out, j = [], 0
    for g in grid:
        if g > t[-1]:
            out.append(None)
            continue
        while j < len(t) - 2 and t[j + 1] < g:
            j += 1
        t0, t1 = t[j], t[j + 1]
        w = 0.0 if t1 == t0 else (g - t0) / (t1 - t0)
        out.append(round(y[j] + (y[j + 1] - y[j]) * min(max(w, 0.0), 1.0), 2))
    return out


def viz_compare(text: str, parsed: dict) -> dict:
    snap = snapshot()
    ids = all_episode_ids(text)
    if len(ids) >= 2:
        metas = [snap.meta(i) for i in ids[:3]]
        missing = [i for i, m in zip(ids, metas) if m is None]
        if missing:
            return _reply([f"I don't have {', '.join(missing)}."], "not found")
    else:
        first = _meta(text, parsed)
        if first is None:
            return _reply(["No recording matches that."], "no match")
        twin = analysis.counterfactual(first, snap.episodes)
        metas = [first] + ([twin] if twin else [])
    signal = signal_of(text)
    label = "total normal force (N)" if signal == "force" else f"largest-pod {SIGNAL_LABEL[signal]}"
    traces, peaks, table = {}, [], []
    for m in metas:
        f = snap.facts(m["episode"])
        vals = _scalar(f, signal)
        name = f"{m['episode']} · SafeGrip {'on' if m['safe_grip'] else 'off'} · {m['outcome']}"
        traces[name] = (f["_series"]["t"], vals)
        peaks.append(max(vals))
        table.append(f"| {m['episode']} | {m['object']} | {m['scenario']} | {'on' if m['safe_grip'] else 'off'} | "
                     f"**{m['outcome']}** | {max(vals):.2f} | `{charts.sparkline(vals, 24)}` |")
    # Episodes have different lengths: resample all onto one time grid, blank after each one ends.
    end = max(t[-1] for t, _ in traces.values())
    grid = [round(end * i / 119, 2) for i in range(120)]
    aligned = {k: _interp(grid, t, y) for k, (t, y) in traces.items()}
    md = [f"## Visualizer · comparison · {label}",
          "Same object, scenario and seed with SafeGrip flipped." if len(ids) < 2 and len(metas) == 2 else "", "",
          charts.line(grid, aligned, f"Overlay · {label}", "time (s)", label), "",
          "| Episode | Object | Scenario | SafeGrip | Outcome | Peak | Trace |", "|---|---|---|---|---|---|---|", *table]
    if len(metas) == 2:
        lower = metas[peaks.index(min(peaks))]["episode"]
        md += ["", f"**Peak difference:** {abs(peaks[0] - peaks[1]):.2f}, lower in {lower}."]
    return _reply(md, "comparison", metas[0]["episode"])


def viz_pie(parsed: dict) -> dict:
    rows, src = episode_rows()
    rows, filt = filtered(rows, parsed)
    if not rows:
        return _reply(["No recordings match that filter."], "no match")
    vc = Counter(r["outcome"] for r in rows).most_common()
    md = ["## Visualizer · outcome share" + (f" ({filt})" if filt else ""), _src_line(src, len(rows)), "",
          charts.pie([k for k, _ in vc], [v for _, v in vc], "Outcomes" + (f" · {filt}" if filt else "")), "",
          "| Outcome | Episodes | Share |", "|---|---|---|"]
    md += [f"| {k} | {v} | {v / len(rows) * 100:.0f}% `{'█' * round(v / len(rows) * 30)}` |" for k, v in vc]
    return _reply(md, "outcome pie")


def viz_heatmap(parsed: dict) -> dict:
    rows, src = episode_rows()
    rows, filt = filtered(rows, parsed)
    objs = sorted({r["object"] for r in rows})
    scens = sorted({r["scenario"] for r in rows})
    cell = {}
    for o in objs:
        for s in scens:
            sel = [r["success"] for r in rows if r["object"] == o and r["scenario"] == s]
            cell[(o, s)] = sum(sel) / len(sel) * 100 if sel else None

    def swatch(v):
        return "⬜" if v is None else "🟩" if v >= 75 else "🟨" if v >= 50 else "🟧" if v >= 25 else "🟥"

    lines = [f"| **{o}** | " + " | ".join(f"{swatch(cell[(o, s)])} {fmt(cell[(o, s)], 0)}%" for s in scens) + " |"
             for o in objs]
    valid = {k: v for k, v in cell.items() if v is not None}
    if not valid:
        return _reply(["No recordings match that filter."], "no match")
    hard, easy = min(valid, key=valid.get), max(valid, key=valid.get)
    md = ["## Visualizer · success heatmap, object × scenario" + (f" ({filt})" if filt else ""), _src_line(src, len(rows)), "",
          "| object | " + " | ".join(scens) + " |", "|---|" + "---|" * len(scens), *lines, "",
          "🟩 ≥75% · 🟨 50–74% · 🟧 25–49% · 🟥 <25%", "",
          charts.grouped_bar(objs, {s: [round(cell[(o, s)] or 0) for o in objs] for s in scens},
                             "Success % by object and scenario", "success %"), "",
          f"**Hardest cell:** {hard[0]} in `{hard[1]}` ({valid[hard]:.0f}%). "
          f"**Easiest:** {easy[0]} in `{easy[1]}` ({valid[easy]:.0f}%)."]
    return _reply(md, "heatmap")


def viz_histogram(text: str, parsed: dict) -> dict:
    col, label = (find_metrics(text) or [METRICS["peak force"]])[0]
    rows, src = fleet_rows()
    rows, filt = filtered(rows, parsed)
    vals = sorted(r[col] for r in rows if r[col] is not None)
    if not vals:
        return _reply([f"No `{label}` values for that filter."], "no data")
    q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]  # noqa: E731
    by_outcome = defaultdict(list)
    for r in rows:
        if r[col] is not None:
            by_outcome[r["outcome"]].append(r[col])
    med = sorted(((k, statistics.median(v)) for k, v in by_outcome.items()), key=lambda kv: kv[1])
    md = [f"## Visualizer · distribution of {label}" + (f" ({filt})" if filt else ""), _src_line(src, len(vals)), "",
          charts.histogram(vals, 12, f"{label} across {len(vals)} episodes", label), "",
          f"p10 **{q(0.1):.2f}** · median **{statistics.median(vals):.2f}** · p90 **{q(0.9):.2f}** · max **{vals[-1]:.2f}**", "",
          "| Outcome | Median |", "|---|---|"] + [f"| {k} | {v:.2f} |" for k, v in med]
    return _reply(md, f"{label} histogram")


def viz_scatter(text: str, parsed: dict) -> dict:
    metrics = find_metrics(text)
    # "Y vs X": the first metric named goes on the y axis. Defaults: peak force vs mass.
    if len(metrics) >= 2:
        y, x = metrics[:2]
    else:
        y = metrics[0] if metrics else METRICS["peak force"]
        x = METRICS["mass"] if y != METRICS["mass"] else METRICS["peak force"]
    rows, src = fleet_rows()
    rows, filt = filtered(rows, parsed)
    pts = [r for r in rows if r[x[0]] is not None and r[y[0]] is not None]
    groups = defaultdict(list)
    for r in pts:
        groups[r["outcome"]].append((round(r[x[0]], 3), round(r[y[0]], 3)))
    try:
        r_val = statistics.correlation([r[x[0]] for r in pts], [r[y[0]] for r in pts])
    except (statistics.StatisticsError, ValueError):
        r_val = float("nan")
    strength = "strong" if abs(r_val) > 0.6 else "moderate" if abs(r_val) > 0.3 else "weak"
    md = [f"## Visualizer · {y[1]} vs {x[1]}" + (f" ({filt})" if filt else ""), _src_line(src, len(pts)), "",
          charts.scatter(dict(sorted(groups.items())), f"{y[1]} vs {x[1]} (colour = outcome)", x[1], y[1]), "",
          f"Pearson r = **{r_val:.2f}** ({strength} {'positive' if r_val > 0 else 'negative'} relationship).", "",
          f"| Outcome | n | mean {x[1]} | mean {y[1]} |", "|---|---|---|---|"]
    md += [f"| {o} | {len(g)} | {mean(p[0] for p in g):.2f} | {mean(p[1] for p in g):.2f} |"
           for o, g in sorted(groups.items())]
    return _reply(md, "scatter")


def viz_fleet_dashboard(parsed: dict) -> dict:
    rows, src = episode_rows()
    rows, filt = filtered(rows, parsed)
    vc = Counter(r["outcome"] for r in rows).most_common()
    objs = sorted({r["object"] for r in rows})

    def rate(sel):
        sel = list(sel)
        return sum(r["success"] for r in sel) / len(sel) * 100 if sel else 0.0

    by_obj = {o: rate(r for r in rows if r["object"] == o) for o in objs}
    sg = {b: [round(rate(r for r in rows if r["object"] == o and bool(r["safe_grip"]) == b)) for o in objs]
          for b in (True, False)}
    md = ["## Visualizer · dataset dashboard" + (f" ({filt})" if filt else ""), _src_line(src, len(rows)), "",
          charts.pie([k for k, _ in vc], [v for _, v in vc], "Outcomes"), "",
          charts.grouped_bar(objs, {"SafeGrip on": sg[True], "SafeGrip off": sg[False]},
                             "Success % by object, SafeGrip on vs off", "success %"), "",
          "| Object | Success | Bar |", "|---|---|---|"]
    md += [f"| {o} | {v:.0f}% | `{'█' * round(v / 5)}` |" for o, v in sorted(by_obj.items(), key=lambda kv: -kv[1])]
    md += ["", "More: “histogram of peak force”, “scatter latency vs mass”, “heatmap”, "
               "“plot episode 4”, “compare episode 4 and episode 0”, “radar of episode 40”."]
    return _reply(md, "fleet dashboard")
