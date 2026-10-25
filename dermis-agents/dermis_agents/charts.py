"""Charts as QuickChart image URLs, so they render inline in ASI:One's markdown chat."""

from __future__ import annotations

import json
from urllib.parse import quote

from . import config

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
SPARK = "▁▂▃▄▅▆▇█"


def _url(chart: dict, w: int = 720, h: int = 340) -> str:
    c = json.dumps(chart, separators=(",", ":"))
    return f"{config.QUICKCHART_URL}?w={w}&h={h}&bkg=white&c={quote(c)}"


def image(alt: str, chart: dict, **kw) -> str:
    return f"![{alt}]({_url(chart, **kw)})" if config.CHARTS else ""


def sparkline(values: list[float], width: int = 40) -> str:
    if not values:
        return ""
    step = max(1, len(values) // width)
    v = values[::step]
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    return "".join(SPARK[int((x - lo) / span * (len(SPARK) - 1))] for x in v)


def force_chart(facts: dict, pod_labels: dict, limit_n: float | None) -> str:
    s = facts["_series"]
    force = s["force"]
    pods = [p for p in pod_labels if p in force]
    datasets = [
        {"label": pod_labels[p], "data": force[p], "borderColor": COLORS[i % len(COLORS)],
         "fill": False, "pointRadius": 0, "borderWidth": 2}
        for i, p in enumerate(pods) if max(force[p] or [0]) > 0.05
    ]
    if limit_n:
        datasets.append({"label": f"limit {limit_n:g} N", "data": [limit_n] * len(s["t"]),
                         "borderColor": "#6b7280", "borderDash": [6, 4], "fill": False,
                         "pointRadius": 0, "borderWidth": 1})
    kind = "reference force" if facts["has_ground_truth_force"] else "estimated from field, uncalibrated"
    title = f"{facts['episode']} · {facts['object']} · normal force per pod (N, {kind})"
    if facts.get("fault"):
        title += f" · {facts['fault']} at {facts['fault_at_s']} s"
    chart = {
        "type": "line",
        "data": {"labels": s["t"], "datasets": datasets},
        "options": {
            "title": {"display": True, "text": title},
            "scales": {"xAxes": [{"scaleLabel": {"display": True, "labelString": "time (s)"},
                                  "ticks": {"maxTicksLimit": 12}}],
                       "yAxes": [{"scaleLabel": {"display": True, "labelString": "N"}}]},
        },
    }
    return image("force per pod", chart)


def fleet_chart(fleet: dict, title: str) -> str:
    objs = list(fleet["by_object"])
    on = [round((fleet["by_object"][o]["success_safegrip"] or 0) * 100) for o in objs]
    off = [round((fleet["by_object"][o]["success_no_safegrip"] or 0) * 100) for o in objs]
    chart = {
        "type": "bar",
        "data": {"labels": objs, "datasets": [
            {"label": "SafeGrip on", "data": on, "backgroundColor": COLORS[2]},
            {"label": "SafeGrip off", "data": off, "backgroundColor": COLORS[1]},
        ]},
        "options": {"title": {"display": True, "text": title},
                    "scales": {"yAxes": [{"ticks": {"min": 0, "max": 100},
                                          "scaleLabel": {"display": True, "labelString": "success %"}}]}},
    }
    return image("success by object", chart)


def bar_chart(labels: list[str], values: list[float], title: str, ylabel: str) -> str:
    chart = {
        "type": "bar",
        "data": {"labels": labels, "datasets": [{"label": ylabel, "data": values, "backgroundColor": COLORS[0]}]},
        "options": {"title": {"display": True, "text": title}, "legend": {"display": False}},
    }
    return image(title, chart)


def score_chart(rows: list[tuple[str, int]], title: str) -> str:
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [COLORS[2] if v >= 80 else COLORS[4] if v >= 50 else COLORS[1] for v in vals]
    chart = {
        "type": "horizontalBar",
        "data": {"labels": labels, "datasets": [{"data": vals, "backgroundColor": colors}]},
        "options": {"title": {"display": True, "text": title}, "legend": {"display": False},
                    "scales": {"xAxes": [{"ticks": {"min": 0, "max": 100}}]}},
    }
    return image(title, chart, h=60 + 30 * len(rows))


# Generic builders for the Visualizer and Data Explorer agents.

def _opts(title: str, xlabel: str | None = None, ylabel: str | None = None, **extra) -> dict:
    o = {"title": {"display": True, "text": title}}
    scales = {}
    if xlabel:
        scales["xAxes"] = [{"scaleLabel": {"display": True, "labelString": xlabel}, "ticks": {"maxTicksLimit": 12}}]
    if ylabel:
        scales["yAxes"] = [{"scaleLabel": {"display": True, "labelString": ylabel}}]
    if scales:
        o["scales"] = scales
    o.update(extra)
    return o


def line(labels: list, series: dict[str, list[float]], title: str, xlabel: str, ylabel: str,
         dashed: tuple[str, ...] = ()) -> str:
    datasets = [{"label": name, "data": vals, "borderColor": COLORS[i % len(COLORS)], "fill": False,
                 "pointRadius": 0, "borderWidth": 2, **({"borderDash": [6, 4]} if name in dashed else {})}
                for i, (name, vals) in enumerate(series.items())]
    return image(title, {"type": "line", "data": {"labels": labels, "datasets": datasets},
                         "options": _opts(title, xlabel, ylabel)})


def scatter(groups: dict[str, list[tuple[float, float]]], title: str, xlabel: str, ylabel: str) -> str:
    datasets = [{"label": name, "data": [{"x": x, "y": y} for x, y in pts],
                 "backgroundColor": COLORS[i % len(COLORS)], "borderColor": COLORS[i % len(COLORS)],
                 "pointBackgroundColor": COLORS[i % len(COLORS)], "pointRadius": 4}
                for i, (name, pts) in enumerate(groups.items())]
    o = _opts(title)
    o["scales"] = {"xAxes": [{"type": "linear", "scaleLabel": {"display": True, "labelString": xlabel}}],
                   "yAxes": [{"scaleLabel": {"display": True, "labelString": ylabel}}]}
    return image(title, {"type": "scatter", "data": {"datasets": datasets}, "options": o})


def pie(labels: list[str], values: list[float], title: str) -> str:
    return image(title, {"type": "doughnut", "data": {"labels": labels, "datasets": [
        {"data": values, "backgroundColor": [COLORS[i % len(COLORS)] for i in range(len(values))]}]},
        "options": {"title": {"display": True, "text": title},
                    "plugins": {"datalabels": {"color": "#fff", "font": {"weight": "bold"}}}}}, w=560, h=380)


def radar(axes: list[str], series: dict[str, list[float]], title: str) -> str:
    datasets = [{"label": name, "data": vals, "borderColor": COLORS[i % len(COLORS)],
                 "backgroundColor": COLORS[i % len(COLORS)] + "33", "pointRadius": 3}
                for i, (name, vals) in enumerate(series.items())]
    return image(title, {"type": "radar", "data": {"labels": axes, "datasets": datasets},
                         "options": {"title": {"display": True, "text": title},
                                     "scale": {"ticks": {"min": 0, "max": 100, "stepSize": 25,
                                                         "backdropColor": "rgba(0,0,0,0)"}}}}, w=560, h=440)


def histogram(values: list[float], bins: int, title: str, xlabel: str) -> str:
    lo, hi = min(values), max(values)
    width = (hi - lo) / bins or 1.0
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int((v - lo) / width))] += 1
    fmt = "{:.0f}" if width >= 1 else "{:.2f}"
    labels = [f"{fmt.format(lo + i * width)}–{fmt.format(lo + (i + 1) * width)}" for i in range(bins)]
    return image(title, {"type": "bar", "data": {"labels": labels, "datasets": [
        {"label": "episodes", "data": counts, "backgroundColor": COLORS[0]}]},
        "options": _opts(title, xlabel, "episodes", legend={"display": False})})


def stacked_bar(labels: list[str], series: dict[str, list[float]], title: str, ylabel: str) -> str:
    datasets = [{"label": name, "data": vals, "backgroundColor": COLORS[i % len(COLORS)]}
                for i, (name, vals) in enumerate(series.items())]
    o = _opts(title)
    o["scales"] = {"xAxes": [{"stacked": True}],
                   "yAxes": [{"stacked": True, "scaleLabel": {"display": True, "labelString": ylabel}}]}
    return image(title, {"type": "bar", "data": {"labels": labels, "datasets": datasets}, "options": o})


def grouped_bar(labels: list[str], series: dict[str, list[float]], title: str, ylabel: str) -> str:
    datasets = [{"label": name, "data": vals, "backgroundColor": COLORS[i % len(COLORS)]}
                for i, (name, vals) in enumerate(series.items())]
    return image(title, {"type": "bar", "data": {"labels": labels, "datasets": datasets},
                         "options": _opts(title, None, ylabel)})
