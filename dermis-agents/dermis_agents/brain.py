"""The agent loop: an ASI:One LLM that holds the conversation, plans, and calls tools.

Tools cover the Dermis recordings, any dataset the user brings (URL or pasted), web pages, the live
SpacetimeDB, other agents, interactive cards and the paid report. Every number comes from a tool, never
from the model. Charts reach the model as short [[chart:N]] tokens and are swapped back in afterwards,
because the QuickChart URLs are thousands of characters. If the LLM is unavailable the caller falls
back to the deterministic report, so an agent always answers.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Awaitable, Callable

from . import analysis, cards, charts, connectors, dataset, llm, payments, specialist, viz
from .profiles import BY_KEY, PROFILES, Profile
from .store import DataUnavailable, snapshot

MAX_ROUNDS = 8
HISTORY_TURNS = 8
TOOL_RESULT_CHARS = 6000
IMAGE_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)]+)\)")
APPS = [p.key for p in PROFILES if p.kind == "touch"]

# Agent-to-agent call the hub can make: (agent key, question) -> markdown answer.
AskAgent = Callable[[str, str], Awaitable[str]]


class Charts:
    """Holds chart images out of the model's context and puts them back into the final answer."""

    def __init__(self):
        self.images: list[str] = []

    def stash(self, md: str) -> str:
        def repl(m: re.Match) -> str:
            self.images.append(m.group(0))
            return f"[[chart:{len(self.images)}]]"
        return IMAGE_RE.sub(repl, md)

    def restore(self, text: str) -> str:
        used = set()

        def repl(m: re.Match) -> str:
            i = int(m.group(1))
            if 1 <= i <= len(self.images):
                used.add(i)
                return self.images[i - 1]
            return ""
        text = re.sub(r"\[\[chart:(\d+)\]\]", repl, text)
        text = re.sub(r"^\s*\[chart\]\s*$", "", text, flags=re.M)  # placeholder echoed from memory
        # If the model placed no chart at all, show the first one it looked at.
        if not used and self.images:
            text += "\n\n" + self.images[0]
        return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------- tools

def _fn(name: str, description: str, props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


EPISODE = {"type": "string", "description": "recording id, e.g. episode_0004 (0-191)"}
OBJECT = {"type": "string", "enum": ["can", "egg", "ball", "cup", "bottle", "sponge"]}
SCENARIO = {"type": "string", "enum": ["nominal", "crush", "light_grip", "drop"]}
APP = {"type": "string", "enum": APPS, "description": "application whose limits to judge against"}

DATA_TOOLS = [
    _fn("list_recordings", "Find recordings. Returns id, object, scenario, SafeGrip, outcome for each match.",
        {"object": OBJECT, "scenario": SCENARIO, "safe_grip": {"type": "boolean"},
         "outcome": {"type": "string", "enum": ["success", "fracture", "slip", "impact", "early_release", "incomplete"]},
         "limit": {"type": "integer", "description": "max rows, default 15"}}),
    _fn("analyze_recording", "Analyze one recording for an application: per-pod force, contact, slip vibration, "
        "sensor saturation and noise, touch latency, faults, findings, readiness score, the SafeGrip on/off twin "
        "outcome, and a force chart.", {"episode": EPISODE, "application": APP}, ["episode"]),
    _fn("fleet_stats", "Success and failure rates across recordings, by object, scenario and SafeGrip, optionally "
        "limited to the objects an application handles.", {"application": APP, "object": OBJECT}),
    _fn("compare_applications", "Score one recording against every application's limits (humanoid, industrial, "
        "quadruped, prosthetic, surgical, warehouse, food, eldercare, teleop, sensor QA).", {"episode": EPISODE},
        ["episode"]),
    _fn("sensor_health", "Scan all recordings for sensor problems: magnetometer saturation, noise vs spec, dropped "
        "frames, touch detection latency."),
    _fn("dataset_overview", "What the dataset contains: size, outcome counts, coverage gaps, and how much touch "
        "improves a baseline model over IMU only."),
    _fn("make_chart", "Draw a chart and get its numbers. Describe it in words, e.g. 'dashboard', 'plot episode 4', "
        "'plot the raw field for episode 4', 'compare episode 4 with safegrip on', 'compare episode 4 vs episode 0', "
        "'histogram of peak force', 'scatter latency vs peak force', 'pie of outcomes for the bottle', "
        "'heatmap of success', 'radar of episode 40'.", {"request": {"type": "string"}}, ["request"]),
    _fn("make_table", "Build a pandas-style table. Describe it, e.g. 'overview', 'success by scenario', "
        "'peak force by object and scenario', 'latency by safe grip', 'stats for episode 4'.",
        {"request": {"type": "string"}}, ["request"]),
]
ASK_TOOL = _fn("ask_agent", "Ask one of the specialist agents (a separate Fetch.ai agent) and get its full answer. "
               "Use it when a question is about a specific application.",
               {"agent": {"type": "string", "enum": [p.key for p in PROFILES]}, "question": {"type": "string"}},
               ["agent", "question"])


def _parsed(args: dict) -> dict:
    return {k: v for k, v in args.items() if k in ("object", "scenario", "safe_grip") and v is not None}


def t_list_recordings(args: dict) -> dict:
    eps = snapshot().episodes
    rows = [e for e in eps
            if (not args.get("object") or e["object"] == args["object"])
            and (not args.get("scenario") or e["scenario"] == args["scenario"])
            and (args.get("safe_grip") is None or bool(e["safe_grip"]) == args["safe_grip"])
            and (not args.get("outcome") or e["outcome"] == args["outcome"])]
    limit = int(args.get("limit") or 15)
    return {"matches": len(rows), "recordings": [
        {k: e[k] for k in ("episode", "object", "scenario", "safe_grip", "outcome", "seconds")} for e in rows[:limit]]}


def t_analyze_recording(args: dict, default_app: str) -> dict:
    snap = snapshot()
    eid = _episode_id(args.get("episode", ""))
    meta = snap.meta(eid)
    if meta is None:
        return {"error": f"no recording {eid}; ids run episode_0000 to episode_{len(snap.episodes) - 1:04d}"}
    p = BY_KEY.get(args.get("application") or default_app) or BY_KEY["humanoid_arm"]
    f = snap.facts(eid)
    findings = analysis.evaluate(p, f)
    twin = analysis.counterfactual(meta, snap.episodes)
    key = "peak_force_gt_n" if f["has_ground_truth_force"] else "peak_force_est_n"
    return {
        "episode": eid, "application": p.key, "object": meta["object"], "mass_g": meta["mass_g"],
        "scenario": meta["scenario"], "safe_grip": meta["safe_grip"], "outcome": meta["outcome"],
        "seconds": f["seconds"], "force_limit_n": p.limit_for(meta["object"]),
        "readiness_score": analysis.score(findings),
        "findings": [{k: x[k] for k in ("severity", "message", "action")} for x in findings],
        "pods": {p.pod_labels[k]: {"peak_force_n": v[key], "contact_s": v["contact_s"], "slip_vibration_s": v["slip_signal_s"],
                                   "rail_frames": v["rail_frames"], "noise_vs_spec": v["noise_vs_spec"]}
                 for k, v in f["pods"].items()},
        "touch_detect_latency_ms": f.get("touch_detect_latency_ms"), "fault": f.get("fault"),
        "fault_at_s": f.get("fault_at_s"), "time_to_hold_s": f.get("time_to_hold_s"),
        "min_friction_margin": f.get("min_friction_margin"), "max_lift_cm": f.get("max_lift_cm"),
        "safegrip_twin": ({"episode": twin["episode"], "safe_grip": twin["safe_grip"], "outcome": twin["outcome"]}
                          if twin else None),
        "force_is": "reference force" if f["has_ground_truth_force"] else "uncalibrated field estimate",
        "proxy_note": p.proxy_note or None,
        "chart": charts.force_chart(f, p.pod_labels, p.limit_for(meta["object"])),
    }


def t_fleet_stats(args: dict) -> dict:
    eps = snapshot().episodes
    p = BY_KEY.get(args.get("application") or "")
    objects = (args["object"],) if args.get("object") else (p.objects if p else ())
    f = analysis.fleet(eps, objects)
    f["chart"] = charts.fleet_chart(f, "Success rate by object")
    return f


def t_compare_applications(args: dict) -> dict:
    snap = snapshot()
    eid = _episode_id(args.get("episode", ""))
    if snap.meta(eid) is None:
        return {"error": f"no recording {eid}"}
    f = snap.facts(eid)
    rows = []
    for p in PROFILES:
        if not p.in_matrix:
            continue
        fs = analysis.evaluate(p, f)
        rows.append({"application": p.title.replace(" Agent", ""), "score": analysis.score(fs),
                     "top_finding": fs[0]["message"] if fs else "no problems", "first_action": fs[0]["action"] if fs else ""})
    return {"episode": eid, "outcome": snap.meta(eid)["outcome"], "applications": rows,
            "chart": charts.score_chart([(r["application"], r["score"]) for r in rows], f"Readiness by application · {eid}")}


def t_sensor_health(args: dict) -> dict:
    rows, src = specialist._sensor_scan()
    railed = [r for r in rows if r["rail"]]
    lat = sorted(r["latency"] for r in rows if r["latency"] is not None)
    return {"recordings": len(rows), "frames": sum(r["frames"] for r in rows),
            "railed_recordings": len(railed), "noisy_recordings": sum(r["noise"] > 3 for r in rows),
            "recordings_with_dropped_frames": sum(bool(r["dropped"]) for r in rows),
            "latency_ms_median": lat[len(lat) // 2] if lat else None, "latency_ms_worst": lat[-1] if lat else None,
            "worst_railed": [{"episode": r["episode"], "object": r["object"], "railed_frames": r["rail"],
                              "also_grip_fault": r["fault"]} for r in sorted(railed, key=lambda r: -r["rail"])[:5]],
            "calibration": "field-to-force is an assumed 1.5 mT/N until load-cell calibration"}


def t_dataset_overview(args: dict) -> dict:
    snap = snapshot()
    f = analysis.fleet(snap.episodes)
    out = {"recordings": len(snap.episodes), "frames": sum(e.get("frames") or 0 for e in snap.episodes),
           "minutes": round(sum(e.get("seconds") or 0 for e in snap.episodes) / 60, 1),
           "source": snap.source, "outcomes": f["outcomes"], "by_scenario": f["by_scenario"],
           "hardware_recordings": sum(1 for e in snap.episodes if e.get("source") == "hardware")}
    b = snap.baseline or {}
    try:
        out["baseline_macro_f1"] = {
            "imu_only": {"grasp": b["IMU only"]["session"]["grasp"]["macro_f1"], "fault": b["IMU only"]["session"]["fault"]["macro_f1"]},
            "imu_plus_touch": {"grasp": b["IMU + touch"]["session"]["grasp"]["macro_f1"], "fault": b["IMU + touch"]["session"]["fault"]["macro_f1"]}}
    except KeyError:
        pass
    return out


def _episode_id(s: str) -> str:
    m = re.search(r"(\d{1,4})", s or "")
    return f"episode_{int(m.group(1)):04d}" if m else s


# ---------------------------------------------------------------- any dataset, web, live DB, cards, payment

FILTERS = {"type": "array", "description": "conditions, all must hold",
           "items": {"type": "object", "properties": {
               "column": {"type": "string"}, "op": {"type": "string", "enum": list(dataset.OPS)},
               "value": {"description": "number, text, boolean, or a list for 'in'"}},
               "required": ["column", "op", "value"]}}
METRICS = {"type": "array", "items": {"type": "object", "properties": {
    "column": {"type": "string", "description": "omit for a row count"},
    "agg": {"type": "string", "enum": list(dataset.AGGS)}}, "required": ["agg"]}}

DATASET_TOOLS = [
    _fn("load_dataset", "Load a table to analyze: 'dermis' (one row per recording with computed metrics), a URL "
        "to a CSV/TSV/JSON/JSONL file (GitHub links work), or 'pasted' for the table the user pasted. It becomes "
        "the active dataset for this conversation. Returns columns, types, stats and sample rows.",
        {"source": {"type": "string"}}, ["source"]),
    _fn("describe_dataset", "Columns, types, ranges, top values and sample rows of the active dataset."),
    _fn("query_dataset", "Filter, group, aggregate, sort and list rows of the active dataset (like pandas "
        "query/groupby/agg/sort_values/head).", {"filters": FILTERS, "group_by": {"type": "array", "items": {"type": "string"}},
        "metrics": METRICS, "sort_by": {"type": "string"}, "descending": {"type": "boolean"},
        "limit": {"type": "integer"}, "columns": {"type": "array", "items": {"type": "string"}}}),
    _fn("add_column", "Add a computed column to the active dataset from a formula over existing columns, e.g. "
        "name 'tip_pct', expression 'tip / total_bill * 100'. Allowed: columns, numbers, + - * / // % **, "
        "abs/round/min/max/log/sqrt. The column stays available for the rest of the conversation.",
        {"name": {"type": "string"}, "expression": {"type": "string"}}, ["name", "expression"]),
    _fn("pivot_dataset", "Cross-tab of the active dataset: rows = index column, columns = another column, cells = "
        "an aggregation (count by default).", {"index": {"type": "string"}, "columns": {"type": "string"},
        "value": {"type": "string"}, "agg": {"type": "string", "enum": list(dataset.AGGS)}, "filters": FILTERS},
        ["index", "columns"]),
    _fn("correlate_dataset", "Strongest correlations between numeric columns of the active dataset.",
        {"filters": FILTERS}),
    _fn("chart_dataset", "Chart the active dataset. kind: bar (y aggregated by x), line (y over x), scatter (x vs "
        "y), histogram (distribution of x), pie (share of x). Optional series column splits it into colors.",
        {"kind": {"type": "string", "enum": ["bar", "line", "scatter", "histogram", "pie"]},
         "x": {"type": "string"}, "y": {"type": "string"}, "agg": {"type": "string", "enum": list(dataset.AGGS)},
         "series": {"type": "string"}, "filters": FILTERS, "title": {"type": "string"}, "limit": {"type": "integer"}},
        ["kind", "x"]),
]
WEB_TOOLS = [
    _fn("read_web_page", "Read a public web page, document or API response as text (for facts, specs, papers, "
        "docs the user mentions).", {"url": {"type": "string"}}, ["url"]),
    _fn("query_live_database", "Read-only SQL (SELECT) on the team's live SpacetimeDB of Dermis recordings, for "
        "data captured after the snapshot. Tables: episode, frame.", {"sql": {"type": "string"}}, ["sql"]),
]
FORM_TOOL = _fn("show_start_form", "Show the user an interactive form to pick a robot application, object, "
                "scenario and SafeGrip. Use it when someone wants to start but hasn't said what to look at.")
BUY_TOOL = _fn("offer_certified_report", f"Offer the paid certified readiness report for one recording "
               f"({payments.PRICE_FET} FET). The user gets a review card with a Pay button; payment is checked "
               "on-chain before the report is issued. Use it when the user asks for a certified/full/official "
               "report or certificate.", {"episode": EPISODE}, ["episode"])


class Turn:
    """Everything one user turn accumulates besides the text: charts, a card, an offered purchase."""

    def __init__(self, key: str, session: dict):
        self.key = key
        self.session = session
        self.stash = Charts()
        self.card: dict | None = None
        self.purchase: str | None = None
        self._table: dataset.Table | None = None

    def table(self) -> dataset.Table:
        if self._table is None:
            ds = self.session.get("dataset")
            if not ds:
                raise dataset.DatasetError("no dataset loaded yet: call load_dataset ('dermis', a URL, or 'pasted')")
            src = ds["text"] if ds.get("source") == "pasted" else ds["source"]
            self._table = dataset.load(src, ds.get("name"))
            for name, expr in ds.get("derived", []):  # columns the user added earlier in the conversation
                try:
                    self._table.derive(name, expr)
                except dataset.DatasetError:
                    pass
        return self._table


def t_load_dataset(args: dict, turn: Turn) -> dict:
    src = (args.get("source") or "").strip()
    if src.lower() == "pasted":
        if (turn.session.get("dataset") or {}).get("source") != "pasted":
            return {"error": "the user hasn't pasted a table in this conversation"}
    else:
        turn.session["dataset"] = {"source": "dermis" if src.lower() == "dermis" else src}
    turn._table = None
    t = turn.table()
    turn.session["dataset"]["name"] = t.name
    return t.schema()


def t_add_column(args: dict, turn: Turn) -> dict:
    out = turn.table().derive(args.get("name", ""), args.get("expression", ""))
    ds = turn.session["dataset"]
    ds["derived"] = [d for d in ds.get("derived", []) if d[0] != args["name"]] + [[args["name"], args["expression"]]]
    return out


def t_chart_dataset(args: dict, turn: Turn) -> dict:
    t = turn.table()
    kind, x = args["kind"], t.col(args["x"])
    y = t.col(args["y"]) if args.get("y") else None
    series = t.col(args["series"]) if args.get("series") else None
    rows = t.filtered(args.get("filters"))
    agg = args.get("agg") or ("mean" if y else "count")
    title = args.get("title") or f"{agg if y else 'count'} {y or ''} by {x}".replace("  ", " ")
    limit = int(args.get("limit") or 25)
    if kind == "scatter":
        if not y:
            return {"error": "scatter needs both x and y"}
        groups: dict = {}
        for r in rows[:1500]:
            if isinstance(r[x], (int, float)) and isinstance(r[y], (int, float)):
                groups.setdefault(str(r[series]) if series else y, []).append((r[x], r[y]))
        return {"chart": charts.scatter(groups, title, x, y), "points": sum(map(len, groups.values()))}
    if kind == "histogram":
        vals = [r[x] for r in rows if isinstance(r[x], (int, float)) and not isinstance(r[x], bool)]
        if not vals:
            return {"error": f"{x} has no numeric values"}
        return {"chart": charts.histogram(vals, 12, title, x), "n": len(vals),
                "min": min(vals), "max": max(vals), "median": dataset.statistics.median(vals)}
    res = t.query(filters=args.get("filters"), group_by=[x] + ([series] if series else []),
                  metrics=[{"column": y, "agg": agg}] if y else [{"agg": "count"}], limit=100000)
    mname = dataset._metric_name(y, agg) if y else "count"
    if kind == "pie":
        top = sorted(res["rows"], key=lambda r: -(r[mname] or 0))[:limit]
        return {"chart": charts.pie([str(r[x]) for r in top], [r[mname] or 0 for r in top], title), "rows": top}
    labels = sorted({r[x] for r in res["rows"]}, key=dataset._sortkey)
    if kind == "bar" and not series:
        top = sorted(res["rows"], key=lambda r: -(r[mname] or 0))[:limit]
        return {"chart": charts.bar_chart([str(r[x]) for r in top], [r[mname] or 0 for r in top], title, mname),
                "rows": top}
    labels = labels[:200]
    groups = sorted({r[series] for r in res["rows"]}, key=dataset._sortkey)[:8] if series else [None]
    cell = {(r[x], r.get(series)): r[mname] for r in res["rows"]}
    data = {str(g if g is not None else mname): [cell.get((lab, g)) for lab in labels] for g in groups}
    chart = (charts.line([str(lab) for lab in labels], data, title, x, mname) if kind == "line"
             else charts.grouped_bar([str(lab) for lab in labels], {k: [v or 0 for v in vs] for k, vs in data.items()}, title, mname))
    return {"chart": chart, "rows": res["rows"][:40]}


async def run_tool(name: str, args: dict, default_app: str, turn: Turn, ask: AskAgent | None) -> str:
    try:
        if name == "make_chart":
            text = (await asyncio.to_thread(viz.visualize, args.get("request", ""), _parsed(args)))["markdown"]
        elif name == "make_table":
            text = (await asyncio.to_thread(viz.explore, args.get("request", ""), _parsed(args)))["markdown"]
        elif name == "ask_agent" and ask:
            text = await ask(args.get("agent", ""), args.get("question", ""))
        elif name == "show_start_form":
            turn.card = cards.start_form()
            text = json.dumps({"shown": "a form with application, object, scenario and SafeGrip"})
        elif name == "offer_certified_report":
            eid = _episode_id(args.get("episode", ""))
            if snapshot().meta(eid) is None:
                text = json.dumps({"error": f"no recording {eid}"})
            else:
                turn.purchase = eid
                text = json.dumps({"offered": eid, "price_fet": payments.PRICE_FET, "network": payments.NETWORK,
                                   "next": "the user sees a review card with a Pay button; keep your reply to 1-2 sentences"})
        else:
            sync = {"list_recordings": t_list_recordings, "fleet_stats": t_fleet_stats,
                    "compare_applications": t_compare_applications, "sensor_health": t_sensor_health,
                    "dataset_overview": t_dataset_overview,
                    "read_web_page": lambda a: connectors.read_web_page(a.get("url", "")),
                    "query_live_database": lambda a: connectors.query_live_database(a.get("sql", "")),
                    "load_dataset": lambda a: t_load_dataset(a, turn),
                    "describe_dataset": lambda a: turn.table().schema(),
                    "query_dataset": lambda a: turn.table().query(**{k: a.get(k) for k in (
                        "filters", "group_by", "metrics", "sort_by", "columns") if a.get(k) is not None},
                        descending=a.get("descending", True), limit=a.get("limit") or 20),
                    "pivot_dataset": lambda a: turn.table().pivot(a["index"], a["columns"], a.get("value"),
                                                                  a.get("agg") or "count", a.get("filters")),
                    "correlate_dataset": lambda a: turn.table().correlations(a.get("filters")),
                    "add_column": lambda a: t_add_column(a, turn),
                    "chart_dataset": lambda a: t_chart_dataset(a, turn)}
            if name == "analyze_recording":
                out = await asyncio.to_thread(t_analyze_recording, args, default_app)
                if "error" not in out:
                    turn.card = cards.recording_card(out, payments.PRICE_FET)
            elif name in sync:
                out = await asyncio.to_thread(sync[name], args)
            else:
                return json.dumps({"error": f"unknown tool {name}"})
            text = json.dumps(out, default=str)
    except (DataUnavailable, dataset.DatasetError) as e:
        text = json.dumps({"error": str(e)})
    except Exception as e:  # a broken tool must not kill the conversation; the model can try another way
        text = json.dumps({"error": f"{name} failed: {type(e).__name__}: {e}"})
    text = turn.stash.stash(text)
    return text if len(text) <= TOOL_RESULT_CHARS else text[:TOOL_RESULT_CHARS] + " …(truncated)"


# ---------------------------------------------------------------- personas

def system_prompt(title: str, role: str, priorities: str, extra: str = "") -> str:
    return (
        f"You are {title}, an expert AI agent on Fetch.ai's Agentverse, built by the Dermis team. Dermis makes a "
        "magnetic tactile glove (4 touch pods + IMUs at 350 Hz) that gives robots a sense of touch. "
        f"{role} You care most about: {priorities}.\n\n"
        "You are as capable as a top general assistant: answer general questions (robotics, sensors, ML, data, "
        "engineering, business, anything) well from your own knowledge, and use tools whenever the answer "
        "depends on data, a file, a web page, or another agent.\n\n"
        "Built-in data: the Dermis dataset, 192 grasp episodes of 6 objects (can, egg, ball, cup, bottle, sponge) in "
        "4 scenarios (nominal, crush, light_grip, drop), each with SafeGrip force limiting on or off. Don't add "
        "remarks about where the data comes from; if someone asks directly, answer truthfully that it is the "
        "Dermis simulation dataset in the glove's data format, with hardware sessions being added.\n"
        "Any dataset: users can give a CSV/JSON URL or paste a table; load it with load_dataset and analyze it "
        "with the dataset tools like a pandas expert (describe, filter, group, pivot, correlate, chart).\n\n"
        "How to work:\n"
        "- Plan multi-step requests, then call the tools you need (several at once when independent). Check "
        "results; if a tool errors, fix the arguments or try another route instead of giving up.\n"
        "- Every number about data must come from a tool result. Never invent figures. Your own general "
        "knowledge is fine for explanations, but keep it clearly separate from measured results.\n"
        "- Ask a clarifying question only when no sensible default exists; otherwise pick one and say which.\n"
        "- Tool results may contain tokens like [[chart:3]]. Place the useful ones on their own line where the "
        "chart belongs; never write image URLs yourself.\n"
        "- Lead with the answer in 1-3 sentences, then evidence (tight bullets or a small markdown table), then "
        "1-3 concrete next steps. Match length to the question; no filler.\n"
        "- You remember this conversation: resolve 'it', 'that one', 'the egg' from earlier turns.\n"
        "- A message that is a JSON object with an \"action\" key is a button press on a card you showed: "
        "analyze → find the matching recording (list_recordings) and analyze_recording it for that application; "
        "compare → compare_applications; safegrip_compare → make_chart 'compare <episode> with safegrip'; "
        "buy_report → offer_certified_report if you can, otherwise tell them the Dermis Hub sells it."
        + (f"\n\n{extra}" if extra else ""))


def persona(key: str) -> tuple[str, str]:
    """(system prompt, default application) for an agent key."""
    if key == "hub":
        return system_prompt(
            "Dermis Hub", "You are the front door and orchestrator: answer directly with your tools, bring in "
                          "specialist agents with ask_agent when an application's expertise matters (ask several "
                          "and combine), and sell the certified readiness report when someone wants a formal result.",
            "getting people a correct, actionable answer fast; picking the right specialist; comparing applications",
            f"When the user doesn't name an application, judge recordings as a humanoid arm and say so. When "
            f"someone is new or says hi, briefly say what you can do and use show_start_form. "
            f"The certified report costs {payments.PRICE_FET} FET ({payments.NETWORK}). "
            "Specialist agents: " + "; ".join(f"{p.key} = {p.title}" for p in PROFILES)), "humanoid_arm"
    p = BY_KEY[key]
    if p.kind == "visualizer":
        role = ("You turn questions into the right chart, for the Dermis recordings or any dataset the user brings, "
                "then explain what the chart shows.")
    elif p.kind == "explorer":
        role = ("You are a data analyst for any dataset: load what the user gives you (URL or pasted table, or the "
                "Dermis recordings), explore it like a pandas expert, and explain findings with tables and charts.")
    else:
        role = f"You analyze touch data for {p.domain}, judging it against that application's limits."
    return system_prompt(p.title, role, ", ".join(p.priorities), p.proxy_note), (
        p.key if p.kind == "touch" else "humanoid_arm")


def tools_for(key: str, ask: AskAgent | None) -> list[dict]:
    tools = DATA_TOOLS + DATASET_TOOLS + WEB_TOOLS + [FORM_TOOL]
    if key == "hub":
        tools.append(BUY_TOOL)
    if ask:
        tools.append(ASK_TOOL)
    return tools


# ---------------------------------------------------------------- the loop

def attach_inline_dataset(text: str, session: dict) -> str:
    """If the message carries a dataset link or a pasted table, load it and tell the model about it."""
    url = dataset.find_url(text)
    if url:
        session["dataset"] = {"source": url}
        return text + f"\n\n[The user linked a dataset: {url}. It is now the active dataset; use the dataset tools.]"
    lines = text.strip().splitlines()
    for i in range(0, min(6, max(0, len(lines) - 2))):
        block = "\n".join(lines[i:])
        if dataset.looks_like_table(block):
            try:
                t = dataset.parse(block, "pasted data")
            except dataset.DatasetError:
                return text
            session["dataset"] = {"source": "pasted", "name": "pasted data", "text": block[:200_000]}
            question = "\n".join(lines[:i]).strip() or "Describe this dataset and what's interesting in it."
            return (question + f"\n\n[The user pasted a table: {len(t.rows)} rows, columns {', '.join(t.columns[:30])}. "
                               "It is now the active dataset; use the dataset tools.]")
    return text


async def converse(key: str, text: str, history: list[dict] | None = None,
                   ask: AskAgent | None = None, session: dict | None = None) -> dict | None:
    """One user turn → {"text", "card", "purchase"}; None if the LLM is unavailable (caller falls back).
    `session` is the per-user dict the caller persists (active dataset)."""
    session = session if session is not None else {}
    system, default_app = persona(key)
    turn = Turn(key, session)
    user = attach_inline_dataset(text, session)
    messages = [{"role": "system", "content": system}, *(history or [])[-2 * HISTORY_TURNS:],
                {"role": "user", "content": user}]
    tools = tools_for(key, ask)
    answer = None
    narration: list[str] = []  # text the model wrote alongside its tool calls (often its intro)
    for _ in range(MAX_ROUNDS):
        msg = await llm.chat(messages, tools)
        if msg is None:
            return None
        calls = msg.get("tool_calls") or []
        if not calls:
            answer = (msg.get("content") or "").strip()
            break
        if (msg.get("content") or "").strip():
            narration.append(msg["content"].strip())
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        results = await asyncio.gather(*(run_tool(c["function"]["name"], _args(c), default_app, turn, ask) for c in calls))
        for c, res in zip(calls, results):
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": res})
    else:
        messages.append({"role": "user", "content": "Answer now with what you have."})
        answer = ((await llm.chat(messages, [])) or {}).get("content")
    if not answer:
        return None
    intro = "\n\n".join(n for n in narration if n not in answer)
    text = f"{intro}\n\n{answer}" if intro and len(answer) < 600 else answer
    return {"text": turn.stash.restore(text), "card": turn.card, "purchase": turn.purchase}


def _args(call: dict) -> dict:
    try:
        a = json.loads(call["function"].get("arguments") or "{}")
        return a if isinstance(a, dict) else {}
    except json.JSONDecodeError:
        return {}


def remember(history: list[dict], user: str, answer: str) -> list[dict]:
    """Keep the last turns as plain text (no tool traces, no chart URLs) so storage stays small."""
    short = IMAGE_RE.sub("", answer)
    return (history + [{"role": "user", "content": user[:1500]},
                       {"role": "assistant", "content": short[:2500]}])[-2 * HISTORY_TURNS:]
