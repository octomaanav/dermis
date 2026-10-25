"""What a specialist agent does with a request. Pure async functions, no uAgents, so it's testable.

respond() returns {"markdown", "score", "headline", "episode"}; the agent layer wraps it in messages.
"""

from __future__ import annotations

import asyncio

from . import analysis, charts, llm
from .profiles import Profile
from .store import DataUnavailable, snapshot

ICON = {"critical": "🔴", "warning": "🟡", "info": "🔵"}


async def respond(profile: Profile, text: str, parsed: dict | None = None, brief: bool = False) -> dict:
    parsed = parsed if parsed is not None else await llm.parse_request(text)
    if profile.kind != "touch":
        from . import viz
        return await viz.respond(profile, text, parsed)
    mode = parsed.get("mode")
    try:
        if mode == "help":
            return {"markdown": help_text(profile), "score": None, "headline": "help", "episode": None}
        wants_episode = parsed.get("episode") or parsed.get("object") or parsed.get("scenario")
        if mode == "fleet" or (profile.fleet_first and not wants_episode and mode != "episode"):
            return await asyncio.to_thread(fleet_report, profile)
        return await episode_report(profile, text, parsed, brief)
    except DataUnavailable as e:
        return {"markdown": f"**{profile.title}**: I couldn't load data ({e}).",
                "score": None, "headline": f"data unavailable: {e}", "episode": None}


def help_text(p: Profile) -> str:
    ex = "\n".join(f"- “{e}”" for e in p.examples)
    note = f"\n\n_{p.proxy_note}_" if p.proxy_note else ""
    return (f"## {p.title}\nI analyze Dermis touch-sensor recordings for **{p.domain}**.\n\n"
            f"I focus on: {', '.join(p.priorities)}.\n\nTry:\n{ex}{note}")


def _data_line(src_name: str) -> str:
    return f"Data: {src_name}"


async def episode_report(p: Profile, text: str, parsed: dict, brief: bool) -> dict:
    snap = await asyncio.to_thread(snapshot)
    episodes, src_name = snap.episodes, snap.label
    if parsed.get("episode"):
        meta = next((e for e in episodes if e["episode"] == parsed["episode"]), None)
        if meta is None:
            return {"markdown": f"I don't have **{parsed['episode']}**. There are {len(episodes)} episodes "
                                f"({episodes[0]['episode']} to {episodes[-1]['episode']}).",
                    "score": None, "headline": "episode not found", "episode": parsed["episode"]}
    else:
        meta = analysis.pick_episode(p, episodes, parsed.get("object"), parsed.get("scenario"),
                                     parsed.get("safe_grip"))
        if meta is None:
            return {"markdown": "No recording matches that object/scenario combination.",
                    "score": None, "headline": "no match", "episode": None}

    facts = snap.facts(meta["episode"])
    findings = analysis.evaluate(p, facts)
    score = analysis.score(findings)
    limit = p.limit_for(facts.get("object") or "")
    top = findings[0] if findings else None
    headline = top["message"] if top else "No problems found."

    if brief:
        return {"markdown": "", "score": score, "headline": headline, "episode": meta["episode"],
                "action": top["action"] if top else ""}

    story = await llm.interpret(p.domain, p.priorities, text, facts, findings) or _story(p, facts, findings)

    sg = "on" if facts["safe_grip"] else "off"
    lines = [
        f"## {p.title} · {facts['episode']}",
        f"**{facts['object']}** ({facts['mass_g']} g, μ {facts['mu']}) · scenario `{facts['scenario']}` · "
        f"SafeGrip {sg} · outcome **{facts['outcome']}** · {facts['seconds']} s, {facts['frames']:,} frames "
        f"@ {facts['rate_hz']:g} Hz",
        f"_{_data_line(src_name)}_",
        "",
        f"**Readiness for {p.title.replace(' Agent', '')}: {score}/100**",
        "",
        charts.force_chart(facts, p.pod_labels, limit),
        "",
        "### What happened",
        story,
        "",
        "### Findings",
    ]
    lines += [f"- {ICON[f['severity']]} {f['message']}" for f in findings] or ["- 🟢 No problems found."]

    force_key = "peak_force_gt_n" if facts["has_ground_truth_force"] else "peak_force_est_n"
    lines += ["", "### Per pod",
              f"| {'Foot' if p.key == 'quadruped' else 'Pod'} | Peak force (N) | Contact (s) | Slip band (s) | Rail frames | Noise vs spec | Force trace |",
              "|---|---|---|---|---|---|---|"]
    for pod, v in facts["pods"].items():
        lines.append(f"| {p.pod_labels[pod]} | {v[force_key]} | {v['contact_s']} | {v['slip_signal_s']} | "
                     f"{v['rail_frames']} | {v['noise_vs_spec']}x | `{charts.sparkline(facts['_series']['force'].get(pod, []), 24)}` |")
    extra = []
    if facts.get("touch_detect_latency_ms") is not None:
        extra.append(f"touch detected {facts['touch_detect_latency_ms']:+.0f} ms vs true contact")
    if facts.get("time_to_hold_s") is not None:
        extra.append(f"stable hold at {facts['time_to_hold_s']} s")
    if facts.get("min_friction_margin") is not None:
        extra.append(f"min friction margin {facts['min_friction_margin']}")
    if facts.get("max_lift_cm") is not None:
        extra.append(f"max lift {facts['max_lift_cm']} cm")
    if extra:
        lines += ["", " · ".join(extra)]

    twin = analysis.counterfactual(meta, episodes)
    if twin:
        lines += ["", f"### What-if: SafeGrip {'on' if twin['safe_grip'] else 'off'}",
                  f"Same object, scenario and seed with SafeGrip {'on' if twin['safe_grip'] else 'off'} "
                  f"→ **{twin['outcome']}** ({twin['episode']})."]

    actions = []
    for f in findings:
        if f["action"] and f["action"] not in actions:
            actions.append(f["action"])
    if twin and twin["outcome"] == "success" and facts["outcome"] != "success" and not facts["safe_grip"]:
        actions.insert(0, f"Turn SafeGrip on: the identical run with it enabled succeeded ({twin['episode']}).")
    lines += ["", "### Recommended next steps"] + [f"{i}. {a}" for i, a in enumerate(actions[:5], 1)] \
        if actions else ["", "### Recommended next steps", "1. Nothing blocking. Move to the next scenario or real hardware runs."]
    if p.proxy_note:
        lines += [f"_{p.proxy_note}_"]
    return {"markdown": "\n".join(lines), "score": score, "headline": headline, "episode": meta["episode"],
            "action": actions[0] if actions else ""}


def _story(p: Profile, facts: dict, findings: list[dict]) -> str:
    pod, peak = analysis.peak_force(facts)
    s = (f"The {facts['object']} was handled for {facts['seconds']} s with peak force {peak:.1f} N on the "
         f"{p.pod_labels.get(pod, pod)}. ")
    if facts.get("fault"):
        s += f"It ended in **{facts['fault'].replace('_', ' ')}** at {facts['fault_at_s']} s. "
    else:
        s += "No fault occurred. "
    crit = [f for f in findings if f["severity"] == "critical"]
    s += f"{len(crit)} critical and {len(findings) - len(crit)} other findings for {p.domain}."
    return s


def fleet_report(p: Profile) -> dict:
    if p.key == "sensor_qa":
        return sensor_fleet_report(p)
    if p.key == "data_curator":
        return curator_report(p)
    snap = snapshot()
    episodes, src_name = snap.episodes, snap.label
    f = analysis.fleet(episodes, p.objects)
    lines = [f"## {p.title} · fleet view",
             f"{f['episodes']} recordings on {', '.join(f['by_object'])} · _{_data_line(src_name)}_", "",
             charts.fleet_chart(f, f"{p.title}: success rate by object"), "",
             "| Object | Runs | Success, SafeGrip on | Success, SafeGrip off | Failures |", "|---|---|---|---|---|"]
    pct = lambda x: "–" if x is None else f"{x * 100:.0f}%"  # noqa: E731
    for obj, r in f["by_object"].items():
        fails = ", ".join(f"{k} {v}" for k, v in sorted(r["failures"].items(), key=lambda kv: -kv[1])) or "none"
        lines.append(f"| {obj} | {r['n']} | {pct(r['success_safegrip'])} | {pct(r['success_no_safegrip'])} | {fails} |")
    lines += ["", "**By scenario:** " + " · ".join(f"`{s}` {pct(v['success'])}" for s, v in f["by_scenario"].items())]

    worst_obj = min(f["by_object"].items(), key=lambda kv: kv[1]["success_safegrip"] or 0)
    worst_scen = min(f["by_scenario"].items(), key=lambda kv: kv[1]["success"] or 0)
    crit_fails = {k: v for k, v in f["outcomes"].items() if k in p.critical_faults}
    total = f["episodes"] or 1
    lines += ["", "### What this means for " + p.domain.split(",")[0]]
    if crit_fails:
        lines.append("- Faults this domain can't accept: " +
                     ", ".join(f"{k} in {v} of {total} runs ({v / total * 100:.0f}%)" for k, v in crit_fails.items()) + ".")
    gains = [(o, r) for o, r in f["by_object"].items()
             if r["success_safegrip"] is not None and r["success_no_safegrip"] is not None
             and r["success_safegrip"] - r["success_no_safegrip"] >= 0.1]
    if gains:
        lines.append("- SafeGrip helps on " + ", ".join(
            f"{o} ({pct(r['success_no_safegrip'])} → {pct(r['success_safegrip'])})" for o, r in gains) + ".")
    flat = [o for o, r in f["by_object"].items() if r["success_safegrip"] == r["success_no_safegrip"]]
    if flat:
        lines.append(f"- SafeGrip makes no difference on {', '.join(flat)}: the failures there aren't force-limit problems.")
    lines += [f"- Weakest object even with SafeGrip: **{worst_obj[0]}** ({pct(worst_obj[1]['success_safegrip'])}). "
              f"Weakest scenario: **{worst_scen[0]}** ({pct(worst_scen[1]['success'])}).",
              "", "### Recommended next steps",
              f"1. Drill into the worst case: ask me to analyze the {worst_obj[0]} in the `{worst_scen[0]}` scenario.",
              "2. " + (p.action("fault_" + max(crit_fails, key=crit_fails.get)) if crit_fails else
                       "Re-run the weakest scenario on real hardware to confirm the sim result."),
              "3. Record the same scenarios on the real glove so these rates come from hardware, not the sim."]
    if p.proxy_note:
        lines.append(f"_{p.proxy_note}_")
    head = f"{pct(worst_obj[1]['success_safegrip'])} success on {worst_obj[0]} with SafeGrip"
    return {"markdown": "\n".join(lines), "score": None, "headline": head, "episode": None}


def _sensor_scan() -> tuple[list[dict], str]:
    snap = snapshot()
    rows = []
    for e in snap.episodes:
        f = snap.facts(e["episode"])
        rows.append({
            "episode": e["episode"], "object": e["object"], "scenario": e["scenario"], "frames": f["frames"],
            "rail": sum(v["rail_frames"] for v in f["pods"].values()),
            "noise": max(v["noise_vs_spec"] for v in f["pods"].values()),
            "dropped": f["dropped_frames"],
            "latency": f.get("touch_detect_latency_ms"),
            "fault": f.get("fault"),
            "worst_pod": max(f["pods"], key=lambda k: f["pods"][k]["rail_frames"]),
        })
    return rows, snap.label


def sensor_fleet_report(p: Profile) -> dict:
    rows, src_name = _sensor_scan()
    frames = sum(r["frames"] for r in rows)
    railed = [r for r in rows if r["rail"]]
    noisy = [r for r in rows if r["noise"] > 3]
    dropped = [r for r in rows if r["dropped"]]
    lat = sorted(r["latency"] for r in rows if r["latency"] is not None)
    rail_with_fault = [r for r in railed if r["fault"]]
    lines = [f"## {p.title} · fleet scan",
             f"Scanned **{len(rows)} recordings, {frames:,} frames** · _{_data_line(src_name)}_", "",
             "| Check | Recordings affected |", "|---|---|",
             f"| Magnetometer rail (saturation) | {len(railed)} |",
             f"| Noise > 3x spec | {len(noisy)} |",
             f"| Dropped frames | {len(dropped)} |"]
    if lat:
        lines.append(f"| Touch detection latency (median / worst) | {lat[len(lat) // 2]:.0f} ms / {lat[-1]:.0f} ms |")
    if railed:
        by_obj: dict = {}
        for r in railed:
            by_obj[r["object"]] = by_obj.get(r["object"], 0) + 1
        lines += ["", charts.bar_chart(list(by_obj), list(by_obj.values()), "Recordings with railed magnetometer, by object", "recordings")]
        lines += ["", "### Sensor fault or grip failure?",
                  f"{len(rail_with_fault)} of the {len(railed)} railed recordings also end in a grip fault. "
                  "In those, the force peak is clipped, so the measured peak understates the real force: "
                  "treat them as sensor-limited, not as a clean measurement of the failure."]
        worst = sorted(railed, key=lambda r: -r["rail"])[:5]
        lines += ["", "Worst recordings: " + ", ".join(f"{r['episode']} ({r['object']}, {r['rail']} railed frames)" for r in worst)]
    lines += ["", "### Recommended next steps"]
    steps = []
    if railed:
        steps.append(p.action("rail"))
    if noisy:
        steps.append(p.action("noisy"))
    if dropped:
        steps.append(p.action("frame_drops"))
    steps.append("Calibrate field-to-force against a load cell (FORCE_REF): force is still an assumed 1.5 mT/N.")
    lines += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
    head = f"{len(railed)} of {len(rows)} recordings railed, {len(noisy)} noisy"
    return {"markdown": "\n".join(lines), "score": None, "headline": head, "episode": None}


def curator_report(p: Profile) -> dict:
    snap = snapshot()
    episodes, src_name = snap.episodes, snap.label
    f = analysis.fleet(episodes)
    frames = sum(e.get("frames") or 0 for e in episodes)
    minutes = sum(e.get("seconds") or 0 for e in episodes) / 60
    outcomes = sorted(f["outcomes"].items(), key=lambda kv: -kv[1])
    lines = [f"## {p.title}",
             f"**{len(episodes)} episodes, {frames:,} frames, {minutes:.1f} min** · _{_data_line(src_name)}_", "",
             charts.bar_chart([k for k, _ in outcomes], [v for _, v in outcomes], "Episodes by outcome", "episodes"), "",
             "| Outcome | Episodes | Share |", "|---|---|---|"]
    lines += [f"| {k} | {v} | {v / len(episodes) * 100:.0f}% |" for k, v in outcomes]

    rare = [(k, v) for k, v in outcomes if k != "success" and v < 0.1 * len(episodes)]
    combos: dict = {}
    for e in episodes:
        combos.setdefault((e["object"], e["scenario"]), []).append(e["outcome"])
    lines += ["", "### Coverage gaps"]
    if rare:
        lines.append("- Under-represented failure classes: " + ", ".join(f"**{k}** ({v})" for k, v in rare) +
                     ". A detector for these will overfit.")
    slip_objs = sorted({o for (o, _), outs in combos.items() if "slip" in outs})
    if slip_objs:
        lines.append(f"- Slip only appears on: {', '.join(slip_objs)}. Collect slip on other objects so a "
                     "slip model doesn't learn the object instead of the slip.")

    if snap.baseline:
        b = snap.baseline
        imu, touch = b.get("IMU only", {}), b.get("IMU + touch", {})
        try:
            lines += ["", "### What touch adds (baseline, held-out sessions)",
                      "| Features | Grasp-state F1 | Fault-warning F1 |", "|---|---|---|",
                      f"| IMU only | {imu['session']['grasp']['macro_f1']:.3f} | {imu['session']['fault']['macro_f1']:.3f} |",
                      f"| IMU + touch | {touch['session']['grasp']['macro_f1']:.3f} | {touch['session']['fault']['macro_f1']:.3f} |",
                      "", "Touch improves grasp-state recognition; fault warning is near chance with a linear model."]
        except KeyError:
            pass
    lines += ["", "### What to record next"]
    steps = [f"Record more **{k}** episodes ({v} today)." for k, v in rare[:2]]
    steps += ["Split train/test by session or object, never by random frames (the baseline does this).",
              "Train a temporal model (GRU or temporal CNN) for fault warning; the linear baseline can't do it.",
              "Push new glove sessions into SpacetimeDB so every agent picks them up."]
    lines += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
    head = f"{len(episodes)} episodes; rare classes: {', '.join(k for k, _ in rare) or 'none'}"
    return {"markdown": "\n".join(lines), "score": None, "headline": head, "episode": None}


def certified_report(episode: str, receipt: str) -> str:
    """The paid product: one recording judged for every application, with evidence and a fix list."""
    import json as _json
    from datetime import datetime, timezone

    from .profiles import PROFILES

    snap = snapshot()
    meta = snap.meta(episode)
    if meta is None:
        return f"I couldn't find {episode} to certify. Your payment receipt: {receipt}"
    f = snap.facts(episode)
    rows, verdicts, fixes = [], {}, []
    for p in PROFILES:
        if not p.in_matrix:
            continue
        fs = analysis.evaluate(p, f)
        s = analysis.score(fs)
        verdict = "PASS" if s >= 80 else "CONDITIONAL" if s >= 50 else "FAIL"
        verdicts[p.key] = {"score": s, "verdict": verdict}
        crit = [x for x in fs if x["severity"] == "critical"]
        rows.append(f"| {p.title.replace(' Agent', '')} | **{s}** | {verdict} | "
                    f"{(crit or fs or [{'message': 'no issues'}])[0]['message']} |")
        for x in fs:
            if x["action"] and x["action"] not in fixes and x["severity"] != "info":
                fixes.append(x["action"])
    twin = analysis.counterfactual(meta, snap.episodes)
    obj_fleet = analysis.fleet(snap.episodes, (meta["object"],))["by_object"].get(meta["object"], {})
    pod, peak = analysis.peak_force(f)
    passes = sum(v["verdict"] == "PASS" for v in verdicts.values())
    issued = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summary = {"episode": episode, "object": meta["object"], "scenario": meta["scenario"],
               "safe_grip": meta["safe_grip"], "outcome": meta["outcome"], "verdicts": verdicts,
               "peak_force_n": peak, "issued": issued, "receipt": receipt, "data": snap.label}
    lines = [
        f"## Dermis certified readiness report · {episode}",
        f"**{meta['object']}** ({meta['mass_g']} g) · `{meta['scenario']}` · SafeGrip {'on' if meta['safe_grip'] else 'off'} · "
        f"outcome **{meta['outcome']}** · issued {issued}",
        "",
        f"**Passes {passes} of {len(verdicts)} applications.** Peak force {peak:.1f} N on the "
        f"{pod.replace('_', ' ') if pod else 'n/a'}; "
        + (f"fault: {f['fault'].replace('_', ' ')} at {f['fault_at_s']} s." if f.get("fault") else "no fault."),
        "",
        charts.score_chart([(p.title.replace(" Agent", ""), verdicts[p.key]["score"]) for p in PROFILES if p.in_matrix],
                           f"Readiness by application · {episode}"),
        "",
        "| Application | Score | Verdict | Deciding finding |", "|---|---|---|---|", *rows, "",
        "### Evidence",
        f"- Touch detected {f.get('touch_detect_latency_ms')} ms vs true contact; stable hold at {f.get('time_to_hold_s')} s; "
        f"min friction margin {f.get('min_friction_margin')}.",
        f"- Sensor health: {sum(v['rail_frames'] for v in f['pods'].values())} saturated frames, worst noise "
        f"{max(v['noise_vs_spec'] for v in f['pods'].values())}x spec, {f['dropped_frames']} dropped frames.",
    ]
    if twin:
        lines.append(f"- SafeGrip what-if: the same run with SafeGrip {'on' if twin['safe_grip'] else 'off'} "
                     f"({twin['episode']}) ended in **{twin['outcome']}**.")
    if obj_fleet:
        sg = obj_fleet.get("success_safegrip")
        nosg = obj_fleet.get("success_no_safegrip")
        lines.append(f"- Fleet context for {meta['object']}: {obj_fleet['n']} runs, success "
                     f"{'–' if sg is None else f'{sg * 100:.0f}%'} with SafeGrip vs "
                     f"{'–' if nosg is None else f'{nosg * 100:.0f}%'} without.")
    lines += ["", "### Fix list, in priority order"] + [f"{i}. {x}" for i, x in enumerate(fixes[:6], 1)] or ["1. None needed."]
    lines += ["", f"_Receipt: {receipt}. Data: {snap.label}._", "",
              "```json", _json.dumps(summary, indent=1), "```"]
    return "\n".join(lines)
