"""Dermis Hub: decides which specialists answer, asks them, and combines what comes back.

`ask(key, text, parsed, brief)` is injected: over uAgents it messages the specialist agent; in tests
it calls specialist.respond directly.
"""

from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable

import dermis_charts as charts, dermis_llm as llm
from dermis_profiles import BY_KEY, PROFILES

Ask = Callable[[str, str, dict, bool], Awaitable[dict]]

ALL_RE = re.compile(r"\b(all|every|each)\b.*\b(agents?|applications?|industr(y|ies)|domains?|robots?|use cases?)\b"
                    r"|\bcompare (across|every|all)\b|\bacross (applications|industries|domains)\b")


def keyword_route(text: str) -> list[str]:
    t = text.lower()
    scores = []
    for p in PROFILES:
        s = sum(1 for k in p.keywords if re.search(rf"\b{re.escape(k)}", t))
        if s and p.kind != "touch":
            s += 2  # asking for a chart or a table is the intent; domain words are what to chart
        if s:
            scores.append((s, p.key))
    scores.sort(key=lambda x: -x[0])
    return [k for _, k in scores[:3]]


async def choose(text: str) -> list[str]:
    if ALL_RE.search(text.lower()):
        return ["all"]
    keys = keyword_route(text)
    if not keys:
        catalog = [{"key": p.key, "domain": p.domain} for p in PROFILES]
        keys = [k for k in await llm.route(text, catalog) if k in BY_KEY or k == "all"]
    return keys or ["humanoid_arm"]


def help_text() -> str:
    rows = "\n".join(f"| **{p.title}** | {p.domain} | “{p.examples[0]}” |" for p in PROFILES)
    return ("## Dermis Hub\nI route your question to touch-sensing specialists. Each one reads the same Dermis "
            "recordings (SpacetimeDB, with a local fallback) through the needs of one application.\n\n"
            "| Agent | Domain | Try |\n|---|---|---|\n" + rows +
            "\n\nAsk about an episode (“analyze episode 4”), an object (“the egg”), a fleet view "
            "(“which objects fail most?”), or compare every application at once "
            "(“compare all applications on episode 4”).")


async def respond(text: str, ask: Ask) -> str:
    parsed = await llm.parse_request(text)
    if parsed.get("mode") == "help" and not keyword_route(text):
        return help_text()
    keys = await choose(text)

    if keys == ["all"]:
        return await matrix(text, parsed, ask)

    results = await asyncio.gather(*(ask(k, text, parsed, False) for k in keys), return_exceptions=True)
    parts = []
    for k, r in zip(keys, results):
        if isinstance(r, Exception):
            parts.append(f"**{BY_KEY[k].title}** failed: {r}")
        else:
            parts.append(r["markdown"])
    routed = ", ".join(BY_KEY[k].title for k in keys)
    return f"_Dermis Hub routed this to: {routed}_\n\n" + "\n\n---\n\n".join(parts)


async def matrix(text: str, parsed: dict, ask: Ask) -> str:
    """One recording, every application's verdict."""
    if not parsed.get("episode"):
        # Pin one episode so every specialist judges the same recording.
        first = await ask("humanoid_arm", text, {**parsed, "mode": "episode"}, True)
        if not first.get("episode"):
            return first.get("markdown") or "I couldn't find a recording to compare on."
        parsed = {**parsed, "episode": first["episode"]}
    parsed = {**parsed, "mode": "episode"}
    judges = [p for p in PROFILES if p.in_matrix]
    results = await asyncio.gather(*(ask(p.key, text, parsed, True) for p in judges), return_exceptions=True)
    rows, chart_rows = [], []
    for p, r in zip(judges, results):
        if isinstance(r, Exception) or r.get("score") is None:
            msg = str(r) if isinstance(r, Exception) else r.get("headline", "")
            rows.append(f"| {p.title} | – | {msg} | |")
            continue
        verdict = "ready" if r["score"] >= 80 else "needs work" if r["score"] >= 50 else "not safe"
        rows.append(f"| {p.title} | **{r['score']}** {verdict} | {r['headline']} | {r.get('action', '')} |")
        chart_rows.append((p.title.replace(" Agent", ""), r["score"]))
    ep = parsed["episode"]
    return "\n".join([
        f"## One recording, {len(judges)} applications · {ep}",
        "Each specialist judged the same touch data against its own limits and failure tolerances.", "",
        charts.score_chart(chart_rows, f"Readiness by application · {ep}"), "",
        "| Application | Readiness | Top finding | First action |", "|---|---|---|---|", *rows, "",
        f"Ask for any one in depth, e.g. “industrial review of {ep}”.",
    ])
