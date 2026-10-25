"""ASI:One Interactive Cards (card protocol v1): forms, detail views and payment reviews in the chat.

A card rides in a ChatMessage as one MetadataContent next to the narration text. Button presses come
back as an ordinary chat message (JSON of the CTA's `selection`, or prose via the planner).
"""

from __future__ import annotations

import json
import re

from uagents_core.contrib.protocols.chat import MetadataContent

from dermis_profiles import PROFILES

OBJECTS = ["can", "egg", "ball", "cup", "bottle", "sponge"]
SCENARIOS = ["nominal", "crush", "light_grip", "drop"]
MAX_PAYLOAD = 60_000


def content(card: dict) -> MetadataContent:
    meta = {"card_protocol_version": "1", "requires_card_interaction": "false" if card.get("terminal") else "true",
            "card_kind": card["kind"], "card_payload": json.dumps(card["payload"])}
    if card.get("terminal"):
        meta["is_terminal"] = "true"
    if card.get("width"):
        meta["preferred_drawer_width_px"] = str(card["width"])
    if len(meta["card_payload"]) > MAX_PAYLOAD:
        raise ValueError("card payload over 64 KB")
    return MetadataContent(type="metadata", metadata=meta)


def start_form() -> dict:
    apps = [{"value": p.key, "label": p.title.replace(" Agent", "")} for p in PROFILES if p.kind == "touch"]
    return {"kind": "form", "width": 480, "payload": {
        "title": "Analyze a robot grasp",
        "fields": [
            {"name": "application", "kind": "select", "label": "Robot / application", "required": True,
             "options": apps},
            {"name": "object", "kind": "select", "label": "Object", "required": True,
             "options": [{"value": o, "label": o} for o in OBJECTS]},
            {"name": "scenario", "kind": "select", "label": "Scenario", "required": True,
             "options": [{"value": s, "label": s.replace("_", " ")} for s in SCENARIOS]},
            {"name": "safe_grip", "kind": "checkbox", "label": "SafeGrip force limiting on"},
        ],
        "submit_cta": {"label": "Analyze", "selection": {"action": "analyze"}},
    }}


def recording_card(a: dict, price_fet: str) -> dict:
    """Detail card for analyze_recording output."""
    eid = a["episode"]
    rows = [
        {"label": "Outcome", "value": a["outcome"]},
        {"label": "Readiness", "value": f"{a['readiness_score']}/100 for {a['application'].replace('_', ' ')}"},
        {"label": "Object", "value": f"{a['object']} ({a['mass_g']} g), {a['scenario'].replace('_', ' ')}"},
        {"label": "SafeGrip", "value": "on" if a["safe_grip"] else "off"},
        {"label": "Force limit", "value": f"{a['force_limit_n']} N"},
    ]
    if a.get("fault"):
        rows.append({"label": "Fault", "value": f"{a['fault'].replace('_', ' ')} at {a['fault_at_s']} s"})
    if a.get("findings"):
        rows.append({"label": "Top finding", "value": a["findings"][0]["message"][:160]})
    payload = {"title": f"{eid} · {a['object']} · {a['outcome']}", "summary_rows": rows, "ctas": [
        {"label": "Compare all applications", "selection": {"action": "compare", "episode": eid}, "primary": True},
        {"label": "SafeGrip on vs off", "selection": {"action": "safegrip_compare", "episode": eid}},
        {"label": f"Certified report ({price_fet} FET)", "selection": {"action": "buy_report", "episode": eid}},
    ]}
    hero = _first_image(a.get("chart", ""))
    if hero:
        payload["hero_image"] = hero
    return {"kind": "detail", "width": 640, "payload": payload}


def purchase_review(episode: str, price_fet: str, network: str) -> dict:
    return {"kind": "review", "width": 480, "payload": {
        "title": "Certified readiness report",
        "summary_rows": [
            {"label": "Recording", "value": episode},
            {"label": "Includes", "value": "all 10 applications scored, faults, SafeGrip what-if, fleet context, fix list"},
            {"label": "Price", "value": f"{price_fet} FET"},
            {"label": "Network", "value": network},
            {"label": "Verification", "value": "payment checked on-chain before the report is issued"},
        ],
        "approve_cta": {"label": f"Pay {price_fet} FET", "selection": {"action": "confirm_purchase", "episode": episode},
                        "primary": True},
        "reject_cta": {"label": "Not now", "selection": {"action": "cancel_purchase", "episode": episode}},
    }}


def parse_selection(text: str) -> dict | None:
    """A card button press arrives as JSON (direct chat) or prose (via the planner)."""
    t = text.strip()
    if t.startswith("{"):
        try:
            sel = json.loads(t)
            return sel if isinstance(sel, dict) and sel.get("action") else None
        except json.JSONDecodeError:
            return None
    m = re.search(r"\b(confirm_purchase|cancel_purchase|buy_report)\b", t)
    if m:
        ep = re.search(r"episode_\d{4}", t)
        return {"action": m.group(1), **({"episode": ep.group(0)} if ep else {})}
    return None


def _first_image(md: str) -> str | None:
    m = re.search(r"!\[[^\]]*\]\((https?://[^)]+)\)", md or "")
    return m.group(1) if m else None
