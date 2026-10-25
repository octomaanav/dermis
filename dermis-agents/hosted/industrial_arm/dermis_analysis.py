"""Judging recordings: per-profile findings, scores, fleet stats. Pure Python, runs anywhere.

Works on the facts dicts that features.analyze_episode produced at snapshot build time.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from dermis_profiles import Profile

GRIP = ["open", "touch", "grip", "hold", "capped", "crush"]
FAULT = ["none", "fracture", "slip", "impact", "early_release", "overpress"]


def peak_force(facts: dict) -> tuple[str | None, float]:
    key = "peak_force_gt_n" if facts["has_ground_truth_force"] else "peak_force_est_n"
    best = max(facts["pods"].items(), key=lambda kv: kv[1][key] or 0, default=(None, {key: 0}))
    return best[0], float(best[1][key] or 0)


def evaluate(profile: Profile, facts: dict) -> list[dict]:
    """Findings for one profile, most important first: severity, then the profile's check order."""
    out = []
    checks = profile.checks

    def add(check, sev, code, msg):
        if check in checks:
            out.append({"severity": sev, "code": code, "check": check, "message": msg,
                        "action": profile.action(code)})

    label = profile.pod_labels
    # Sensor problems decide whether the grip numbers can be trusted at all.
    for p, v in facts["pods"].items():
        if v["rail_frames"]:
            add("sensor", "warning", "rail", f"{label[p]} magnetometer railed for {v['rail_frames']} frames "
                f"({v['rail_frames'] / facts['rate_hz'] * 1000:.0f} ms): force readings there are not valid.")
        if v["noise_vs_spec"] > 3:
            add("sensor", "warning", "noisy", f"{label[p]} noise is {v['noise_vs_spec']}x spec ({v['noise_ut']} µT).")
    if facts["dropped_frames"]:
        add("sensor", "warning", "frame_drops", f"{facts['dropped_frames']} dropped frames.")
    lat = facts.get("touch_detect_latency_ms")
    if lat is not None and lat > profile.latency_ms:
        add("timing", "warning", "late_detection",
            f"Touch detected {lat:.0f} ms after true contact (this domain tolerates {profile.latency_ms:.0f} ms).")
    if facts.get("touch_onset_s") is None:
        add("timing", "warning", "no_contact", "No contact was detected on any pod.")

    pod, peak = peak_force(facts)
    limit = profile.limit_for(facts.get("object") or "")
    if pod and peak > limit:
        kind = "reference force" if facts["has_ground_truth_force"] else "field estimate, uncalibrated"
        add("force", "critical", "force_over_limit",
            f"Peak force {peak:.1f} N on {label[pod]} ({kind}) is over this profile's {limit:.1f} N limit "
            f"for a {facts.get('object')}.")

    if facts.get("fault"):
        crit = facts["fault"] in profile.critical_faults
        add("fault", "critical" if crit else "info", f"fault_{facts['fault']}",
            f"Fault: {facts['fault'].replace('_', ' ')} at {facts['fault_at_s']} s"
            + ("." if crit else " (not a failure mode this domain is judged on)."))

    m = facts.get("min_friction_margin")
    if m is not None and m < profile.min_margin:
        add("margin", "warning", "low_margin", f"Friction margin fell to {m:.2f} (profile floor {profile.min_margin:.2f}).")

    slip_pods = [p for p, v in facts["pods"].items() if v["slip_signal_s"] > 0.05]
    if slip_pods:
        first = min(facts["pods"][p]["slip_signal_first_s"] for p in slip_pods)
        add("slip_signal", "warning" if facts.get("fault") else "info", "slip_signal",
            f"Slip-band vibration on {', '.join(label[p] for p in slip_pods)} from {first} s.")

    rank = {"critical": 0, "warning": 1, "info": 2}
    out.sort(key=lambda f: (rank[f["severity"]], checks.index(f["check"])))
    return out


def score(findings: list[dict]) -> int:
    penalty = {"critical": 35, "warning": 10, "info": 2}
    return max(0, 100 - sum(penalty[f["severity"]] for f in findings))


def counterfactual(meta: dict, episodes: list[dict]) -> dict | None:
    """Same object, scenario and seed with SafeGrip flipped: what changed."""
    for e in episodes:
        if (e["object"], e["scenario"], e["seed"]) == (meta["object"], meta["scenario"], meta["seed"]) \
                and e["safe_grip"] != meta["safe_grip"]:
            return e
    return None


def fleet(episodes: list[dict], objects: tuple[str, ...] = ()) -> dict:
    eps = [e for e in episodes if not objects or e["object"] in objects]
    by_obj = defaultdict(lambda: {True: Counter(), False: Counter()})
    for e in eps:
        by_obj[e["object"]][bool(e["safe_grip"])][e["outcome"]] += 1

    def rate(cnt: Counter) -> float | None:
        total = sum(cnt.values())
        return round(cnt["success"] / total, 3) if total else None

    table = {}
    for obj, d in sorted(by_obj.items()):
        table[obj] = {
            "n": sum(d[True].values()) + sum(d[False].values()),
            "success_safegrip": rate(d[True]),
            "success_no_safegrip": rate(d[False]),
            "failures": dict((d[True] + d[False]) - Counter({"success": 10**9})),
        }
    outcomes = Counter(e["outcome"] for e in eps)
    scen = defaultdict(Counter)
    for e in eps:
        scen[e["scenario"]][e["outcome"]] += 1
    return {
        "episodes": len(eps),
        "outcomes": dict(outcomes),
        "by_object": table,
        "by_scenario": {s: {"n": sum(c.values()), "success": rate(c)} for s, c in sorted(scen.items())},
        "sources": dict(Counter(e.get("source", "sim") for e in eps)),
    }


def pick_episode(profile: Profile, episodes: list[dict], obj: str | None = None,
                 scenario: str | None = None, safe_grip: bool | None = None) -> dict | None:
    """The most informative episode for this profile: a critical fault on a relevant object first."""
    pool = [e for e in episodes
            if (not obj or e["object"] == obj)
            and (not scenario or e["scenario"] == scenario)
            and (safe_grip is None or bool(e["safe_grip"]) == safe_grip)
            and (obj or not profile.objects or e["object"] in profile.objects)]
    if not pool:
        return None

    order = {o: i for i, o in enumerate(profile.objects)}

    def rank(e):
        faults = e.get("faults") or []
        crit = [profile.critical_faults.index(f) for f in faults if f in profile.critical_faults]
        # Worst fault this domain cares about, on its primary object, canonical seed first.
        return (min(crit) if crit else 99, e["outcome"] == "success", order.get(e["object"], 99),
                e.get("seed", 0), e["episode"])

    return sorted(pool, key=rank)[0]
