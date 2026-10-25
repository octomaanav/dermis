import asyncio
import os

os.environ["DERMIS_CHARTS"] = "0"
os.environ["ASI1_API_KEY"] = ""

from dermis_agents import agents, analysis, hub, specialist  # noqa: E402
from dermis_agents.store import snapshot  # noqa: E402
from dermis_agents.profiles import BY_KEY, PROFILES  # noqa: E402


def run(coro):
    return asyncio.run(coro)


async def local_ask(key, text, parsed, brief):
    return await specialist.respond(BY_KEY[key], text, parsed, brief)


def facts_for(eid):
    return snapshot().facts(eid)


def test_noise_matches_sensor_spec():
    f = facts_for("episode_0000")
    for pod in f["pods"].values():
        assert pod["noise_vs_spec"] < 2  # sim noise is generated at spec


def test_touch_latency_is_against_true_force():
    f = facts_for("episode_0000")
    assert abs(f["touch_detect_latency_ms"]) < 30


def test_fracture_episode_is_critical_for_humanoid_not_quadruped():
    f = facts_for("episode_0004")
    hum = analysis.evaluate(BY_KEY["humanoid_arm"], f)
    quad = analysis.evaluate(BY_KEY["quadruped"], f)
    assert hum[0]["severity"] == "critical"
    assert not any(x["severity"] == "critical" for x in quad)


def test_sensor_agent_ignores_grip_failures():
    f = facts_for("episode_0004")
    assert all(x["check"] in ("sensor", "timing") for x in analysis.evaluate(BY_KEY["sensor_qa"], f))


def test_counterfactual_finds_safegrip_twin():
    eps = snapshot().episodes
    meta = next(e for e in eps if e["episode"] == "episode_0004")
    twin = analysis.counterfactual(meta, eps)
    assert twin["safe_grip"] != meta["safe_grip"] and twin["object"] == meta["object"]


def test_routing():
    assert hub.keyword_route("robot dog foot slip")[0] == "quadruped"
    assert hub.keyword_route("warehouse picking drops")[0] == "warehouse"
    assert run(hub.choose("compare all applications on episode 4")) == ["all"]


def test_episode_report_end_to_end():
    r = run(specialist.respond(BY_KEY["humanoid_arm"], "analyze episode 4"))
    assert r["episode"] == "episode_0004" and r["score"] < 50
    assert "What-if: SafeGrip on" in r["markdown"]


def test_unknown_episode_is_reported_not_raised():
    r = run(specialist.respond(BY_KEY["humanoid_arm"], "analyze episode 9999"))
    assert "don't have" in r["markdown"]


def test_matrix_covers_every_judging_profile():
    md = run(hub.respond("compare all applications on episode 4", local_ask))
    for p in PROFILES:
        assert (p.title in md) == p.in_matrix


def test_fleet_reports():
    for key in ("warehouse", "data_curator", "industrial_arm"):
        r = run(specialist.respond(BY_KEY[key], "fleet summary"))
        assert r["markdown"].startswith("## ")


def test_addresses_are_stable_and_distinct():
    addrs = [agents.address_for(k) for k in agents.all_keys()]
    assert len(set(addrs)) == len(addrs) == 14
    asyncio.set_event_loop(asyncio.new_event_loop())  # Agent() needs a loop; asyncio.run closed ours
    assert agents.build("quadruped").address == agents.address_for("quadruped")


def test_routing_to_viz_agents():
    assert hub.keyword_route("plot a histogram of peak force")[0] == "visualizer"
    assert hub.keyword_route("pivot table of outcomes")[0] == "data_explorer"


def test_explorer_tables():
    r = run(specialist.respond(BY_KEY["data_explorer"], "overview", {}))
    assert "pd.crosstab" in r["markdown"] and "| can |" in r["markdown"]
    r = run(specialist.respond(BY_KEY["data_explorer"], "success by scenario", {}))
    assert "| drop |" in r["markdown"]
    r = run(specialist.respond(BY_KEY["data_explorer"], "stats for episode 4", {"episode": "episode_0004"}))
    assert "calibration" in r["markdown"]


def test_visualizer_picks_chart_type():
    from dermis_agents import viz
    assert viz.chart_kind("scatter latency vs mass") == "scatter"
    assert viz.chart_kind("peak force vs mass") == "scatter"
    assert viz.chart_kind("compare episode 4 vs episode 0") == "compare"
    assert viz.chart_kind("heatmap of success") == "heatmap"
    for q in ("dashboard", "plot episode 4", "compare episode 4 with safegrip on", "pie of outcomes",
              "heatmap", "radar of episode 40", "plot the raw field for episode 4"):
        r = run(specialist.respond(BY_KEY["visualizer"], q, {}))
        assert r["markdown"].startswith("## Visualizer ·"), q


def test_runtime_needs_no_numpy_or_pandas():
    """Hosted Agentverse agents have no pandas; keep every runtime module free of numpy too."""
    import subprocess
    import sys
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, *a):\n"
        "        if name.split('.')[0] in ('numpy', 'pandas'): raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import asyncio\n"
        "from dermis_agents import agents, brain, cards, connectors, dataset, hub, payments, specialist, viz\n"
        "from dermis_agents.profiles import BY_KEY\n"
        "for k, q in [('visualizer', 'dashboard'), ('data_explorer', 'overview'), ('quadruped', 'episode 4')]:\n"
        "    r = asyncio.run(specialist.respond(BY_KEY[k], q, {}))\n"
        "    assert r['markdown'].startswith('## '), (k, r['markdown'][:200])\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**__import__("os").environ, "DERMIS_CHARTS": "0", "ASI1_API_KEY": ""})
    assert out.stdout.strip() == "ok", out.stderr[-2000:]


def test_agent_loop_uses_tools_and_restores_charts(monkeypatch):
    from dermis_agents import brain, llm

    script = iter([
        {"content": "", "tool_calls": [{"id": "1", "function": {"name": "analyze_recording",
                                                                "arguments": '{"episode": "episode_0004"}'}}]},
        {"content": "It crushed the can.\n\n[[chart:1]]\n\nTurn SafeGrip on."},
    ])
    seen = []

    async def fake_chat(messages, tools, max_tokens=1200):
        seen.append(messages[-1])
        return next(script)

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(brain.charts.config, "CHARTS", True)
    result = run(brain.converse("humanoid_arm", "why did episode 4 fail?"))
    answer = result["text"]
    assert result["card"]["kind"] == "detail"  # analyze_recording attaches a detail card
    tool_result = seen[1]["content"]
    assert seen[1]["role"] == "tool" and '"outcome": "fracture"' in tool_result
    assert "quickchart" not in tool_result and "[[chart:1]]" in tool_result  # model never sees the URL
    assert answer.startswith("It crushed the can.") and "quickchart.io" in answer


def test_agent_returns_none_without_llm(monkeypatch):
    from dermis_agents import brain, llm

    async def down(*a, **k):
        return None

    monkeypatch.setattr(llm, "chat", down)
    assert run(brain.converse("hub", "hello")) is None  # caller falls back to the structured report


def test_memory_drops_chart_urls():
    from dermis_agents import brain
    h = brain.remember([], "q", "answer\n\n![c](https://quickchart.io/chart?c=" + "x" * 5000 + ")")
    assert "quickchart" not in h[-1]["content"] and len(h) == 2


def test_generic_dataset_engine():
    from dermis_agents import dataset as D
    t = D.parse("city,temp,rain\nA,20,1\nB,25,0\nA,22,3\nC,,2", "weather")
    assert t.types == {"city": "text", "temp": "number", "rain": "number"}
    g = t.query(group_by=["city"], metrics=[{"column": "rain", "agg": "sum"}], sort_by="sum_rain")
    assert g["rows"][0] == {"city": "A", "sum_rain": 4}
    assert t.query(filters=[{"column": "temp", "op": ">", "value": 21}])["matched_rows"] == 2
    assert D.parse('[{"a": 1, "b": {"c": 2}}, {"a": 3, "b": {"c": 4}}]', "j").columns == ["a", "b.c"]
    assert len(D.load("dermis").rows) == 192


def test_pasted_table_becomes_active_dataset():
    from dermis_agents import brain
    session = {}
    msg = brain.attach_inline_dataset("which team scores most?\nteam,goals\nRed,3\nBlue,5\nRed,4", session)
    assert session["dataset"]["source"] == "pasted" and "3 rows" in msg and msg.startswith("which team")
    session = {}
    brain.attach_inline_dataset("analyze https://example.com/data/sales.csv please", session)
    assert session["dataset"]["source"] == "https://example.com/data/sales.csv"


def test_chart_dataset_tool():
    from dermis_agents import brain
    turn = brain.Turn("data_explorer", {"dataset": {"source": "pasted", "text": "k,v\na,1\nb,2\na,5"}})
    out = brain.t_chart_dataset({"kind": "bar", "x": "k", "y": "v", "agg": "sum"}, turn)
    assert out["rows"][0] == {"k": "a", "sum_v": 6} and "chart" in out


class _Tx:
    def __init__(self, ok, transfer):
        self._ok, self.events = ok, {"transfer": transfer}

    def is_successful(self):
        return self._ok


def _ledger(tx, on_testnet=True):
    class L:
        def __init__(self, testnet):
            self.testnet = testnet

        def query_tx(self, h):
            if self.testnet != on_testnet:
                raise KeyError("not found")
            return tx
    return lambda testnet=True: L(testnet)


def test_payment_verification():
    from dermis_agents import payments as P
    good = {"recipient": "fetch1me", "sender": "fetch1buyer", "amount": f"{10**17}{P.DENOM}"}
    assert P.verify_payment("tx", "0.1", "fetch1me", "fetch1buyer", _ledger(_Tx(True, good)))[0]
    assert P.verify_payment("tx", "0.1", "fetch1me", None, _ledger(_Tx(True, good)))[0]
    assert not P.verify_payment("tx", "0.1", "fetch1other", None, _ledger(_Tx(True, good)))[0]
    assert not P.verify_payment("tx", "0.1", "fetch1me", None, _ledger(_Tx(False, good)))[0]
    low = {**good, "amount": f"{10**16}{P.DENOM}"}
    assert "price" in P.verify_payment("tx", "0.1", "fetch1me", None, _ledger(_Tx(True, low)))[1]


def test_cards_and_selections():
    from dermis_agents import brain, cards
    a = brain.t_analyze_recording({"episode": "episode_4"}, "humanoid_arm")
    card = cards.recording_card(a, "0.1")
    meta = cards.content(card).metadata
    assert meta["card_kind"] == "detail" and meta["card_protocol_version"] == "1"
    assert cards.content(cards.start_form()).metadata["card_kind"] == "form"
    assert cards.parse_selection('{"action": "confirm_purchase", "episode": "episode_0004"}')["episode"] == "episode_0004"
    assert cards.parse_selection("User chose confirm_purchase for episode_0004")["action"] == "confirm_purchase"
    assert cards.parse_selection("what is a friction cone?") is None


def test_certified_report_contents():
    md = specialist.certified_report("episode_0004", "0.1 FET · tx T · stable-testnet")
    assert "Passes 2 of 10 applications" in md and "tx T" in md and '"verdicts"' in md


def test_payment_found_on_the_other_network():
    from dermis_agents import payments as P
    main = {"recipient": "fetch1me", "sender": "fetch1buyer", "amount": f"{10**17}afet"}
    ok, why = P.verify_payment("tx", "0.1", "fetch1me", None, _ledger(_Tx(True, main), on_testnet=False))
    assert ok and "mainnet" in why
