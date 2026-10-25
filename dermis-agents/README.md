# Dermis Agents: robot touch intelligence on Fetch.ai

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

Robots are getting hands, but no sense of touch: they crush eggs, drop bottles and can't tell a sensor
glitch from a failed grip. Dermis makes a magnetic tactile glove, and these agents turn its touch data
into decisions. Ask in ASI:One, in plain words, whether a grasp is safe for *your* robot (humanoid,
factory arm, robot dog, prosthetic hand, warehouse picker…), why it failed, and what to change. The
agents plan, call tools over real data, consult each other, answer with charts and interactive cards,
and sell a certified report for 0.1 FET over the Payment Protocol. They also analyze any dataset you
bring.

Built for the Fetch.ai track at Cal Hacks 13.0
([hackpack](https://www.fetch.ai/events/hackathons/calhacks-13-0/hackpack)).

## Fetch.ai challenge checklist

| Requirement | How this project meets it |
|---|---|
| Agent registered on Agentverse | 8 hosted agents (`tools/deploy_hosted.py`), or mailbox agents from `run.py` + `connect.py` |
| Agent Chat Protocol | every agent includes `chat_protocol_spec` (`dermis_agents/agents.py`) |
| Discoverable and usable in ASI:One | agent READMEs with keywords, descriptions and starter prompts; the full workflow runs in the chat |
| Meaningful tool use / multi-agent orchestration | ASI:One `asi1` agents with 19 tools; the hub consults specialist agents with `ask_agent` (uAgents messages) |
| Whole workflow inside ASI:One | ask → analysis with charts → card buttons → payment → certified report, no custom frontend |
| Public GitHub repo with instructions | this repo; Run it section below |
| Bonus: multi-agent collaboration | hub ↔ specialists over a typed `TouchQuery`/`TouchReport` protocol, answers combined by the hub |
| Bonus: Payment Protocol, credible model | hub sells a certified readiness report: 0.1 FET, `fet_direct`, verified on-chain with cosmpy (`payments.py`) |
| Bonus: ASI Interactive Cards | start form, recording detail card with actions, purchase review card (`cards.py`) |
| Bonus: reliability and recovery | tool errors go back to the model to retry; dead specialist → hub runs it; LLM down → structured report |
| Bonus: real-time data and external services | live SpacetimeDB SQL, any CSV/JSON URL, web pages, QuickChart, Fetch ledger |
| Bonus: keeps running after the hackathon | hosted on Agentverse, data from a versioned snapshot, rebuilt when new recordings land |
| Innovation Lab badges | at the top of this README and every agent README |

## The agents

All of them speak the Chat Protocol and are LLM agents with tools. Hosted: the 8 marked ★ (free
Agentverse accounts hold 8); the other specialists answer through the hub.

| Key | Agent name | What it does |
|---|---|---|
| `hub` ★ | dermis_hub | Front door and orchestrator. Answers with tools, consults specialists, shows cards, sells the certified report |
| `humanoid_arm` ★ | dermis_humanoid_arm | Dexterous humanoid hands: grasp stability, fragile objects, finger timing |
| `industrial_arm` ★ | dermis_industrial_arm | Production grippers: yield, repeatability, headroom |
| `quadruped` ★ | dermis_quadruped | Robot dogs: foot contact timing, foot slip, touchdown impact |
| `warehouse` ★ | dermis_warehouse | Pick and place: drops, slip on heavy items, impact |
| `sensor_qa` ★ | dermis_sensor_qa | Sensor health: saturation, noise vs spec, dropped frames; separates sensor faults from grip failures |
| `data_explorer` ★ | dermis_data_explorer | Data Analyst: any CSV/TSV/JSON by URL or pasted, plus the Dermis data. Filter, group, pivot, computed columns, correlations, charts |
| `visualizer` ★ | dermis_visualizer | Charts for the recordings or any dataset: dashboards, time series, overlays, histograms, scatter, pie, radar, heatmaps |
| `prosthetic_hand` | dermis_prosthetics | Prosthetics: never crush, never drop |
| `surgical` | dermis_surgical | Delicate handling: 50% safety factor on yield, noise floor |
| `food_agri` | dermis_food_agri | Produce: crushing and bruising |
| `eldercare` | dermis_eldercare | Care robots: hand-over safety, over-pressure, predictable release |
| `teleop` | dermis_teleop | The glove itself: touch detection latency, what the operator feels |
| `data_curator` | dermis_data_curator | Training data: label balance, coverage gaps, what to record next |

Hosted agents get their addresses from Agentverse; after deploying, they're printed by
`tools/deploy_hosted.py` and stored in `hosted/<agent>/.avctl/config.toml`.

### Deployed agents (hosted on Agentverse)

**Start here: Dermis Hub (@dermis-hub)** `agent1q02mu7cnwewnnc0k4v474nx5qjda4mdp9tvxlwjtrtccmyce7r5qqgzq3xj` · [Agentverse profile](https://agentverse.ai/agents/details/agent1q02mu7cnwewnnc0k4v474nx5qjda4mdp9tvxlwjtrtccmyce7r5qqgzq3xj)

**Chat now:** [asi1.ai/ai/agent1q02mu7cnwewnnc0k4v474nx5qjda4mdp9tvxlwjtrtccmyce7r5qqgzq3xj](https://asi1.ai/ai/agent1q02mu7cnwewnnc0k4v474nx5qjda4mdp9tvxlwjtrtccmyce7r5qqgzq3xj) (works on desktop and in a phone browser)

| Agent | Handle | Address | Agentverse |
|---|---|---|---|
| `hub` | @dermis-hub | `agent1q02mu7cnwewnnc0k4v474nx5qjda4mdp9tvxlwjtrtccmyce7r5qqgzq3xj` | [profile](https://agentverse.ai/agents/details/agent1q02mu7cnwewnnc0k4v474nx5qjda4mdp9tvxlwjtrtccmyce7r5qqgzq3xj) |
| `data_explorer` | @dermis-analyst | `agent1qwls2zmjuzdr9q89448pq9wnumjx8fw2l2k5wfyyvrkr2nxnec87k97gsx5` | [profile](https://agentverse.ai/agents/details/agent1qwls2zmjuzdr9q89448pq9wnumjx8fw2l2k5wfyyvrkr2nxnec87k97gsx5) |
| `visualizer` | @dermis-charts | `agent1qtnldfaawly9j4gt55ujddflthw688dulmqm9hvzznmgprd7s94fgryg5rl` | [profile](https://agentverse.ai/agents/details/agent1qtnldfaawly9j4gt55ujddflthw688dulmqm9hvzznmgprd7s94fgryg5rl) |
| `humanoid_arm` | @dermis-humanoid | `agent1qgpdsvnnqklc5gy3xvenk30saa2dzn34sve8ekfad562t6wctueqypav4fv` | [profile](https://agentverse.ai/agents/details/agent1qgpdsvnnqklc5gy3xvenk30saa2dzn34sve8ekfad562t6wctueqypav4fv) |
| `industrial_arm` | @dermis-industrial | `agent1qt7agfdeh7vzppus94fclf7l27v0sqxhhq5stdr553w3rv8p9cadxzgfazs` | [profile](https://agentverse.ai/agents/details/agent1qt7agfdeh7vzppus94fclf7l27v0sqxhhq5stdr553w3rv8p9cadxzgfazs) |
| `quadruped` | @dermis-quadruped | `agent1q0zjcjwll2v7utcwh0crghdtf4ckw0lvetxmlsu66xvaj9633mkfyrdpyrq` | [profile](https://agentverse.ai/agents/details/agent1q0zjcjwll2v7utcwh0crghdtf4ckw0lvetxmlsu66xvaj9633mkfyrdpyrq) |
| `warehouse` | @dermis-warehouse | `agent1qvykzu7slhz3nzchcu20v7gyhg25kp9crtekl6gu2rul8l6v5qy265xanhg` | [profile](https://agentverse.ai/agents/details/agent1qvykzu7slhz3nzchcu20v7gyhg25kp9crtekl6gu2rul8l6v5qy265xanhg) |
| `sensor_qa` | @dermis-sensors | `agent1qfr6rvpp8uhl53fsq773e2ykz2nfzd57v5lug57hf3supv8km6947c9xur0` | [profile](https://agentverse.ai/agents/details/agent1qfr6rvpp8uhl53fsq773e2ykz2nfzd57v5lug57hf3supv8km6947c9xur0) |

## Try it in ASI:One

- `hi` → a form card to pick robot, object, scenario and SafeGrip
- `Why did episode 4 fail, and would SafeGrip have saved it?` → analysis, chart and a detail card
- `Ask the warehouse agent which items we drop most` → the hub consults a specialist agent
- `Compare all applications on episode 4` → one recording scored for 10 robot types
- `I need a certified report for episode 4` → review card → Pay 0.1 FET → report verified on-chain
- `Analyze https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv` → any dataset
- paste a CSV into the chat and ask a question about it
- `Read https://en.wikipedia.org/wiki/Tactile_sensor and compare it to Dermis`

## Payments and cards

- **Certified readiness report** (`payments.py`, `specialist.certified_report`): one recording scored
  against all 10 applications with verdicts, evidence, the SafeGrip what-if, fleet context, a fix list and
  a JSON summary that includes the payment receipt. Price `DERMIS_REPORT_PRICE_FET` (default 0.1 FET), paid
  with `fet_direct` to the hub's wallet. The hub verifies the transaction on-chain (recipient, amount,
  denom, buyer when given; each transaction is accepted once) before sending `CompletePayment` and the
  report. Testnet by default (`DERMIS_FET_TESTNET=true`), so demos cost nothing.
- **Interactive cards** (`cards.py`, card protocol v1): a start form, a detail card after each analysis
  (compare all applications / SafeGrip on vs off / buy the report), and a review card before paying. The
  Pay and Not now buttons are handled without the LLM, so money paths stay deterministic.

## How it works

```
 SpacetimeDB / local 350 Hz recordings
        │  python -m dermis_agents.snapshot   (numpy, build time: features.py)
        ▼
 snapshot/dermis_snapshot.json  ── public URL ──┐  every recording analysed once: facts, chart series,
                                                 │  calibration fits, sample rows (2.3 MB, 0.3 MB gzipped)
ASI:One ──ACP──▶ Dermis Hub ──TouchQuery──▶ specialists (parallel) ──TouchReport──▶ Hub ──▶ ASI:One
                   │ ASI:One LLM: parse + route (rule-based fallback)
                   └─ a specialist that isn't deployed, or doesn't answer, runs inside the hub
```

The agents never touch raw frames at runtime, so the same code runs locally and as Agentverse-hosted
agents (which have no pandas and can't ship 222 MB of recordings).

- **Every agent speaks the Agent Chat Protocol**, so each one is usable on its own from ASI:One,
  not only through the hub.
- **The analysis is real computation on the recordings, not LLM text.** It covers per-pod force,
  touch-onset latency against true contact, slip-band vibration (high-passed field energy),
  magnetometer rail, robust noise against the MLX90393 spec, dropped frames and friction margin. It
  also finds the SafeGrip on/off twin of the same run to show what would have changed.
- **Every agent is an ASI:One LLM agent with tools** (`dermis_agents/brain.py`). The `asi1` model holds the
  conversation (it remembers the last turns and the active dataset per user), plans, and calls tools:
  recordings (list, analyze, fleet stats, compare applications, sensor health, overview, charts, tables),
  any dataset (`dataset.py`: load by URL or paste, describe, query, pivot, add computed columns with a
  whitelisted formula parser, correlate, chart), outside data (`connectors.py`: web pages, live
  SpacetimeDB SQL), a start form, and on the hub the paid report and `ask_agent`, which messages a
  specialist agent over uAgents and folds its answer in. Set `ASI1_AGENT_MODEL` to change the model
  (`asi1-ultra` needs ASI:One Pro). The model is told to take every number from a tool. Charts reach
  the model as `[[chart:N]]` tokens and are swapped back in afterwards. If ASI:One is unreachable, the
  agent answers with the structured report instead.
- **Charts** are QuickChart image URLs, so they render inline in ASI:One. The chart URL contains
  the downsampled force series. Set `DERMIS_CHARTS=0` to leave the charts out.
- **The profiles** in `dermis_agents/profiles.py` set each domain's force safety factor, friction
  margin floor, critical faults, check priority, latency tolerance and recommended actions. To add an
  application, add one `Profile`.

## Run it

Use Python 3.10 to 3.13. uagents doesn't run on 3.14.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env                       # optional: SpacetimeDB, ASI1_API_KEY

# 1. No network at all: hub + specialists in-process
.venv/bin/python -m dermis_agents.cli "compare all applications on episode 4"
.venv/bin/python -m dermis_agents.cli quadruped "check slip for the robot dog"

# 2. Real agents, real Chat Protocol, locally
.venv/bin/python run.py                    # terminal 1: all 14, one process each (hub :8000, others :8001-8013)
.venv/bin/python ask.py "which items do we drop most in the warehouse?"   # terminal 2
.venv/bin/python ask.py --to sensor_qa "sensor health check across the fleet"

# 3. Mailbox agents on Agentverse, running on this machine
DERMIS_MAILBOX=1 .venv/bin/python run.py
.venv/bin/python connect.py                # connects mailboxes (AGENTVERSE_API_KEY in .env)

.venv/bin/python -m pytest -q tests
```

### Hosted on Agentverse (always on, no machine needed)

```bash
.venv/bin/python -m dermis_agents.snapshot      # after data changes; commit + push the snapshot
avctl auth login                                 # once, in the browser
.venv/bin/python tools/deploy_hosted.py          # builds hosted/<agent>/ and deploys all 8
```

Hosted agents need your ASI:One key as an Agentverse secret named `DERMIS_ASI1_API_KEY` (put
`DERMIS_ASI1_API_KEY=...` in a file and run `avctl hosting add secrets -a <agent address> -f <file>` per
agent). Without it they use the key Agentverse injects, which stops working when the account's LLM budget
runs out. Each agent logs a `self-check:` line at startup saying whether its data and LLM work.

After the first deploy, use the Agentverse API key (`AGENTVERSE_API_KEY` in `.env`, never expires)
instead of the hourly avctl login:

```bash
.venv/bin/python tools/deploy_api.py        # rebuild, check, upload code, restart, show self-checks
.venv/bin/python tools/logs.py all          # one health line per agent
.venv/bin/python tools/logs.py hub          # an agent's recent log: messages, answer times, errors
```

`tools/build_hosted.py` turns the package into one folder per hosted agent (same modules, imports
flattened). Free accounts hold 8 agents, so the hosted set is the hub, Visualizer, Data Explorer,
humanoid arm, industrial arm, quadruped, warehouse and sensor health; the other specialists answer
through the hub. Hosted agents read the snapshot from
`https://raw.githubusercontent.com/Soham1010/dermis-agents/main/snapshot/dermis_snapshot.json`.

Questions to try in ASI:One:

- Analyze episode 4 for a humanoid arm
- Compare all applications on episode 4
- Which items do we drop most in the warehouse?
- Are any sensor readings unreliable? Sensor health check across the fleet
- Would a prosthetic hand drop the cup?
- What data should we collect next for training?
- Pivot table of success by scenario · Peak force by object and scenario
- Dashboard · Histogram of peak force · Scatter latency vs peak force · Heatmap of success
- Compare episode 4 with SafeGrip on · Plot the raw field for episode 4 · Radar of episode 40
- Help

## SpacetimeDB contract

`data.py` reads over SpacetimeDB's HTTP SQL API (`POST /v1/database/{db}/sql`) when the snapshot is
built. It uses the local dataset whenever `SPACETIME_DB` is unset or the query fails, and the snapshot
records which source it came from.

| Table | Columns |
|---|---|
| `episode` | `episode_id` (pk), `application` (`hand`, `quadruped`, …), `source` (`sim`/`hardware`), `object`, `mass_g`, `mu`, `scenario`, `safe_grip`, `seed`, `frames`, `seconds`, `outcome`, `faults` (JSON text), `first_fault_frame` |
| `frame` | `episode_id`, `frame`, then the Dermis column names (`t_device_us`, `index_tip_bx_ut` … `label_fault`) |

Hardware sessions can leave out the ground-truth columns (`*_normal_n`, labels, `object_lift_m`).
The analysis then falls back to the field-based force estimate and marks it uncalibrated.

## What the numbers are

The built-in data is the Dermis dataset: 192 grasp episodes in the glove hub's column layout. Field-to-
force uses 1.5 mT/N until the load-cell calibration (FORCE_REF) is done. Object yield forces are
engineering estimates (can 18 N, egg 25 N, cup 4 N). The quadruped agent reads hand episodes with the
four pods mapped to four feet until quadruped data exists, and it says so.

## Layout

```
dermis_agents/
  profiles.py    the 13 agents' domains, limits and keywords
  data.py        SpacetimeDB + local dataset, same shapes (build time)
  features.py    numpy feature extraction from raw frames (build time)
  snapshot.py    builds snapshot/dermis_snapshot.json
  store.py       loads the snapshot from a file or URL (runtime)
  analysis.py    per-profile evaluation, fleet stats, SafeGrip twin (pure Python)
  specialist.py  episode / fleet / sensor-scan / data-curation reports
  hub.py         routing, fan-out, cross-application matrix
  llm.py         ASI:One LLM client with rule fallbacks
  viz.py         Data Explorer (tables) and Visualizer (chart picker), pure Python
  charts.py      QuickChart URLs + text sparklines
  agents.py      uAgents: Chat Protocol, internal TouchQuery protocol, recovery, cards, payments
  brain.py       the LLM agent loop and its 19 tools
  dataset.py     table engine for any CSV/TSV/JSON dataset (pure Python)
  connectors.py  web pages and live SpacetimeDB SQL
  cards.py       ASI:One Interactive Cards
  payments.py    Payment Protocol seller with on-chain FET verification
run.py           starts one process per agent (local or mailbox)
connect.py       connects/disconnects mailbox agents on Agentverse
tools/           build_hosted.py + deploy_hosted.py for Agentverse-hosted agents
hosted/          generated hosted agents (don't edit; rebuild)
ask.py           Chat Protocol test client
examples/        the original two-agent starter
```
