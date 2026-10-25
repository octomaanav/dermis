"""Connect (or disconnect) running agents' mailboxes on Agentverse, like the Inspector's Connect button.

  1. Put your Agentverse API key in .env:   AGENTVERSE_API_KEY=...
     (agentverse.ai → your profile → API Keys → create one)
  2. Start the agents in mailbox mode:      DERMIS_MAILBOX=1 .venv/bin/python run.py
  3. In another terminal:
       .venv/bin/python connect.py                          all agents, most important first
       .venv/bin/python connect.py visualizer data_explorer  just these
       .venv/bin/python connect.py --disconnect food_agri    free a slot

Free Agentverse accounts hold 8 mailbox agents. The hub answers for every agent either way: when a
specialist isn't connected, the hub runs its analysis itself.
"""

import argparse
import os
import sys

import httpx

from dermis_agents import agents, config  # noqa: F401  (config loads .env)

# Who gets the 8 free slots: the front door, the two general agents, then the broadest specialists.
PRIORITY = ["hub", "visualizer", "data_explorer", "humanoid_arm", "industrial_arm", "quadruped",
            "warehouse", "sensor_qa"]

ap = argparse.ArgumentParser()
ap.add_argument("keys", nargs="*", help="agent keys (default: all, in priority order)")
ap.add_argument("--disconnect", action="store_true", help="disconnect the named agents instead")
args = ap.parse_args()

token = os.getenv("AGENTVERSE_API_KEY", "").strip()
if not token:
    sys.exit("AGENTVERSE_API_KEY is empty. Create one at agentverse.ai → profile → API Keys and put it in .env")

if args.disconnect and not args.keys:
    sys.exit("Name the agents to disconnect, e.g.: connect.py --disconnect food_agri surgical")
unknown = set(args.keys) - set(agents.all_keys())
if unknown:
    sys.exit(f"Unknown agents: {', '.join(sorted(unknown))}. Choose from: {', '.join(agents.all_keys())}")
keys = args.keys or PRIORITY + [k for k in agents.all_keys() if k not in PRIORITY]

path, body = ("/disconnect", {"user_token": token}) if args.disconnect else \
    ("/connect", {"user_token": token, "agent_type": "mailbox"})
ok = 0
for key in keys:
    url = f"http://127.0.0.1:{agents.port_for(key)}{path}"
    try:
        r = httpx.post(url, json=body, timeout=30)
        reply = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"detail": r.text}
        success = r.status_code == 200 and reply.get("success")
    except httpx.HTTPError as e:
        success, reply = False, {"detail": f"agent not running on {url} ({e.__class__.__name__}). Is run.py up?"}
    ok += bool(success)
    print(f"{'✅' if success else '❌'} {key:16} {agents.address_for(key)}  {'' if success else reply.get('detail', reply)}")

verb = "disconnected" if args.disconnect else "connected"
print(f"\n{ok}/{len(keys)} {verb}.")
if not args.disconnect and ok < len(keys):
    print("Agents that didn't connect still work through the hub, which runs their analysis itself.")
