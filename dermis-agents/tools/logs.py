"""Read hosted agents' logs with the Agentverse API key from .env (no avctl login, never expires).

  python tools/logs.py hub              last log lines of one agent
  python tools/logs.py hub -n 200       more lines
  python tools/logs.py all              a summary line per agent
"""

import argparse
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_hosted as hb  # noqa: E402

API = "https://agentverse.ai/v1/hosting/agents"


def api_key() -> str:
    env = (hb.ROOT / ".env").read_text() if (hb.ROOT / ".env").exists() else ""
    m = re.search(r"^AGENTVERSE_API_KEY=(.+)$", env, re.M)
    if not m:
        sys.exit("Put AGENTVERSE_API_KEY=... in .env (agentverse.ai → profile → API Keys)")
    return m.group(1).strip()


def logs(address: str, key: str) -> list[dict]:
    r = httpx.get(f"{API}/{address}/logs/latest", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    return sorted(r.json(), key=lambda e: e["log_timestamp"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", help="hub, data_explorer, … or all")
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--all-lines", action="store_true", help="include envelope/registration noise")
    args = ap.parse_args()
    key = api_key()
    keys = hb.HOSTED if args.agent == "all" else [args.agent]
    for k in keys:
        addr = hb.deployed_address(k)
        if not addr:
            print(f"{k}: not deployed")
            continue
        entries = logs(addr, key)
        if not args.all_lines:
            entries = [e for e in entries if not re.search(r"Almanac|Registered to|registration", e["log_entry"])]
        if args.agent == "all":
            last = next((e for e in reversed(entries) if "self-check" in e["log_entry"]), None)
            chats = [e for e in entries if "chat from" in e["log_entry"]]
            print(f"{k:15} chats={len(chats):3}  {last['log_entry'][12:140] if last else 'no self-check yet'}")
            continue
        for e in entries[-args.n:]:
            print(e["log_timestamp"][11:23], e["log_entry"][:400])


if __name__ == "__main__":
    main()
