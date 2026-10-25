"""Run the Dermis agents, one process per agent (the Agentverse Inspector can't connect Bureaus).

A supervisor restarts any agent that dies and restarts all of them when the code changes.

  python run.py                       all 14 agents: hub on :8000, the rest on :8001-8013
  DERMIS_MAILBOX=1 python run.py      same, reachable from Agentverse / ASI:One through mailboxes
  python run.py --only hub,quadruped  a subset (the hub runs missing specialists itself)
  python run.py --solo hub            one agent in this process
  python run.py --addresses           print every agent's address and port, then exit
"""

import argparse
import logging
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from dermis_agents import agents, config


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def stream(key: str, proc: subprocess.Popen) -> None:
    for line in proc.stdout:
        print(f"{key:15}| {line}", end="", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated agent keys: " + ",".join(agents.all_keys()))
    ap.add_argument("--solo", choices=agents.all_keys())
    ap.add_argument("--addresses", action="store_true")
    args = ap.parse_args()

    if args.solo:
        # Hide the "not enough funds for the Almanac contract" noise: the free Almanac API
        # registration is what Agentverse and ASI:One use, and it succeeds without funds.
        logging.getLogger("uagents.registration").setLevel(logging.ERROR)
        agents.build(args.solo).run()
        return

    keys = args.only.split(",") if args.only else agents.all_keys()
    unknown = set(keys) - set(agents.all_keys())
    if unknown:
        ap.error(f"unknown agents: {', '.join(sorted(unknown))}")

    if args.addresses:
        for k in agents.all_keys():
            print(f"{k:16} :{agents.port_for(k)}  {agents.address_for(k)}")
        return

    busy = [k for k in keys if port_in_use(agents.port_for(k))]
    if busy:
        sys.exit(f"Ports already in use for: {', '.join(busy)}. The agents are probably already running in "
                 "another terminal. Stop that one first (Ctrl+C there).")

    mode = "mailbox (Agentverse)" if config.MAILBOX else "local"
    print(f"Starting {len(keys)} Dermis agents · {mode}")
    for k in keys:
        print(f"  {k:16} :{agents.port_for(k)}  {agents.address_for(k)}")
    if config.MAILBOX:
        print("\nFirst run only: open each link, click Connect → Mailbox (leave this running):")
        for k in keys:
            print(f"  {k:16} https://agentverse.ai/inspect/?uri=http%3A//127.0.0.1%3A{agents.port_for(k)}"
                  f"&address={agents.address_for(k)}")
    print(flush=True)

    Supervisor(keys).run()


class Supervisor:
    """Keeps every agent process alive: restarts one that dies (with back-off), and restarts all of them
    when the code or .env changes, so a running agent never mixes old code in memory with new files."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        self.procs: dict[str, subprocess.Popen] = {}
        self.backoff = {k: 2.0 for k in keys}
        self.next_start = {k: 0.0 for k in keys}
        self.stopping = False

    def start(self, key: str) -> None:
        p = subprocess.Popen([sys.executable, __file__, "--solo", key], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.procs[key] = p
        threading.Thread(target=stream, args=(key, p), daemon=True).start()

    def stop_all(self) -> None:
        for p in self.procs.values():
            if p.poll() is None:
                p.terminate()
        for p in self.procs.values():
            try:
                p.wait(timeout=20)  # agents mark themselves inactive on Agentverse while stopping
            except subprocess.TimeoutExpired:
                p.kill()
        self.procs.clear()

    def run(self) -> None:
        def on_signal(*_):
            self.stopping = True
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        for k in self.keys:
            self.start(k)
        started = {k: time.time() for k in self.keys}
        stamp = code_stamp()
        while not self.stopping:
            time.sleep(1)
            now = time.time()
            if (new := code_stamp()) != stamp:
                stamp = new
                print("supervisor     | code or .env changed: restarting all agents", flush=True)
                self.stop_all()
                for k in self.keys:
                    self.start(k)
                    started[k] = time.time()
                continue
            for k in self.keys:
                p = self.procs.get(k)
                if p is not None and p.poll() is None:
                    if now - started[k] > 120:
                        self.backoff[k] = 2.0  # it stayed up: forget earlier crashes
                    continue
                if p is not None:  # it died
                    print(f"supervisor     | {k} exited with code {p.returncode}; restarting in "
                          f"{self.backoff[k]:.0f} s", flush=True)
                    self.procs.pop(k)
                    self.next_start[k] = now + self.backoff[k]
                    self.backoff[k] = min(self.backoff[k] * 2, 60)
                elif now >= self.next_start[k]:
                    self.start(k)
                    started[k] = now
        print("supervisor     | stopping agents…", flush=True)
        self.stop_all()


def code_stamp() -> tuple:
    root = Path(__file__).resolve().parent
    files = sorted((root / "dermis_agents").glob("*.py")) + [root / ".env", root / "run.py"]
    return tuple((str(f), f.stat().st_mtime) for f in files if f.exists())


if __name__ == "__main__":
    main()
