"""Ask the agents from the terminal without any network: python -m dermis_agents.cli "your question"

Runs the hub and specialists in-process, so it's the fastest way to check answers before going live.
"""

import asyncio
import sys

from . import hub, specialist
from .profiles import BY_KEY


async def local_ask(key: str, text: str, parsed: dict, brief: bool) -> dict:
    return await specialist.respond(BY_KEY[key], text, parsed, brief)


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] in BY_KEY:  # talk to one specialist directly
        r = asyncio.run(specialist.respond(BY_KEY[args[0]], " ".join(args[1:]) or "help"))
        print(r["markdown"])
        return
    print(asyncio.run(hub.respond(" ".join(args) or "help", local_ask)))


if __name__ == "__main__":
    main()
