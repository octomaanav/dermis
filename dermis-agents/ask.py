"""Chat with a running agent over the real Agent Chat Protocol, offline.

  python run.py                                # terminal 1
  python ask.py "compare all applications on episode 4"            # terminal 2, talks to the hub
  python ask.py --to quadruped "check slip for the robot dog"      # talk to one specialist

This is the same ChatMessage flow ASI:One uses, routed straight to the local Bureau.
"""

import argparse
import json
import asyncio
import logging
import os

from uagents import Agent, Context
from uagents.resolver import RulesBasedResolver
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    MetadataContent,
    TextContent,
    chat_protocol_spec,
)
from uagents import Protocol

from dermis_agents import agents

ap = argparse.ArgumentParser()
ap.add_argument("text", nargs="+")
ap.add_argument("--to", default=agents.HUB_KEY, choices=agents.all_keys())
ap.add_argument("--timeout", type=float, default=120)
ap.add_argument("--port", type=int, default=8199, help="this client's own port")
args = ap.parse_args()

endpoint = f"http://127.0.0.1:{agents.port_for(args.to)}/submit"
target = agents.address_for(args.to)
client = Agent(
    name="dermis_cli",
    seed=f"{agents.seed_secret()}:dermis:cli",
    port=args.port,
    endpoint=[f"http://127.0.0.1:{args.port}/submit"],
    resolve=RulesBasedResolver({target: endpoint}),
    publish_agent_details=False,
    enable_agent_inspector=False,
    log_level=logging.WARNING,
)
logging.getLogger("uagents.registration").setLevel(logging.ERROR)
proto = Protocol(spec=chat_protocol_spec)


@client.on_event("startup")
async def send(ctx: Context):
    await ctx.send(target, ChatMessage(content=[TextContent(type="text", text=" ".join(args.text))]))

    async def give_up():
        await asyncio.sleep(args.timeout)
        print(f"No answer from {args.to} after {args.timeout:.0f} s. Is run.py running?")
        os._exit(1)

    asyncio.ensure_future(give_up())


@proto.on_message(ChatMessage)
async def on_reply(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(sender, ChatAcknowledgement(acknowledged_msg_id=msg.msg_id))
    print("\n" + "\n".join(c.text for c in msg.content if isinstance(c, TextContent)), flush=True)
    for c in msg.content:
        if isinstance(c, MetadataContent) and c.metadata.get("card_kind"):
            payload = json.loads(c.metadata["card_payload"])
            print(f"\n[card: {c.metadata['card_kind']} · {payload.get('title')}]", flush=True)
    os._exit(0)


@proto.on_message(ChatAcknowledgement)
async def on_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info("agent acknowledged, working…")


client.include(proto)

if __name__ == "__main__":
    client.run()
