"""Agent Payment Protocol: the hub sells a certified readiness report for a small FET fee.

Flow (Fetch.ai's fet_direct pattern): the hub sends RequestPayment → the user's ASI:One wallet pays
and replies CommitPayment with the transaction hash → the hub checks that transaction on-chain with
cosmpy (recipient = this agent's wallet, amount ≥ price, right denom) → CompletePayment and the report,
or CancelPayment with the reason. Testnet by default, so demos cost nothing.
"""

from __future__ import annotations

import os
import uuid
from typing import Awaitable, Callable

from uagents import Context, Protocol
from uagents_core.contrib.protocols.chat import ChatMessage, TextContent
from uagents_core.contrib.protocols.payment import (
    CancelPayment,
    CommitPayment,
    CompletePayment,
    Funds,
    RejectPayment,
    RequestPayment,
    payment_protocol_spec,
)

PRICE_FET = os.getenv("DERMIS_REPORT_PRICE_FET", "0.1")
TESTNET = os.getenv("DERMIS_FET_TESTNET", "true").lower() == "true"
NETWORK = "stable-testnet" if TESTNET else "mainnet"
DENOM = "atestfet" if TESTNET else "afet"

_wallet = None


def set_wallet(wallet) -> None:
    global _wallet
    _wallet = wallet


def wallet_address() -> str | None:
    return str(_wallet.address()) if _wallet else None


def _ledger(testnet: bool = TESTNET):
    from cosmpy.aerial.client import LedgerClient, NetworkConfig
    return LedgerClient(NetworkConfig.fetchai_stable_testnet() if testnet else NetworkConfig.fetchai_mainnet())


def verify_payment(tx_hash: str, amount_fet: str, recipient: str, sender: str | None = None,
                   ledger_factory=_ledger) -> tuple[bool, str]:
    """Check a transfer of ≥ amount_fet to `recipient` (from `sender`, when known) in transaction tx_hash.

    Looks on the configured network first, then the other one: which network a buyer's wallet pays on
    isn't documented, and a payment shouldn't fail just because it landed on the other chain."""
    reasons = []
    for testnet in (TESTNET, not TESTNET):
        ok, why = _verify_on(tx_hash, amount_fet, recipient, sender, ledger_factory, testnet)
        if ok:
            return True, f"verified on {'stable-testnet' if testnet else 'mainnet'}"
        reasons.append(f"{'testnet' if testnet else 'mainnet'}: {why}")
    return False, "; ".join(reasons)


def _verify_on(tx_hash, amount_fet, recipient, sender, ledger_factory, testnet: bool) -> tuple[bool, str]:
    denom = "atestfet" if testnet else "afet"
    try:
        expected = int(round(float(amount_fet) * 10**18))
        tx = ledger_factory(testnet).query_tx(tx_hash)
    except Exception as e:
        return False, f"transaction {tx_hash[:16]}… not found ({type(e).__name__})"
    if not tx.is_successful():
        return False, "the transaction failed on-chain"
    transfer = tx.events.get("transfer") or {}
    if transfer.get("recipient") != recipient:
        return False, "the transaction didn't pay this agent's wallet"
    if sender and transfer.get("sender") != sender:
        return False, "the transaction came from a different wallet"
    amount = str(transfer.get("amount", ""))
    if not amount.endswith(denom):
        return False, f"the payment wasn't in {denom}"
    try:
        paid = int(amount[: -len(denom)])
    except ValueError:
        return False, "couldn't read the paid amount"
    if paid < expected:
        return False, f"paid {paid / 10**18:g} FET, the price is {amount_fet} FET"
    return True, "verified"


async def request_payment(ctx: Context, buyer: str, episode: str) -> bool:
    recipient = wallet_address()
    if not recipient:
        return False
    reference = uuid.uuid4().hex
    ctx.storage.set(f"purchase:{buyer}", {"episode": episode, "reference": reference})
    await ctx.send(buyer, RequestPayment(
        accepted_funds=[Funds(currency="FET", amount=PRICE_FET, payment_method="fet_direct")],
        recipient=recipient,
        deadline_seconds=300,
        reference=reference,
        description=f"Dermis certified readiness report for {episode}",
        metadata={"service": "certified_readiness_report", "episode": episode, "fet_network": NETWORK,
                  "mainnet": "false" if TESTNET else "true", "provider_agent_wallet": recipient,
                  "content": f"Pay {PRICE_FET} FET for the certified readiness report on {episode}."},
    ))
    ctx.logger.info(f"[payment] RequestPayment {PRICE_FET} FET to {buyer[:16]}… for {episode}")
    return True


def payment_protocol(deliver: Callable[[str, str], Awaitable[str]]) -> Protocol:
    """`deliver(episode, receipt)` returns the paid report's markdown."""
    proto = Protocol(spec=payment_protocol_spec, role="seller")

    async def say(ctx: Context, to: str, text: str) -> None:
        await ctx.send(to, ChatMessage(content=[TextContent(type="text", text=text)]))

    @proto.on_message(CommitPayment)
    async def on_commit(ctx: Context, sender: str, msg: CommitPayment):
        pending = ctx.storage.get(f"purchase:{sender}") or {}
        episode = pending.get("episode") or (msg.metadata or {}).get("episode")
        if not episode:
            await ctx.send(sender, CancelPayment(transaction_id=msg.transaction_id, reason="no pending purchase"))
            return
        if ctx.storage.get(f"paid_tx:{msg.transaction_id}"):
            await ctx.send(sender, CancelPayment(transaction_id=msg.transaction_id, reason="transaction already used"))
            return
        buyer_wallet = None
        if isinstance(msg.metadata, dict):
            buyer_wallet = msg.metadata.get("buyer_fet_wallet") or msg.metadata.get("buyer_fet_address")
        ctx.logger.info(f"[payment] CommitPayment tx={msg.transaction_id} funds={msg.funds.amount} {msg.funds.currency} "
                        f"method={msg.funds.payment_method} buyer_wallet={buyer_wallet} metadata={msg.metadata}")
        ok, reason = verify_payment(msg.transaction_id, msg.funds.amount or PRICE_FET, wallet_address() or "",
                                    buyer_wallet if isinstance(buyer_wallet, str) else None)
        ctx.logger.info(f"[payment] verification: {ok} ({reason})")
        if not ok:
            ctx.logger.warning(f"[payment] verification failed: {reason}")
            await ctx.send(sender, CancelPayment(transaction_id=msg.transaction_id, reason=reason))
            await say(ctx, sender, f"Payment not accepted: {reason}. You haven't been charged for a report; "
                                   "try again or ask me anything else.")
            return
        ctx.storage.set(f"paid_tx:{msg.transaction_id}", True)
        ctx.storage.set(f"purchase:{sender}", None)
        await ctx.send(sender, CompletePayment(transaction_id=msg.transaction_id))
        receipt = f"{msg.funds.amount} {msg.funds.currency} · tx {msg.transaction_id} · {NETWORK}"
        await say(ctx, sender, await deliver(episode, receipt))

    @proto.on_message(RejectPayment)
    async def on_reject(ctx: Context, sender: str, msg: RejectPayment):
        ctx.logger.info(f"[payment] RejectPayment from {sender[:16]}…: {msg.reason}")
        ctx.storage.set(f"purchase:{sender}", None)
        await say(ctx, sender, "No problem, the purchase is cancelled. Everything else stays free to ask.")

    return proto
