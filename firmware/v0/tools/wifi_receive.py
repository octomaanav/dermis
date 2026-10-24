#!/usr/bin/env python3
"""Receive Dermis frames from the glove's Wi-Fi WebSocket and optionally drive the simulator.

The ESP32 is a WebSocket *server* at ws://192.168.4.1:81/ws by default. Your Mac
is the client, so the glove never needs the Mac's address.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

try:  # direct script execution and package-style tests
    from .ble_receive import _is_decoded_v1, forward_tcp, to_decoded_frame
except ImportError:
    from ble_receive import _is_decoded_v1, forward_tcp, to_decoded_frame
from dermis.server.websocket_bridge import _frame, _open_websocket, _read_message

MAX_LINE_BYTES = 16_384


async def run(url: str, output: Path | None, tcp_host: str | None, tcp_port: int, print_frames: bool) -> None:
    stream = output.open("a", buffering=1) if output else None
    lines = asyncio.Queue(maxsize=1) if tcp_host else None
    forwarder = asyncio.create_task(forward_tcp(lines, tcp_host, tcp_port)) if lines else None
    frames, started = 0, time.monotonic()
    try:
        while True:
            writer = None
            try:
                print(f"Connecting to glove WebSocket at {url}…")
                reader, writer = await _open_websocket(url, [])
                print("Connected to glove WebSocket stream")
                while True:
                    opcode, raw = await _read_message(reader)
                    if opcode == 0x8:
                        raise ConnectionError("glove closed the WebSocket stream")
                    if opcode == 0x9:
                        writer.write(_frame(0xA, raw, mask=True))
                        await writer.drain()
                        continue
                    if opcode != 0x1:
                        if print_frames:
                            print(f"Received non-text WebSocket opcode {opcode} ({len(raw)} bytes); ignored")
                        continue
                    if len(raw) > MAX_LINE_BYTES:
                        print("Dropped oversized frame")
                        continue
                    if print_frames:
                        print(raw.decode("utf-8", errors="replace"))
                    try:
                        packet = json.loads(raw)
                        decoded = packet if _is_decoded_v1(packet) else to_decoded_frame(packet)
                    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                        print(f"Dropped malformed frame: {exc}")
                        continue
                    line = json.dumps(decoded, separators=(",", ":"))
                    if stream:
                        stream.write(line + "\n")
                    if lines is not None:
                        try:
                            lines.put_nowait((line + "\n").encode("utf-8"))
                        except asyncio.QueueFull:
                            lines.get_nowait()
                            lines.put_nowait((line + "\n").encode("utf-8"))
                    frames += 1
                    elapsed = time.monotonic() - started
                    if elapsed >= 1.0:
                        if not print_frames:
                            print(f"frame={decoded['frame']} rate={frames / elapsed:.1f} Hz", end="\r")
                        frames, started = 0, time.monotonic()
            except (ConnectionError, OSError, asyncio.IncompleteReadError, ValueError) as exc:
                print(f"Glove stream unavailable ({exc}); retrying in 1 s")
                await asyncio.sleep(1)
            finally:
                if writer is not None:
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except (ConnectionError, OSError):
                        pass
    finally:
        if forwarder:
            forwarder.cancel()
            try:
                await forwarder
            except asyncio.CancelledError:
                pass
        if stream:
            stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="ws://192.168.4.1:81/ws", help="glove WebSocket URL")
    parser.add_argument("--output", type=Path, help="optional decoded-frame v1 JSONL recording")
    parser.add_argument("--tcp-host", help="forward frames to the local Dermis TCP receiver")
    parser.add_argument("--tcp-port", type=int, default=8765)
    parser.add_argument("--print-frames", action="store_true", help="print every raw WebSocket text payload for debugging")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.url, args.output, args.tcp_host, args.tcp_port, args.print_frames))
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
