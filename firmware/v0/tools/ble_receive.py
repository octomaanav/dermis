#!/usr/bin/env python3
"""Receive newline-delimited Dermis JSON frames over BLE on macOS."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from bleak import BleakClient, BleakScanner

DEVICE_NAME = "Dermis-S3"
SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"


async def run(output: Path | None) -> None:
    print(f"Scanning for {DEVICE_NAME}...")
    device = await BleakScanner.find_device_by_filter(
        lambda dev, adv: dev.name == DEVICE_NAME
        or SERVICE_UUID in [uuid.lower() for uuid in adv.service_uuids],
        timeout=20.0,
    )
    if device is None:
        raise RuntimeError("Dermis-S3 not found; check power and macOS Bluetooth")

    pending: dict[int, tuple[int, dict[int, bytes]]] = {}
    frames = 0
    started = time.monotonic()
    stream = output.open("a", buffering=1) if output else None

    def notification(_sender, data: bytearray) -> None:
        nonlocal frames, started
        if len(data) < 8 or data[0] != 0xD3 or data[1] != 1:
            return
        frame_id = int.from_bytes(data[2:6], "little")
        index, count = data[6], data[7]
        expected, chunks = pending.setdefault(frame_id, (count, {}))
        chunks[index] = bytes(data[8:])
        for old in [key for key in pending if key + 20 < frame_id]:
            pending.pop(old, None)
        if len(chunks) != expected:
            return
        raw = b"".join(chunks[i] for i in range(expected))
        pending.pop(frame_id, None)
        try:
            packet = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            print(f"Dropped corrupt frame {frame_id}")
            return
        packet["t_host_ns"] = time.time_ns()
        line = json.dumps(packet, separators=(",", ":"))
        if stream:
            stream.write(line + "\n")
        frames += 1
        elapsed = time.monotonic() - started
        if elapsed >= 1.0:
            print(f"frame={packet['frame']} rate={frames / elapsed:.1f} Hz", end="\r")
            frames = 0
            started = time.monotonic()

    try:
        async with BleakClient(device) as client:
            print(f"Connected to {device.name}")
            await client.start_notify(TX_UUID, notification)
            while client.is_connected:
                await asyncio.sleep(1)
    finally:
        if stream:
            stream.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="optional JSONL output file")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.output))
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
