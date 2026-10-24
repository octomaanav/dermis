#!/usr/bin/env python3
"""Receive newline-delimited Dermis JSON frames over BLE on macOS."""

import argparse
import asyncio
import json
import socket
import time
from pathlib import Path

from bleak import BleakClient, BleakScanner

DEVICE_NAME = "Dermis-S3"
SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
V1_MAG_FLAGS = {"NOT_FOUND": "COMM_FAIL", "SPI_ERROR": "COMM_FAIL", "SATURATED": "RAIL"}
V1_IMU_FLAGS = {"NOT_FOUND": "COMM_FAIL", "SPI_ERROR": "COMM_FAIL"}


def _flags(values, allowed: dict[str, str]) -> list[str]:
    """Map v0 firmware quality flags to the decoded-frame v1 vocabulary."""
    return sorted({allowed[value] for value in values if value in allowed})


def to_decoded_frame(packet: dict) -> dict:
    """Adapt the v0 BLE packet to the TCP receiver's decoded-frame v1 contract.

    v0 has four magnetometer-only pods and one top-level IMU.  The receiving
    pipeline expects every sensor inside a pod, so the shared IMU becomes the
    ``wrist`` pod.  The v0 firmware has no clock-sync fit, therefore
    ``t_host_ns`` deliberately remains null.
    """
    pods = {}
    for loc, pod in packet.get("pods", {}).items():
        mag = pod.get("mag")
        if not isinstance(mag, dict):
            continue
        pods[loc] = {"pod_id": str(pod.get("pod_id", "")), "mag": {
            "ch": int(mag.get("ch", 0)), "b_ut": mag.get("b_ut"), "temp_c": mag.get("temp_c"),
            "t_us": int(mag.get("t_us", packet["t_device_us"])), "status": int(mag.get("status", 0)),
            "flags": _flags(mag.get("flags", []), V1_MAG_FLAGS), "valid": bool(mag.get("valid", False)),
        }, "imu": None}
    imu = packet.get("imu")
    if isinstance(imu, dict):
        pods["wrist"] = {"pod_id": str(imu.get("imu_id", "IMU-0001")), "mag": None, "imu": {
            "ch": int(imu.get("ch", 0)), "acc_mps2": imu.get("acc_mps2"), "gyr_rads": imu.get("gyr_rads"),
            "temp_c": imu.get("temp_c"), "t_us": int(imu.get("t_us", packet["t_device_us"])),
            "sensortime": int(imu.get("sensortime", 0)), "flags": _flags(imu.get("flags", []), V1_IMU_FLAGS),
            "valid": bool(imu.get("valid", False)),
        }}
    return {"device_id": str(packet["device_id"]), "frame": int(packet["frame"]),
            "t_device_us": int(packet["t_device_us"]), "t_host_ns": None,
            "config_id": 1, "flags": [], "pods": pods}


async def forward_tcp(lines: asyncio.Queue[bytes], host: str, port: int) -> None:
    """Forward the newest completed BLE frames to the local NDJSON receiver.

    TCP_NODELAY and one write+drain per frame avoid intentional batching.  The
    queue holds at most one frame: if the receiver is unavailable or slower
    than the glove, stale frames are discarded in favour of the current pose.
    """
    while True:
        writer = None
        try:
            _reader, writer = await asyncio.open_connection(host, port)
            sock = writer.transport.get_extra_info("socket")
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"Forwarding frames to TCP receiver at {host}:{port}")
            while True:
                writer.write(await lines.get())
                await writer.drain()
        except (ConnectionError, OSError) as exc:
            print(f"TCP receiver unavailable ({exc}); retrying in 1 s")
            await asyncio.sleep(1)
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except (ConnectionError, OSError):
                    pass


async def run(output: Path | None, tcp_host: str | None, tcp_port: int) -> None:
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
    tcp_lines = asyncio.Queue(maxsize=1) if tcp_host else None
    forwarder = asyncio.create_task(forward_tcp(tcp_lines, tcp_host, tcp_port)) if tcp_lines else None

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
        line = json.dumps(to_decoded_frame(packet), separators=(",", ":"))
        if stream:
            stream.write(line + "\n")
        if tcp_lines is not None:
            try:
                tcp_lines.put_nowait((line + "\n").encode("utf-8"))
            except asyncio.QueueFull:
                tcp_lines.get_nowait()       # live control wants the newest pose, never a stale backlog
                tcp_lines.put_nowait((line + "\n").encode("utf-8"))
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
        if forwarder:
            forwarder.cancel()
            try:
                await forwarder
            except asyncio.CancelledError:
                pass
        if stream:
            stream.close()


async def scan(seconds: float) -> None:
    """Print every BLE advertisement seen by macOS; useful before connecting."""
    print(f"Scanning for BLE advertisements for {seconds:g} seconds...")
    found = await BleakScanner.discover(timeout=seconds, return_adv=True)
    if not found:
        print("No BLE advertisements seen. Check macOS Bluetooth permissions and that the ESP32 is powered.")
        return
    for device, adv in found.values():
        services = ", ".join(adv.service_uuids or []) or "none"
        print(f"name={device.name or adv.local_name or '-'} address={device.address} services={services}")


def _is_decoded_v1(packet: dict) -> bool:
    """Whether a recording was already made by this tool's current v1 adapter."""
    if not isinstance(packet.get("config_id"), int) or not isinstance(packet.get("pods"), dict):
        return False
    return all(isinstance(pod, dict) and "mag" in pod and "imu" in pod for pod in packet["pods"].values())


def convert_recording(source: Path, output: Path) -> None:
    """Convert a v0 BLE JSONL capture to replayable decoded-frame v1 JSONL."""
    written = skipped = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as incoming, output.open("w", encoding="utf-8") as outgoing:
        for number, line in enumerate(incoming, 1):
            if not line.strip():
                continue
            try:
                packet = json.loads(line)
                decoded = packet if _is_decoded_v1(packet) else to_decoded_frame(packet)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                skipped += 1
                print(f"Skipped malformed input line {number}: {exc}")
                continue
            outgoing.write(json.dumps(decoded, separators=(",", ":")) + "\n")
            written += 1
    print(f"Converted {written} frames to {output}" + (f"; skipped {skipped} malformed lines" if skipped else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="optional JSONL output file")
    parser.add_argument("--tcp-host", help="forward decoded frames to this local TCP receiver, e.g. 127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=8765, help="TCP receiver port used with --tcp-host")
    parser.add_argument("--scan", action="store_true", help="list visible BLE advertisements and exit")
    parser.add_argument("--scan-seconds", type=float, default=10.0, help="duration for --scan")
    parser.add_argument("--convert", type=Path, metavar="INPUT.jsonl",
                        help="convert a recorded v0 BLE JSONL capture to v1 JSONL; requires --output")
    args = parser.parse_args()
    try:
        if args.convert:
            if not args.output:
                parser.error("--convert requires --output")
            convert_recording(args.convert, args.output)
        else:
            asyncio.run(scan(args.scan_seconds) if args.scan else run(args.output, args.tcp_host, args.tcp_port))
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
