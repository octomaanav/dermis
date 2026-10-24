#!/usr/bin/env python3
"""A small live terminal "OUCH!" demo driven by fingertip magnetic deformation.

The MLX readings are not calibrated force or pressure measurements.  This tool
uses change from an unloaded magnetic baseline as a deliberately playful demo
signal: it alerts when that change exceeds ``--threshold-ut``.

Examples:
    # Terminal 1: keep recording your live BLE stream
    python tools/ble_receive.py --output data/ble_frames.jsonl

    # Terminal 2: hold the finger unloaded for the first 25 frames, then demo
    python tools/ouch.py --file data/ble_frames.jsonl --follow --finger index_tip --threshold-ut 30
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path


def vector_from_frame(frame: dict, finger: str) -> tuple[float, float, float] | None:
    """Read a valid magnetometer vector from either v0 or decoded-frame v1 JSON."""
    mag = frame.get("pods", {}).get(finger, {}).get("mag")
    if not isinstance(mag, dict) or not mag.get("valid"):
        return None
    values = mag.get("b_ut")
    if not isinstance(values, list) or len(values) != 3:
        return None
    try:
        out = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(value) for value in out) else None


def baseline(vectors: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return tuple(statistics.median(vector[axis] for vector in vectors) for axis in range(3))


def deformation(vector: tuple[float, float, float], reference: tuple[float, float, float]) -> float:
    return math.sqrt(sum((value - ref) ** 2 for value, ref in zip(vector, reference)))


def render(finger: str, delta_ut: float | None, threshold_ut: float, active: bool) -> None:
    clear = "\033[2J\033[H"
    reset, red, green, yellow, bold = "\033[0m", "\033[31m", "\033[32m", "\033[33m", "\033[1m"
    print(clear, end="")
    print(f"{bold}DERMIS FINGERTIP DEMO{reset}  ·  {finger}")
    print("Magnetic deformation relative to unloaded baseline — not calibrated pressure.\n")
    if delta_ut is None:
        print(f"{yellow}Waiting for a valid {finger} reading…{reset}")
        return
    color = red if active else green
    word = "OUCH!" if active else "comfortable"
    print(f"{color}{bold}      {word:^18}      {reset}\n")
    print(f"deformation  {delta_ut:8.0f} µT")
    print(f"threshold    {threshold_ut:8.0f} µT")
    print(f"status       {color}{'ABOVE THRESHOLD' if active else 'below threshold'}{reset}")


def follow_lines(path: Path, follow: bool):
    """Yield appended lines.  In --follow mode wait for a file that does not exist yet."""
    while True:
        try:
            with path.open(encoding="utf-8") as stream:
                while True:
                    line = stream.readline()
                    if line:
                        yield line
                    elif follow:
                        time.sleep(0.05)
                    else:
                        return
        except FileNotFoundError:
            if not follow:
                raise
            time.sleep(0.25)


def run(path: Path, follow: bool, finger: str, baseline_frames: int, threshold_ut: float) -> None:
    samples: list[tuple[float, float, float]] = []
    reference = None
    last_active = False
    print(f"Hold {finger} unloaded: collecting {baseline_frames} valid baseline frames…", flush=True)
    for raw in follow_lines(path, follow):
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            continue
        vector = vector_from_frame(frame, finger)
        if vector is None:
            if reference is not None:
                render(finger, None, threshold_ut, False)
            continue
        if reference is None:
            samples.append(vector)
            if len(samples) < baseline_frames:
                print(f"baseline {len(samples)}/{baseline_frames}", end="\r", flush=True)
                continue
            reference = baseline(samples)
            print("\nBaseline captured. Press/release the fingertip to trigger the display.")
        delta_ut = deformation(vector, reference)
        active = delta_ut >= threshold_ut
        render(finger, delta_ut, threshold_ut, active)
        if active and not last_active:
            print("\a", end="", flush=True)  # terminal bell when crossing the threshold
        last_active = active


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, required=True, help="BLE JSONL recording to read")
    parser.add_argument("--follow", action="store_true", help="wait for and continuously read appended frames")
    parser.add_argument("--finger", default="index_tip", help="pod location to monitor")
    parser.add_argument("--baseline-frames", type=int, default=25, help="unloaded valid frames to median for baseline")
    parser.add_argument("--threshold-ut", type=float, default=100.0, help="deformation alert threshold in microtesla")
    args = parser.parse_args()
    if args.baseline_frames < 1 or args.threshold_ut <= 0:
        parser.error("--baseline-frames and --threshold-ut must be positive")
    try:
        run(args.file, args.follow, args.finger, args.baseline_frames, args.threshold_ut)
    except KeyboardInterrupt:
        print("\nOUCH demo stopped")


if __name__ == "__main__":
    main()
