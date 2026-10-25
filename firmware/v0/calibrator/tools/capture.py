#!/usr/bin/env python3
"""Capture Dermis calibrator CSV and summarize force/magnetic calibration data."""

import argparse
import csv
import math
import time
from pathlib import Path

import serial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--output", type=Path, default=Path("calibration.csv"))
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    rows = 0
    with serial.Serial(args.port, 115200, timeout=1) as device, args.output.open("w", newline="") as output:
        device.reset_input_buffer()
        device.write(b"csv\n")
        writer = None
        deadline = time.monotonic() + args.seconds
        try:
            while time.monotonic() < deadline:
                line = device.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("#BEGIN,"):
                    writer = csv.writer(output)
                    writer.writerow(line.removeprefix("#BEGIN,").split(","))
                elif line.startswith("#"):
                    print(line)
                elif writer:
                    fields = line.split(",")
                    if len(fields) == 25:
                        writer.writerow(fields)
                        rows += 1
        finally:
            device.write(b"stop\n")
    print(f"Saved {rows} synchronized samples to {args.output}")


if __name__ == "__main__":
    main()
