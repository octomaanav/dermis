#!/usr/bin/env python3
"""Log Dermis firmware CSV frames and report rate/drop information."""

from __future__ import annotations

import argparse
import csv
import signal
import sys
from collections import deque
from pathlib import Path

import serial

from protocol import BAUD, HEADER, estimate_dropped, parse_frame


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="serial port, e.g. /dev/cu.usbserial-0001")
    parser.add_argument("--out", required=True, type=Path, help="output CSV path")
    parser.add_argument("--baud", default=BAUD, type=int)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    timestamps: deque[int] = deque(maxlen=10_000)
    total_frames = 0
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with serial.Serial(args.port, args.baud, timeout=0.25) as port, args.out.open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(HEADER)
        port.reset_input_buffer()
        port.write(b"s")
        port.flush()
        print(f"Logging {args.port} at {args.baud} baud to {args.out}; Ctrl-C stops cleanly.")

        while not stopping:
            raw = port.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("#RATE"):
                print(line)
                continue
            frame = parse_frame(line)
            if frame is None:
                continue
            writer.writerow(frame.csv_row())
            timestamps.append(frame.t_us)
            total_frames += 1
            if total_frames % 200 == 0:
                output.flush()

        port.write(b"\n")
        port.flush()

    dropped = estimate_dropped(timestamps)
    denominator = total_frames + dropped
    fraction = (100.0 * dropped / denominator) if denominator else 0.0
    print(f"Saved {total_frames} frames; estimated dropped={dropped} ({fraction:.3f}%).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

