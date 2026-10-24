#!/usr/bin/env python3
"""Mock glove: replays decoded_frame.json in a loop at 30 Hz, like a live stream.

    python mock_stream.py                       # ./decoded_frame.json, else firmware/v1/examples/
    python mock_stream.py path/to/frames.json   # one frame, a list of frames, or one frame per line

Parsing and the bend-angle trig live in glove_stream.py (extract_acc, bend_angle), so the mock,
the serial readers and the sim all compute angles the same way.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

from glove_stream import FINGERS, GloveStream, bend_angle, extract_acc, mock_frames

RATE_HZ = 30
HERE = Path(__file__).resolve().parent
CANDIDATES = (HERE / "decoded_frame.json", HERE / "firmware" / "v1" / "examples" / "decoded_frame.json")


def default_path() -> Path:
    for path in CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("no decoded_frame.json in " + ", ".join(str(c.parent) for c in CANDIDATES))


def open_stream(path: Optional[str] = None) -> GloveStream:
    """A GloveStream fed by the mock file; call .start(), then .latest() for {finger: 0..1}."""
    return GloveStream(mock_frames(str(path or default_path()), RATE_HZ))


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else default_path()
    print(f"Streaming {path} at {RATE_HZ} Hz (Ctrl-C to stop)")
    try:
        for n, frame in enumerate(mock_frames(str(path), RATE_HZ)):
            parts = []
            for finger in FINGERS:
                acc = extract_acc(frame, finger)
                parts.append(f"{finger} acc={acc} bend={bend_angle(acc):.3f}" if acc else f"{finger} missing")
            print(f"[{n:5d}] " + "  ".join(parts), end="\r", flush=True)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
