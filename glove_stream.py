#!/usr/bin/env python3
"""Live glove telemetry -> normalized finger bend angles (0.0 open .. 1.0 fully bent).

Sources (pick one):
    python glove_stream.py                                  # synthetic frames, no hardware
    python glove_stream.py --mock firmware/v1/examples/decoded_frame.json
    python glove_stream.py --port /dev/cu.usbmodem1101      # newline-delimited JSON frames
    python glove_stream.py --port /dev/ttys012 --wire       # Dermis Wire v1 binary (firmware/v1)

Every source yields decoded frames shaped like firmware/v1/schemas/decoded_frame.schema.json:
    {"pods": {"index_tip": {"imu": {"acc_mps2": [x, y, z]}}, ...}}
Only imu.acc_mps2 is used; mag is ignored.

Bend angle: with the finger still, the accelerometer measures gravity. The angle between the
current gravity vector and the one captured with the hand open (the rest vector) is how far
the pod has rotated: theta = acos(a . r / (|a||r|)). We divide by --full-bend-deg and clamp.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

FINGERS = ("index_tip", "thumb_tip")
DEFAULT_REST = (0.0, 0.0, 9.81)        # pod lying flat, z up
DEFAULT_FULL_BEND_DEG = 90.0
GRAVITY = 9.81
WIRE_DIR = Path(__file__).resolve().parent / "firmware" / "v1" / "host" / "python"


# --------------------------------------------------------------------------
# Parsing and trig
# --------------------------------------------------------------------------

def extract_acc(frame: dict, finger: str) -> Optional[tuple]:
    """acc_mps2 (x, y, z) for one pod, or None if missing or flagged invalid."""
    pods = frame.get("pods")
    if pods is None and isinstance(frame.get("frame"), dict):   # tolerate a {"frame": {...}} wrapper
        pods = frame["frame"].get("pods")
    imu = ((pods or {}).get(finger) or {}).get("imu") or {}
    acc = imu.get("acc_mps2")
    if not acc or imu.get("valid") is False or len(acc) != 3:
        return None
    return tuple(float(v) for v in acc)


def tilt_deg(acc: tuple, rest: tuple = DEFAULT_REST) -> float:
    """Angle in degrees between the measured gravity vector and the rest vector."""
    dot = sum(a * r for a, r in zip(acc, rest))
    norm = math.sqrt(sum(a * a for a in acc)) * math.sqrt(sum(r * r for r in rest))
    if norm < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / norm))))


def bend_angle(acc: tuple, rest: tuple = DEFAULT_REST, full_bend_deg: float = DEFAULT_FULL_BEND_DEG) -> float:
    """Normalized bend: 0.0 at the rest pose, 1.0 at full_bend_deg or beyond."""
    return max(0.0, min(1.0, tilt_deg(acc, rest) / full_bend_deg))


# --------------------------------------------------------------------------
# Frame sources (each is an iterator of decoded-frame dicts)
# --------------------------------------------------------------------------

def synthetic_frames(rate_hz: float = 60.0) -> Iterator[dict]:
    """Fake glove: index and thumb slowly pinch and release, with the frame format of the real thing."""
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        frame = {"frame": int(t * rate_hz), "pods": {}}
        for i, finger in enumerate(FINGERS):
            theta = math.radians(80.0) * (0.5 - 0.5 * math.cos(2 * math.pi * 0.2 * t + i * 0.3))
            acc = [0.0, GRAVITY * math.sin(theta), GRAVITY * math.cos(theta)]   # rotate about x
            frame["pods"][finger] = {"imu": {"acc_mps2": acc, "valid": True}}
        yield frame
        time.sleep(1.0 / rate_hz)


def mock_frames(path: str, rate_hz: float = 60.0) -> Iterator[dict]:
    """Replay a JSON file forever: one frame object, a list of frames, or one frame per line."""
    text = Path(path).read_text()
    try:
        data = json.loads(text)
        frames = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        frames = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not frames:
        raise ValueError(f"no frames in {path}")
    while True:
        for frame in frames:
            yield frame
            time.sleep(1.0 / rate_hz)


def _open_serial(port: str, baud: int):
    try:
        import serial  # pyserial
    except ImportError:
        sys.exit("--port needs pyserial:  pip install pyserial")
    return serial.Serial(port, baud, timeout=0.05)


def json_serial_frames(port: str, baud: int = 115200) -> Iterator[dict]:
    """Newline-delimited JSON frames. Non-JSON lines (boot logs etc.) are skipped."""
    ser = _open_serial(port, baud)
    try:
        while True:
            line = ser.readline().strip()
            if not line.startswith(b"{"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    finally:
        ser.close()


def wire_serial_frames(port: str, baud: int = 115200) -> Iterator[dict]:
    """Dermis Wire v1 binary packets, decoded with the reference decoder in firmware/v1/host/python."""
    sys.path.insert(0, str(WIRE_DIR))
    import dermis_wire as dw

    ser = _open_serial(port, baud)
    ser.write(b"\x00" + dw.cmd_get_info(0) + dw.cmd_start(1, 0))
    seq, info, dec = 2, None, dw.StreamDecoder()
    try:
        while True:
            for pkt in dec.feed(ser.read(4096)):
                if pkt.type == dw.Msg.DEVICE_INFO:
                    info = dw.DeviceInfo.from_packet(pkt)
                elif pkt.type == dw.Msg.FRAME and info is not None:
                    fr = dw.parse_frame(pkt)
                    if fr.config_id != info.config_id:      # config changed; ask again
                        info = None
                        ser.write(dw.cmd_get_info(seq))
                        seq += 1
                        continue
                    yield dw.decode_frame(fr, info, time.monotonic_ns())
    finally:
        ser.write(dw.cmd_stop(seq))
        ser.close()


# --------------------------------------------------------------------------
# Background reader
# --------------------------------------------------------------------------

class GloveStream:
    """Reads frames on a background thread; latest() returns the most recent bend angles."""

    def __init__(self, frames: Iterator[dict], full_bend_deg: float = DEFAULT_FULL_BEND_DEG):
        self._frames = frames
        self.full_bend_deg = full_bend_deg
        self.rest = {f: DEFAULT_REST for f in FINGERS}
        self._acc = {f: DEFAULT_REST for f in FINGERS}
        self._lock = threading.Lock()
        self.frames_read = 0
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "GloveStream":
        if args.port:
            frames = (wire_serial_frames if args.wire else json_serial_frames)(args.port, args.baud)
        elif args.mock:
            frames = mock_frames(args.mock)
        else:
            frames = synthetic_frames()
        return cls(frames, args.full_bend_deg)

    def start(self) -> "GloveStream":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            for frame in self._frames:
                accs = {f: extract_acc(frame, f) for f in FINGERS}
                with self._lock:
                    for f, acc in accs.items():
                        if acc is not None:
                            self._acc[f] = acc
                    self.frames_read += 1
        except BaseException as e:   # surface reader failures to the main thread
            self.error = e

    def calibrate(self) -> None:
        """Hold the hand open and call this: the current pose becomes bend 0.0."""
        with self._lock:
            self.rest = dict(self._acc)

    def latest(self) -> dict:
        """{"index_tip": 0..1, "thumb_tip": 0..1}"""
        with self._lock:
            return {f: bend_angle(self._acc[f], self.rest[f], self.full_bend_deg) for f in FINGERS}


def add_glove_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("glove source (default: synthetic)")
    g.add_argument("--port", help="serial port, e.g. /dev/cu.usbmodem1101")
    g.add_argument("--baud", type=int, default=115200)
    g.add_argument("--wire", action="store_true", help="port speaks Dermis Wire v1 binary, not JSON lines")
    g.add_argument("--mock", help="JSON file of decoded frames to replay")
    g.add_argument("--full-bend-deg", type=float, default=DEFAULT_FULL_BEND_DEG,
                   help="tilt from rest that counts as fully bent (default %(default)s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_glove_args(ap)
    ap.add_argument("--calibrate", action="store_true", help="treat the first second as the open-hand pose")
    args = ap.parse_args()

    glove = GloveStream.from_args(args).start()
    if args.calibrate:
        print("Hold your hand open...")
        time.sleep(1.0)
        glove.calibrate()
    try:
        while True:
            if glove.error:
                raise glove.error
            angles = glove.latest()
            print(f"frames={glove.frames_read:6d}  " +
                  "  ".join(f"{f}={a:.3f}" for f, a in angles.items()), end="\r", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
