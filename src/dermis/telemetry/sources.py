"""Interchangeable telemetry sources. Every source yields RawMessage(decoded-frame dict, host receive
time) and carries a provenance label; all of them feed the same TelemetryPipeline.

    ReplaySource       saved decoded frames (.json object/list, or .jsonl)            label REPLAY
    SyntheticSource    generated frames with optional injected faults               label SYNTHETIC
    LineSource         live JSONL on any text stream (stdin)                          label GLOVE
    TcpSource          live JSONL over TCP (listens; one client at a time)            label GLOVE
    WireFileSource     Dermis Wire v1 binary capture, via the reference decoder       label REPLAY
    SerialSource       live serial (JSON lines or Dermis Wire v1)                  label GLOVE

SYNTHETIC frames exist for software tests only; they are not evidence of anything about real data.
"""
from __future__ import annotations

import json
import math
import socket
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from .pipeline import RawMessage

REPO = Path(__file__).resolve().parents[3]
WIRE_DIR = REPO / "firmware" / "v1" / "host" / "python"


def _parse_line(line: str) -> RawMessage:
    t = time.monotonic_ns()
    line = line.strip()
    try:
        obj = json.loads(line)
        if not isinstance(obj, dict):
            return RawMessage(None, t, "not a JSON object")
        return RawMessage(obj, t)
    except json.JSONDecodeError as e:
        return RawMessage(None, t, f"bad JSON: {e.msg}")


def load_frames(path: str | Path) -> list[dict]:
    text = Path(path).read_text()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(l) for l in text.splitlines() if l.strip()]


class ReplaySource:
    label = "REPLAY"

    def __init__(self, path: str, rate_hz: float | None = None, loop: bool = False, realtime: bool = True):
        self.path, self.rate_hz, self.loop, self.realtime = path, rate_hz, loop, realtime
        self.frames = load_frames(path)

    def __iter__(self) -> Iterator[RawMessage]:
        while True:
            prev_t = None
            for f in self.frames:
                if self.realtime:
                    if self.rate_hz:
                        time.sleep(1.0 / self.rate_hz)
                    elif prev_t is not None and isinstance(f.get("t_device_us"), int):
                        time.sleep(min(1.0, max(0.0, (f["t_device_us"] - prev_t) / 1e6)))
                    prev_t = f.get("t_device_us") if isinstance(f.get("t_device_us"), int) else prev_t
                yield RawMessage(f, time.monotonic_ns())
            if not self.loop:
                return


class LineSource:
    label = "GLOVE"

    def __init__(self, stream=None):
        self.stream = stream or sys.stdin

    def __iter__(self) -> Iterator[RawMessage]:
        for line in self.stream:
            if line.strip():
                yield _parse_line(line)


class TcpSource:
    """Listens on host:port and reads newline-delimited decoded frames from each client in turn."""
    label = "GLOVE"

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, once: bool = False):
        self.host, self.port, self.once = host, port, once
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]

    def __iter__(self) -> Iterator[RawMessage]:
        try:
            while True:
                conn, _ = self.sock.accept()
                with conn, conn.makefile("r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if line.strip():
                            yield _parse_line(line)
                if self.once:
                    return
        finally:
            self.sock.close()


def _wire():
    if str(WIRE_DIR) not in sys.path:
        sys.path.insert(0, str(WIRE_DIR))
    import dermis_wire
    return dermis_wire


class WireFileSource:
    """Binary capture decoded with the reference decoder (firmware/v1/host/python/dermis_wire.py).
    Frames whose config_id does not match the last DEVICE_INFO are held back, per SPEC section 10."""
    label = "REPLAY"

    def __init__(self, path: str, realtime: bool = False):
        self.path, self.realtime = path, realtime
        self.device_info: dict | None = None
        self.held_frames = 0

    def __iter__(self) -> Iterator[RawMessage]:
        dw = _wire()
        dec, info = dw.StreamDecoder(), None
        prev = None
        with open(self.path, "rb") as fh:
            while chunk := fh.read(4096):
                for pkt in dec.feed(chunk):
                    if pkt.type == dw.Msg.DEVICE_INFO:
                        info = dw.DeviceInfo.from_packet(pkt)
                        self.device_info = info.raw
                    elif pkt.type == dw.Msg.FRAME:
                        fr = dw.parse_frame(pkt)
                        if info is None or fr.config_id != info.config_id:
                            self.held_frames += 1
                            continue
                        if self.realtime and prev is not None:
                            time.sleep(max(0.0, min(0.1, (fr.t_us - prev) / 1e6)))
                        prev = fr.t_us
                        yield RawMessage(dw.decode_frame(fr, info, None), time.monotonic_ns())
        self.stream_errors = dict(dec.errors)


class SerialSource:
    """Live hub over USB serial: one decoded JSON frame per line, or Dermis Wire v1 binary (wire=True).
    Unverified without hardware."""
    label = "GLOVE"

    def __init__(self, port: str, wire: bool = False, baud: int = 115200):
        self.port, self.wire, self.baud = port, wire, baud

    def __iter__(self) -> Iterator[RawMessage]:
        gen = self._wire_frames() if self.wire else self._json_frames()
        for frame in gen:
            yield RawMessage(frame, time.monotonic_ns())

    def _open(self):
        try:
            import serial  # pyserial
        except ImportError:
            raise RuntimeError("--source serial needs pyserial:  pip install -e '.[serial]'") from None
        return serial.Serial(self.port, self.baud, timeout=0.05)

    def _json_frames(self) -> Iterator[dict]:
        """Newline-delimited JSON frames. Non-JSON lines (boot logs etc.) are skipped."""
        ser = self._open()
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

    def _wire_frames(self) -> Iterator[dict]:
        """Dermis Wire v1 packets decoded with the reference decoder. t_host_ns stays None: no clock-sync fit."""
        dw = _wire()
        ser = self._open()
        ser.write(b"\x00" + dw.cmd_get_info(0) + dw.cmd_start(1, 0))
        seq, info, dec = 2, None, dw.StreamDecoder()
        try:
            while True:
                for pkt in dec.feed(ser.read(4096)):
                    if pkt.type == dw.Msg.DEVICE_INFO:
                        info = dw.DeviceInfo.from_packet(pkt)
                    elif pkt.type == dw.Msg.FRAME and info is not None:
                        fr = dw.parse_frame(pkt)
                        if fr.config_id != info.config_id:      # config changed; ask again (SPEC section 10)
                            info = None
                            ser.write(dw.cmd_get_info(seq))
                            seq += 1
                            continue
                        yield dw.decode_frame(fr, info, None)
        finally:
            ser.write(dw.cmd_stop(seq))
            ser.close()


# --- synthetic -------------------------------------------------------------------------

FAULTS = ("drop", "duplicate", "reorder", "stale", "temp_stale", "invalid", "missing_pod", "config_change",
          "gap", "null_valid", "restart")
BASELINE_UT = {"index_tip": (120.0, -280.0, 27000.0), "thumb_tip": (-500.0, 150.0, 27500.0),
               "middle_tip": (40.0, -80.0, 26500.0), "thenar_apb": (10.0, 60.0, 26000.0)}


class SyntheticSource:
    """SYNTHETIC decoded frames for integration tests. Fingertips 'press' periodically (|B| change of
    a few hundred uT); the thenar IMU tilts slowly. Faults are injected at fixed frame indices
    (every `fault_period` frames, offset per fault) so tests are deterministic."""
    label = "SYNTHETIC"

    def __init__(self, rate_hz: float = 100.0, faults=(), n_frames: int | None = None, realtime: bool = True,
                 seed: int = 0, fault_period: int = 50, pods=tuple(BASELINE_UT), device_id: str = "synthetic-000000",
                 profile: str = "periodic"):
        """profile: "periodic" = continuous sin^2 fingertip presses (default); "step" = repeating
        3 s unloaded / 3 s *held* press / 3 s released, for exercising two-point calibration."""
        if profile not in ("periodic", "step"):
            raise ValueError(f"unknown synthetic profile {profile!r}")
        self.profile = profile
        bad = set(faults) - set(FAULTS)
        if bad:
            raise ValueError(f"unknown faults {sorted(bad)}; known: {FAULTS}")
        self.rate_hz, self.faults, self.n_frames, self.realtime = rate_hz, set(faults), n_frames, realtime
        self.rng = np.random.default_rng(seed)
        self.fault_period, self.pods, self.device_id = fault_period, tuple(pods), device_id

    def _hit(self, k: int, fault: str) -> bool:
        return fault in self.faults and k % self.fault_period == (FAULTS.index(fault) * 3 + 7) % self.fault_period

    def frame(self, k: int, counter: int, config_id: int, flags: list) -> dict:
        t_us = 1_000_000 + int(k * 1e6 / self.rate_hz)
        t = k / self.rate_hz
        pods = {}
        for i, loc in enumerate(self.pods):
            if not loc.endswith("_tip"):
                press = 0.0
            elif self.profile == "step":
                ph = t % 9.0
                press = 0.0 if ph < 3.0 or ph >= 6.0 else min(1.0, (ph - 3.0) / 0.3)
            else:
                press = max(0.0, math.sin(2 * math.pi * 0.25 * t - i * 0.6)) ** 2
            b0 = np.array(BASELINE_UT.get(loc, (0.0, 0.0, 26000.0)))
            b = b0 + press * np.array([60.0, -40.0, 400.0]) + self.rng.normal(0, 2.5, 3)
            tilt = 0.4 * math.sin(2 * math.pi * 0.1 * t) if loc == "thenar_apb" else 0.0
            acc = np.array([0.0, 9.81 * math.sin(tilt), 9.81 * math.cos(tilt)]) + self.rng.normal(0, 0.02, 3)
            gyr = self.rng.normal(0, 0.005, 3)
            mag = {"ch": i, "b_ut": [round(float(v), 3) for v in b], "temp_c": 22.0, "t_us": t_us + 672,
                   "status": 3, "flags": [], "valid": True}
            imu = {"ch": i, "acc_mps2": [round(float(v), 5) for v in acc], "gyr_rads": [round(float(v), 5) for v in gyr],
                   "temp_c": 24.0, "t_us": t_us + 100 * i, "sensortime": (t_us // 39) & 0xFFFFFF, "flags": [], "valid": True}
            pods[loc] = {"pod_id": f"SYN-{i:04d}", "mag": mag, "imu": imu}
        return {"device_id": self.device_id, "frame": counter, "t_device_us": t_us, "t_host_ns": None,
                "config_id": config_id, "flags": flags, "pods": pods}

    def frames(self) -> Iterator[dict]:
        """Decoded-frame dicts in delivery order (faults applied)."""
        counter, config_id, k = 1000, 3, 0
        last_mag = {}
        held_back = None
        while self.n_frames is None or k < self.n_frames:
            flags = []
            if self._hit(k, "config_change"):
                config_id += 1
                flags.append("CONFIG_CHANGED")
            if self._hit(k, "drop"):
                counter += 1                                   # one counter value never delivered
            if self._hit(k, "gap"):
                counter += 25
            if self._hit(k, "restart"):
                counter = 0
            fr = self.frame(k, counter, config_id, flags)
            first = self.pods[0]
            if self._hit(k, "stale") and first in last_mag:     # repeated mag + imu values, valid=true
                fr["pods"][first]["mag"] = {**last_mag[first], "flags": ["STALE"]}
                fr["pods"][first]["imu"]["flags"] = ["STALE"]
            if self._hit(k, "temp_stale"):
                fr["pods"][first]["mag"]["flags"] = ["TEMP_STALE"]
            if self._hit(k, "invalid") and len(self.pods) > 1:
                m = fr["pods"][self.pods[1]]["mag"]
                m.update(b_ut=None, temp_c=None, status=1, flags=["BAD_LENGTH"], valid=False)
            if self._hit(k, "null_valid") and len(self.pods) > 2:   # malformed: valid=true but null value
                fr["pods"][self.pods[2]]["mag"]["b_ut"] = None
            if self._hit(k, "missing_pod"):
                fr["pods"].pop(self.pods[-1], None)
            for loc, pod in fr["pods"].items():
                if pod["mag"] and pod["mag"]["valid"] and "STALE" not in pod["mag"]["flags"]:
                    last_mag[loc] = dict(pod["mag"])
            k += 1
            counter = (counter + 1) % (1 << 32)
            if self._hit(k - 1, "reorder"):
                held_back = fr                                 # deliver after the next frame
                continue
            yield fr
            if self._hit(k - 1, "duplicate"):
                yield json.loads(json.dumps(fr))
            if held_back is not None:
                yield held_back
                held_back = None

    def __iter__(self) -> Iterator[RawMessage]:
        period = 1.0 / self.rate_hz
        t0 = time.monotonic()
        for k, fr in enumerate(self.frames(), start=1):
            if self.realtime:                            # deadline schedule: generation time does not accumulate
                time.sleep(max(0.0, t0 + k * period - time.monotonic()))
            yield RawMessage(fr, time.monotonic_ns())


def open_source(kind: str, path=None, rate=None, loop=False, host="127.0.0.1", port=8765, serial_port=None,
                wire=False, faults="", realtime=True, profile="periodic"):
    if kind == "replay":
        if not path:
            raise SystemExit("--source replay needs --path")
        return ReplaySource(path, rate, loop, realtime)
    if kind == "synthetic":
        return SyntheticSource(rate or 100.0, [f for f in faults.split(",") if f], realtime=realtime, profile=profile)
    if kind == "stdin":
        return LineSource(sys.stdin)
    if kind == "tcp":
        return TcpSource(host, port)
    if kind == "wire-file":
        if not path:
            raise SystemExit("--source wire-file needs --path")
        return WireFileSource(path, realtime)
    if kind == "serial":
        if not serial_port:
            raise SystemExit("--source serial needs --serial-port")
        return SerialSource(serial_port, wire)
    raise SystemExit(f"unknown source {kind}")


def find_fixture(name: str) -> Path | None:
    """Locate a supplied protocol fixture anywhere in the repo (not a hardcoded upload path)."""
    for p in sorted(REPO.rglob(name)):
        if ".venv" not in p.parts and "node_modules" not in p.parts:
            return p
    return None
