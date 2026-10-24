#!/usr/bin/env python3
"""Reference implementation of the Dermis Wire Schema v1.0.

Pure Python 3.9+, standard library only (pyserial is needed only for
`dump --port`). The firmware side lives in ../../firmware/dermis_wire.h;
both must produce the same bytes for the vectors in ../../vectors/.

Commands
    python dermis_wire.py vectors > test_vectors.json      # rebuild test vectors
    python dermis_wire.py check-vectors c_output.txt       # compare "name hex" lines (from test_vectors.c)
    python dermis_wire.py simulate --seconds 2 --out capture.bin
    python dermis_wire.py dump --file capture.bin [--csv out.csv]
    python dermis_wire.py dump --port /dev/cu.usbmodem1101 [--csv out.csv]
    python dermis_wire.py serve-pty                        # fake hub on a virtual serial port (macOS/Linux)
"""
from __future__ import annotations

import argparse
import copy
import csv
import enum
import json
import math
import os
import random
import select
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional

# --------------------------------------------------------------------------
# Constants (must match SPEC.md and dermis_wire.h)
# --------------------------------------------------------------------------

PROTO_MAJOR = 1
PROTO_MINOR = 0
PROTO_VERSION = (PROTO_MAJOR << 4) | PROTO_MINOR  # 0x10

HEADER = struct.Struct("<BBHIQ")              # type, version, payload_len, seq, t_us   (16 B)
FRAME_HEADER = struct.Struct("<IHBBBBBBHH")   # frame_counter ... reserved1           (16 B)
MAG_REC = struct.Struct("<BBBBhhhH")          # ch, mlx_status, flags, res, x, y, z, t_raw (12 B)
IMU_REC = struct.Struct("<BBh3h3hhHI")        # ch, flags, t_rel_us, acc[3], gyr[3], temp, res, sensortime (24 B)
ACK = struct.Struct("<IBBH")                  # cmd_seq, cmd_type, status, reserved     (8 B)
EVENT = struct.Struct("<HBB")                 # code, severity, ch                      (4 B)
SYNC_PONG = struct.Struct("<QQ")              # host_time (echo), t_rx_us              (16 B)
CMD_START = struct.Struct("<HH")              # rate_hz, reserved                       (4 B)
CMD_SYNC_PING = struct.Struct("<Q")           # host_time                               (8 B)
CMD_RESET = struct.Struct("<BB")              # kind, ch                                (2 B)
CRC = struct.Struct("<I")

HEADER_SIZE = HEADER.size                     # 16
CRC_SIZE = CRC.size                           # 4
MAX_PACKET = 8192                             # decoded bytes: header + payload + CRC
MAX_PAYLOAD = MAX_PACKET - HEADER_SIZE - CRC_SIZE
MAX_ENCODED = MAX_PACKET + MAX_PACKET // 254 + 1

assert (HEADER.size, FRAME_HEADER.size, MAG_REC.size, IMU_REC.size) == (16, 16, 12, 24)


class Msg(enum.IntEnum):
    # device -> host
    DEVICE_INFO = 0x01
    STATS = 0x02
    EVENT = 0x03
    SYNC_PONG = 0x04
    ACK = 0x05
    LOG = 0x06
    FRAME = 0x10
    IMU_BURST = 0x11   # reserved (v1.1)
    FORCE_REF = 0x12   # reserved
    ENV = 0x13         # reserved
    # host -> device
    GET_INFO = 0x80
    START = 0x81
    STOP = 0x82
    SET_CONFIG = 0x83
    SYNC_PING = 0x84
    RESET_SENSOR = 0x85
    SELFTEST = 0x86
    SET_MODE = 0x87
    REBOOT = 0x88


class AckStatus(enum.IntEnum):
    OK = 0
    UNKNOWN_TYPE = 1
    BAD_PAYLOAD = 2
    BAD_STATE = 3
    FAILED = 4
    VERSION = 5


class EventCode(enum.IntEnum):
    BOOT = 0x0001
    STREAM_STARTED = 0x0002
    STREAM_STOPPED = 0x0003
    CONFIG_CHANGED = 0x0004
    MARKER = 0x0005
    MAG_RESET = 0x0010
    MAG_FAULT = 0x0011
    IMU_RESET = 0x0020
    IMU_FAULT = 0x0021
    TX_OVERFLOW = 0x0030
    RATE_NOT_MET = 0x0031
    SELFTEST_RESULT = 0x0040


SEVERITY = {0: "info", 1: "warning", 2: "error"}
CH_NONE = 0xFF

FRAME_FLAGS = {0x01: "CONFIG_CHANGED", 0x02: "OVERRUN", 0x04: "TX_DROP", 0x08: "MARKER"}
MAG_FLAGS = {0x01: "STALE", 0x02: "BAD_LENGTH", 0x04: "CHIP_ERROR", 0x08: "RAIL",
             0x10: "COMM_FAIL", 0x20: "TEMP_STALE", 0x40: "RECOVERED"}
IMU_FLAGS = {0x01: "STALE", 0x02: "COMM_FAIL", 0x04: "ACC_RAIL", 0x08: "GYR_RAIL",
             0x10: "TEMP_STALE", 0x20: "RECOVERED"}
MAG_INVALID_MASK = 0x02 | 0x04 | 0x10
IMU_INVALID_MASK = 0x02

LOCATIONS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip", "thenar_apb",
             "hypothenar", "palm_center", "hand_dorsum", "wrist", "bench", "unassigned")

G = 9.80665
DEG = math.pi / 180.0


def flag_names(value: int, table: dict) -> list:
    return [name for bit, name in table.items() if value & bit]


# --------------------------------------------------------------------------
# CRC-32 and COBS
# --------------------------------------------------------------------------

def crc32(data: bytes) -> int:
    """CRC-32/ISO-HDLC (same as zlib.crc32). crc32(b"123456789") == 0xCBF43926."""
    return zlib.crc32(data) & 0xFFFFFFFF


def cobs_encode(data: bytes) -> bytes:
    """COBS-encode `data` (no trailing 0x00 delimiter).

    A block that fills to 254 bytes exactly at the end of the input gets no
    extra 0x01 code byte, as in the reference C encoder. dermis_wire.h does
    the same, so both sides produce identical bytes.
    """
    out = bytearray(b"\x00")
    code_idx = 0
    code = 1
    n = len(data)
    for i, b in enumerate(data):
        if b:
            out.append(b)
            code += 1
        if b == 0 or code == 0xFF:
            out[code_idx] = code
            code = 1
            code_idx = len(out)
            if b == 0 or i < n - 1:
                out.append(0)
    if code_idx < len(out):
        out[code_idx] = code
    return bytes(out)


def cobs_decode(enc: bytes) -> bytes:
    out = bytearray()
    i, n = 0, len(enc)
    while i < n:
        code = enc[i]
        if code == 0:
            raise ValueError("0x00 inside COBS data")
        end = i + code
        if end > n:
            raise ValueError("COBS block runs past the end")
        block = enc[i + 1:end]
        if 0 in block:
            raise ValueError("0x00 inside COBS block")
        out += block
        i = end
        if code != 0xFF and i < n:
            out.append(0)
    return bytes(out)


# --------------------------------------------------------------------------
# Packets
# --------------------------------------------------------------------------

@dataclass
class Packet:
    type: int
    version: int
    seq: int
    t_us: int
    payload: bytes

    @property
    def name(self) -> str:
        try:
            return Msg(self.type).name
        except ValueError:
            return f"0x{self.type:02X}"


class PacketError(ValueError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def encode_packet(msg_type: int, payload: bytes = b"", seq: int = 0, t_us: int = 0,
                  version: int = PROTO_VERSION) -> bytes:
    """Header + payload + CRC, COBS-encoded, followed by one 0x00."""
    payload = bytes(payload)
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload too large ({len(payload)} > {MAX_PAYLOAD})")
    body = HEADER.pack(msg_type, version, len(payload), seq & 0xFFFFFFFF,
                       t_us & 0xFFFFFFFFFFFFFFFF) + payload
    body += CRC.pack(crc32(body))
    return cobs_encode(body) + b"\x00"


def parse_packet(raw: bytes) -> Packet:
    """Validate one COBS-decoded packet."""
    if len(raw) < HEADER_SIZE + CRC_SIZE:
        raise PacketError("short", f"packet is {len(raw)} bytes, minimum is 20")
    msg_type, version, plen, seq, t_us = HEADER.unpack_from(raw, 0)
    if HEADER_SIZE + plen + CRC_SIZE != len(raw):
        raise PacketError("length", f"payload_len {plen} does not match packet size {len(raw)}")
    (crc,) = CRC.unpack_from(raw, HEADER_SIZE + plen)
    if crc != crc32(raw[:HEADER_SIZE + plen]):
        raise PacketError("crc", "CRC mismatch")
    if version >> 4 != PROTO_MAJOR:
        raise PacketError("version", f"unsupported major version {version >> 4}")
    return Packet(msg_type, version, seq, t_us, bytes(raw[HEADER_SIZE:HEADER_SIZE + plen]))


class StreamDecoder:
    """Turns an arbitrary byte stream into packets; resyncs on the next 0x00 after any error."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self.packets = 0
        self.errors = {"cobs": 0, "short": 0, "length": 0, "crc": 0, "version": 0, "overflow": 0}

    def feed(self, data: bytes) -> list:
        self._buf += data
        out = []
        while True:
            idx = self._buf.find(0)
            if idx < 0:
                if len(self._buf) > MAX_ENCODED:
                    self.errors["overflow"] += 1
                    self._buf.clear()
                break
            chunk = bytes(self._buf[:idx])
            del self._buf[:idx + 1]
            if not chunk:
                continue                       # back-to-back delimiters are fine
            if len(chunk) > MAX_ENCODED:
                self.errors["overflow"] += 1
                continue
            try:
                raw = cobs_decode(chunk)
            except ValueError:
                self.errors["cobs"] += 1
                continue
            try:
                pkt = parse_packet(raw)
            except PacketError as e:
                self.errors[e.kind] += 1
                continue
            self.packets += 1
            out.append(pkt)
        return out


# --------------------------------------------------------------------------
# FRAME (0x10)
# --------------------------------------------------------------------------

@dataclass
class MagRaw:
    ch: int
    status: int
    flags: int
    x: int
    y: int
    z: int
    t_raw: int


@dataclass
class ImuRaw:
    ch: int
    flags: int
    t_rel_us: int
    acc: tuple
    gyr: tuple
    temp_raw: int
    sensortime: int


@dataclass
class Frame:
    seq: int
    t_us: int
    frame_counter: int
    config_id: int
    flags: int
    acq_us: int
    mags: list
    imus: list


def build_frame_payload(frame_counter: int, config_id: int, flags: int, acq_us: int,
                        mags: Iterable[MagRaw], imus: Iterable[ImuRaw]) -> bytes:
    mags, imus = list(mags), list(imus)
    out = bytearray(FRAME_HEADER.pack(frame_counter & 0xFFFFFFFF, config_id, flags, 0,
                                      len(mags), MAG_REC.size, len(imus), IMU_REC.size,
                                      acq_us, 0))
    for m in mags:
        out += MAG_REC.pack(m.ch, m.status, m.flags, 0, m.x, m.y, m.z, m.t_raw)
    for r in imus:
        out += IMU_REC.pack(r.ch, r.flags, r.t_rel_us, *r.acc, *r.gyr, r.temp_raw, 0,
                            r.sensortime & 0xFFFFFF)
    return bytes(out)


def parse_frame(pkt: Packet) -> Frame:
    p = pkt.payload
    if len(p) < FRAME_HEADER.size:
        raise PacketError("frame", "frame header truncated")
    (fc, cfg, fflags, _r0, n_mag, mag_sz, n_imu, imu_sz, acq_us, _r1) = FRAME_HEADER.unpack_from(p, 0)
    if mag_sz < MAG_REC.size or imu_sz < IMU_REC.size:
        raise PacketError("frame", "record size smaller than v1.0")
    if len(p) < FRAME_HEADER.size + n_mag * mag_sz + n_imu * imu_sz:
        raise PacketError("frame", "records truncated")
    off = FRAME_HEADER.size
    mags, imus = [], []
    for _ in range(n_mag):          # step by mag_rec_size: later minors may append fields
        ch, st, fl, _r, x, y, z, t = MAG_REC.unpack_from(p, off)
        mags.append(MagRaw(ch, st, fl, x, y, z, t))
        off += mag_sz
    for _ in range(n_imu):
        ch, fl, trel, ax, ay, az, gx, gy, gz, temp, _r, st = IMU_REC.unpack_from(p, off)
        imus.append(ImuRaw(ch, fl, trel, (ax, ay, az), (gx, gy, gz), temp, st & 0xFFFFFF))
        off += imu_sz
    return Frame(pkt.seq, pkt.t_us, fc, cfg, fflags, acq_us, mags, imus)


# --------------------------------------------------------------------------
# DEVICE_INFO and conversion to SI units
# --------------------------------------------------------------------------

class ConfigMismatch(Exception):
    """Frame was made with a config the host doesn't have; send GET_INFO."""


class DeviceInfo:
    def __init__(self, obj: dict):
        self.raw = obj
        self.config_id = obj["config_id"]
        self.device_id = obj.get("device_id", "")
        self.mag = {m["ch"]: m for m in obj.get("mag", [])}
        self.imu = {m["ch"]: m for m in obj.get("imu", [])}
        self.pod_by_mag, self.pod_by_imu = {}, {}
        for pod in obj.get("pods", []):
            if pod.get("mag_ch") is not None:
                self.pod_by_mag[pod["mag_ch"]] = pod
            if pod.get("imu_ch") is not None:
                self.pod_by_imu[pod["imu_ch"]] = pod

    @classmethod
    def from_packet(cls, pkt: Packet) -> "DeviceInfo":
        return cls(json.loads(pkt.payload.decode("utf-8")))

    def pod_keys(self) -> list:
        """Stable, ordered keys used in decoded frames and CSV columns."""
        keys = [p["location"] for p in self.raw.get("pods", [])]
        keys += [f"mag{ch}" for ch in sorted(self.mag) if ch not in self.pod_by_mag]
        keys += [f"imu{ch}" for ch in sorted(self.imu) if ch not in self.pod_by_imu]
        return keys


def mag_to_si(m: MagRaw, cfg: dict, frame_t_us: int) -> dict:
    valid = not (m.flags & MAG_INVALID_MASK)
    sx, sy, sz = cfg["scale_ut_per_lsb"]
    return {
        "ch": m.ch,
        "b_ut": [m.x * sx, m.y * sy, m.z * sz] if valid else None,
        "temp_c": (25.0 + (m.t_raw - cfg.get("tref", 46244)) / cfg.get("temp_lsb_per_c", 45.2))
        if valid else None,
        "t_us": frame_t_us + cfg.get("conv_time_us", 0) // 2,
        "status": m.status,
        "flags": flag_names(m.flags, MAG_FLAGS),
        "valid": valid,
    }


def imu_to_si(r: ImuRaw, cfg: dict, frame_t_us: int) -> dict:
    valid = not (r.flags & IMU_INVALID_MASK)
    a, w = cfg["acc_scale_mps2_per_lsb"], cfg["gyr_scale_rads_per_lsb"]
    temp_ok = valid and r.temp_raw != -32768
    return {
        "ch": r.ch,
        "acc_mps2": [v * a for v in r.acc] if valid else None,
        "gyr_rads": [v * w for v in r.gyr] if valid else None,
        "temp_c": (cfg.get("temp_offset_c", 23.0) + r.temp_raw * cfg.get("temp_c_per_lsb", 1 / 512))
        if temp_ok else None,
        "t_us": frame_t_us + r.t_rel_us,
        "sensortime": r.sensortime,
        "flags": flag_names(r.flags, IMU_FLAGS),
        "valid": valid,
    }


def decode_frame(frame: Frame, info: DeviceInfo, t_host_ns: Optional[int] = None) -> dict:
    """Decoded form, matching schemas/decoded_frame.schema.json."""
    if frame.config_id != info.config_id:
        raise ConfigMismatch(f"frame config_id {frame.config_id} != DEVICE_INFO {info.config_id}")
    pods = {}
    for pod in info.raw.get("pods", []):
        pods[pod["location"]] = {"pod_id": pod.get("id", ""), "mag": None, "imu": None}
    for m in frame.mags:
        pod = info.pod_by_mag.get(m.ch)
        key = pod["location"] if pod else f"mag{m.ch}"
        entry = pods.setdefault(key, {"pod_id": "", "mag": None, "imu": None})
        entry["mag"] = mag_to_si(m, info.mag[m.ch], frame.t_us)
    for r in frame.imus:
        pod = info.pod_by_imu.get(r.ch)
        key = pod["location"] if pod else f"imu{r.ch}"
        entry = pods.setdefault(key, {"pod_id": "", "mag": None, "imu": None})
        entry["imu"] = imu_to_si(r, info.imu[r.ch], frame.t_us)
    return {
        "device_id": info.device_id,
        "frame": frame.frame_counter,
        "t_device_us": frame.t_us,
        "t_host_ns": t_host_ns,
        "config_id": frame.config_id,
        "flags": flag_names(frame.flags, FRAME_FLAGS),
        "pods": pods,
    }


POD_COLUMNS = ("bx_ut", "by_ut", "bz_ut", "mag_temp_c", "mag_flags",
               "ax_mps2", "ay_mps2", "az_mps2", "gx_rads", "gy_rads", "gz_rads",
               "imu_temp_c", "imu_t_us", "imu_flags")


def csv_columns(info: DeviceInfo) -> list:
    cols = ["t_device_us", "t_host_ns", "frame", "config_id", "frame_flags"]
    for key in info.pod_keys():
        cols += [f"{key}_{c}" for c in POD_COLUMNS]
    return cols


def frame_to_row(frame: Frame, info: DeviceInfo, t_host_ns: Optional[int] = None) -> dict:
    """One flat row (column names from csv_columns). Invalid values are empty."""
    d = decode_frame(frame, info, t_host_ns)
    row = {"t_device_us": frame.t_us, "t_host_ns": t_host_ns if t_host_ns is not None else "",
           "frame": frame.frame_counter, "config_id": frame.config_id, "frame_flags": frame.flags}
    raw_mag = {m.ch: m for m in frame.mags}
    raw_imu = {r.ch: r for r in frame.imus}

    def f(v, nd=4):
        return "" if v is None else round(v, nd)

    for key, pod in d["pods"].items():
        mg, im = pod["mag"], pod["imu"]
        b = mg["b_ut"] if mg and mg["b_ut"] else [None] * 3
        row[f"{key}_bx_ut"], row[f"{key}_by_ut"], row[f"{key}_bz_ut"] = (f(v, 3) for v in b)
        row[f"{key}_mag_temp_c"] = f(mg["temp_c"], 3) if mg else ""
        row[f"{key}_mag_flags"] = raw_mag[mg["ch"]].flags if mg else ""
        acc = im["acc_mps2"] if im and im["acc_mps2"] else [None] * 3
        gyr = im["gyr_rads"] if im and im["gyr_rads"] else [None] * 3
        row[f"{key}_ax_mps2"], row[f"{key}_ay_mps2"], row[f"{key}_az_mps2"] = (f(v, 5) for v in acc)
        row[f"{key}_gx_rads"], row[f"{key}_gy_rads"], row[f"{key}_gz_rads"] = (f(v, 6) for v in gyr)
        row[f"{key}_imu_temp_c"] = f(im["temp_c"], 3) if im else ""
        row[f"{key}_imu_t_us"] = im["t_us"] if im else ""
        row[f"{key}_imu_flags"] = raw_imu[im["ch"]].flags if im else ""
    return row


# --------------------------------------------------------------------------
# Other messages
# --------------------------------------------------------------------------

def json_payload(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


# device -> host
def msg_device_info(info: dict, seq: int, t_us: int) -> bytes:
    return encode_packet(Msg.DEVICE_INFO, json_payload(info), seq, t_us)


def msg_stats(stats: dict, seq: int, t_us: int) -> bytes:
    return encode_packet(Msg.STATS, json_payload(stats), seq, t_us)


def msg_event(code: int, seq: int, t_us: int, text: str = "", severity: int = 0, ch: int = CH_NONE) -> bytes:
    return encode_packet(Msg.EVENT, EVENT.pack(code, severity, ch) + text.encode("utf-8"), seq, t_us)


def msg_ack(cmd_seq: int, cmd_type: int, status: int, seq: int, t_us: int, text: str = "") -> bytes:
    return encode_packet(Msg.ACK, ACK.pack(cmd_seq, cmd_type, status, 0) + text.encode("utf-8"), seq, t_us)


def msg_sync_pong(host_time: int, t_rx_us: int, seq: int, t_tx_us: int) -> bytes:
    return encode_packet(Msg.SYNC_PONG, SYNC_PONG.pack(host_time, t_rx_us), seq, t_tx_us)


def msg_log(text: str, seq: int, t_us: int) -> bytes:
    return encode_packet(Msg.LOG, text.encode("utf-8"), seq, t_us)


def msg_frame(frame_counter, config_id, flags, acq_us, mags, imus, seq, t_us) -> bytes:
    return encode_packet(Msg.FRAME, build_frame_payload(frame_counter, config_id, flags, acq_us, mags, imus),
                         seq, t_us)


# host -> device
def cmd_get_info(seq: int) -> bytes:
    return encode_packet(Msg.GET_INFO, b"", seq)


def cmd_start(seq: int, rate_hz: int = 0) -> bytes:
    return encode_packet(Msg.START, CMD_START.pack(rate_hz, 0), seq)


def cmd_stop(seq: int) -> bytes:
    return encode_packet(Msg.STOP, b"", seq)


def cmd_set_config(seq: int, patch: dict) -> bytes:
    return encode_packet(Msg.SET_CONFIG, json_payload(patch), seq)


def cmd_sync_ping(seq: int, host_time: int) -> bytes:
    return encode_packet(Msg.SYNC_PING, CMD_SYNC_PING.pack(host_time & 0xFFFFFFFFFFFFFFFF), seq)


def cmd_reset_sensor(seq: int, kind: int, ch: int = CH_NONE) -> bytes:
    return encode_packet(Msg.RESET_SENSOR, CMD_RESET.pack(kind, ch), seq)


def cmd_selftest(seq: int, test: int) -> bytes:
    return encode_packet(Msg.SELFTEST, bytes([test]), seq)


def cmd_set_mode(seq: int, mode: int) -> bytes:
    return encode_packet(Msg.SET_MODE, bytes([mode]), seq)


def cmd_reboot(seq: int) -> bytes:
    return encode_packet(Msg.REBOOT, b"", seq)


def describe(pkt: Packet) -> str:
    """One readable line for logs."""
    t = pkt.type
    if t == Msg.ACK and len(pkt.payload) >= ACK.size:
        cmd_seq, cmd_type, status, _ = ACK.unpack_from(pkt.payload)
        text = pkt.payload[ACK.size:].decode("utf-8", "replace")
        return f"ACK cmd_seq={cmd_seq} cmd={Msg(cmd_type).name if cmd_type in Msg._value2member_map_ else cmd_type:} " \
               f"status={AckStatus(status).name if status in AckStatus._value2member_map_ else status} {text}".rstrip()
    if t == Msg.EVENT and len(pkt.payload) >= EVENT.size:
        code, sev, ch = EVENT.unpack_from(pkt.payload)
        name = EventCode(code).name if code in EventCode._value2member_map_ else f"0x{code:04X}"
        text = pkt.payload[EVENT.size:].decode("utf-8", "replace")
        chs = "" if ch == CH_NONE else f" ch={ch}"
        return f"EVENT {name} {SEVERITY.get(sev, sev)}{chs} {text}".rstrip()
    if t == Msg.SYNC_PONG and len(pkt.payload) >= SYNC_PONG.size:
        host_time, t_rx = SYNC_PONG.unpack_from(pkt.payload)
        return f"SYNC_PONG host_time={host_time} t_rx_us={t_rx} t_tx_us={pkt.t_us}"
    if t in (Msg.LOG, Msg.STATS, Msg.DEVICE_INFO):
        return f"{pkt.name} {pkt.payload.decode('utf-8', 'replace')}"
    return f"{pkt.name} seq={pkt.seq} t_us={pkt.t_us} payload={len(pkt.payload)} B"


# --------------------------------------------------------------------------
# Example configuration (the current hardware) and test vectors
# --------------------------------------------------------------------------

ACC_SCALE_8G = G / 4096.0                 # 4096 LSB/g at +-8 g
GYR_SCALE_2000 = DEG / 16.4               # 16.4 LSB/(deg/s) at +-2000 dps


def _mag_cfg(ch: int, cs: int) -> dict:
    return {"ch": ch, "part": "MLX90393", "bus": "spi", "cs_gpio": cs, "addr": None,
            "gain_sel": 1, "hallconf": 12, "res_xyz": [2, 2, 2], "osr": 0, "osr2": 0, "dig_filt": 2,
            "tcmp_en": False, "scale_ut_per_lsb": [2.403, 2.403, 3.872], "rail_lsb": [22000, 22000, 22000],
            "tref": 46244, "temp_lsb_per_c": 45.2, "conv_time_us": 1344, "mount": "+X+Y+Z"}


def _imu_cfg(ch: int, bus: str, addr: int) -> dict:
    return {"ch": ch, "part": "BMI160", "bus": bus, "cs_gpio": None, "addr": addr,
            "acc_range_g": 8, "acc_odr_hz": 800, "gyr_range_dps": 2000, "gyr_odr_hz": 800,
            "acc_scale_mps2_per_lsb": round(ACC_SCALE_8G, 10),
            "gyr_scale_rads_per_lsb": round(GYR_SCALE_2000, 10),
            "temp_offset_c": 23.0, "temp_c_per_lsb": 0.001953125, "sensortime_us_per_tick": 39.0625,
            "mount": "+X+Y+Z"}


EXAMPLE_DEVICE_INFO = {
    "proto": "1.0",
    "fw": "0.4.0",
    "board": "esp32-s3",
    "device_id": "dermis-3c84279f2a10",
    "role": "glove",
    "hand": "right",
    "config_id": 3,
    "rate_hz": 0,
    "clock": {"source": "esp_timer", "unit": "us"},
    "pods": [
        {"id": "P-0001", "location": "index_tip", "mag_ch": 0, "imu_ch": 0},
        {"id": "P-0002", "location": "thumb_tip", "mag_ch": 1, "imu_ch": 1},
        {"id": "P-0003", "location": "middle_tip", "mag_ch": 2, "imu_ch": 2},
        {"id": "P-0004", "location": "thenar_apb", "mag_ch": 3, "imu_ch": 3},
    ],
    "mag": [_mag_cfg(0, 21), _mag_cfg(1, 4), _mag_cfg(2, 16), _mag_cfg(3, 17)],
    "imu": [_imu_cfg(0, "i2c0", 0x68), _imu_cfg(1, "i2c0", 0x69), _imu_cfg(2, "i2c1", 0x68),
            _imu_cfg(3, "i2c1", 0x69)],
}

EXAMPLE_STATS = {
    "uptime_ms": 754312, "frames_produced": 241023, "frames_sent": 241021, "tx_dropped": 2,
    "rate_hz": 349.6, "acq_us_max": 1712, "period_us_max": 2951,
    "mag_err": [0, 0, 0, 1], "imu_err": [0, 0, 0, 0], "rx_bad_packets": 0,
    "heap_free": 245120, "usb_tx_free_min": 1840, "chip_temp_c": 41.5,
}

EXAMPLE_SET_CONFIG = {
    "pods": [
        {"id": "P-0001", "location": "index_tip", "mag_ch": 0, "imu_ch": 0},
        {"id": "P-0002", "location": "thumb_tip", "mag_ch": 1, "imu_ch": 1},
        {"id": "P-0003", "location": "middle_tip", "mag_ch": 2, "imu_ch": 2},
        {"id": "P-0004", "location": "thenar_apb", "mag_ch": 3, "imu_ch": 3},
    ],
    "mag": [{"ch": 1, "mount": "-Y+X+Z"}],
    "imu": [{"ch": 1, "mount": "-Y+X+Z", "acc_range_g": 16}],
}

VECTOR_MAGS = [
    MagRaw(0, 0x03, 0, 52, -118, 6973, 46109),
    MagRaw(1, 0x03, 0, -210, 64, 7105, 46107),
    MagRaw(2, 0x03, 0x20, 15, -35, 6850, 46056),              # TEMP_STALE
    MagRaw(3, 0x01, 0x02, -20432, -20432, -20432, 0),         # BAD_LENGTH (the bring-up junk frame)
]
VECTOR_IMUS = [
    ImuRaw(0, 0, 212, (35, -60, 4090), (12, -8, 3), 552, 0xA1B2C3),
    ImuRaw(1, 0, 431, (-1210, 2950, 2700), (-150, 95, 40), 540, 0xA1B2D0),
    ImuRaw(2, 0x01, 650, (20, -45, 4101), (5, 2, -4), 548, 0xA1B2DC),   # STALE
    ImuRaw(3, 0, 869, (400, -3900, 900), (30, -22, 18), 560, 0xA1B2E8),
]
VECTOR_HOST_TIME = 0x0123456789ABCDEF

COBS_EXAMPLES = [
    ("00", "0101"),
    ("0000", "010101"),
    ("00110000", "0102110101"),
    ("11220033", "0311220233"),
    ("11223344", "0511223344"),
    ("11000000", "0211010101"),
    (bytes(range(1, 255)).hex(), "ff" + bytes(range(1, 255)).hex()),
    (bytes(range(0, 255)).hex(), "01ff" + bytes(range(1, 255)).hex()),
    (bytes(range(1, 256)).hex(), "ff" + bytes(range(1, 255)).hex() + "02ff"),
]


def _round(obj, nd=6):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    return obj


def vector_packets() -> dict:
    """name -> wire bytes. test_vectors.c must print exactly these."""
    frame = msg_frame(1234, 3, 0x08, 1650, VECTOR_MAGS, VECTOR_IMUS, seq=42, t_us=12_345_678)
    return {
        "get_info": cmd_get_info(0),
        "start": cmd_start(1, 0),
        "sync_ping": cmd_sync_ping(2, VECTOR_HOST_TIME),
        "device_info": msg_device_info(EXAMPLE_DEVICE_INFO, seq=0, t_us=1_000_000),
        "frame": frame,
        "ack": msg_ack(0, Msg.GET_INFO, AckStatus.OK, seq=7, t_us=12_000_000),
        "sync_pong": msg_sync_pong(VECTOR_HOST_TIME, 12_000_120, seq=8, t_tx_us=12_000_150),
        "event_marker": msg_event(EventCode.MARKER, seq=9, t_us=12_100_000, text="button"),
    }


def build_vectors() -> dict:
    pk = vector_packets()
    info = DeviceInfo(EXAMPLE_DEVICE_INFO)
    vectors = []
    for name, wire in pk.items():
        raw = cobs_decode(wire[:-1])
        pkt = parse_packet(raw)
        entry = {
            "name": name,
            "direction": "host_to_device" if pkt.type >= 0x80 else "device_to_host",
            "wire_hex": wire.hex(),
            "packet_hex": raw.hex(),
            "header": {"type": pkt.type, "type_name": pkt.name, "version": pkt.version,
                       "payload_len": len(pkt.payload), "seq": pkt.seq, "t_us": pkt.t_us,
                       "crc32": CRC.unpack_from(raw, len(raw) - 4)[0]},
            "payload_hex": pkt.payload.hex(),
        }
        if name == "frame":
            fr = parse_frame(pkt)
            entry["frame_raw"] = {
                "frame_counter": fr.frame_counter, "config_id": fr.config_id, "frame_flags": fr.flags,
                "acq_us": fr.acq_us,
                "mag": [vars(m) for m in fr.mags],
                "imu": [dict(vars(r), acc=list(r.acc), gyr=list(r.gyr)) for r in fr.imus],
            }
            entry["decoded"] = _round(decode_frame(fr, info))
            entry["decoded_with"] = "device_info vector (config_id 3)"
        elif name == "device_info":
            entry["payload_json"] = EXAMPLE_DEVICE_INFO
        vectors.append(entry)
    return {
        "schema": "dermis.wire test vectors",
        "proto": f"{PROTO_MAJOR}.{PROTO_MINOR}",
        "crc32_check": {"input_ascii": "123456789", "crc32": f"0x{crc32(b'123456789'):08X}"},
        "cobs_examples": [{"in_hex": a, "out_hex": b} for a, b in COBS_EXAMPLES],
        "packets": vectors,
    }


# --------------------------------------------------------------------------
# Simulated hub (offline captures and a fake serial device)
# --------------------------------------------------------------------------

REST_FIELD_UT = {  # rest field per pod with the magnet ~10 mm above the chip
    "index_tip": (120.0, -250.0, 27000.0),
    "thumb_tip": (-300.0, 150.0, 27600.0),
    "middle_tip": (50.0, 80.0, 26500.0),
    "thenar_apb": (-80.0, -40.0, 25800.0),
}
PRESS_SCHEDULE = {  # period s, offset s, duration s, peak dBz uT
    "index_tip": (3.0, 0.5, 0.6, 9000.0),
    "thumb_tip": (3.0, 0.5, 0.6, 8000.0),      # pinches with the index
    "middle_tip": (4.0, 1.7, 0.7, 7000.0),
    "thenar_apb": (5.0, 2.9, 1.0, 6000.0),     # power grasp
}


def _press(t: float, period: float, offset: float, dur: float) -> float:
    ph = (t - offset) % period
    return 0.5 * (1.0 - math.cos(2 * math.pi * ph / dur)) if 0 <= ph < dur else 0.0


class Sim:
    """Plausible synthetic hub: rest fields, presses, shear, a slip burst, hand motion, noise."""

    def __init__(self, info: Optional[dict] = None, seed: int = 7):
        self.info = copy.deepcopy(info or EXAMPLE_DEVICE_INFO)
        self.rng = random.Random(seed)
        self.seq = 0
        self.frame_counter = 0
        self.pending_flags = 0
        self.recovered_mag = set()
        self.recovered_imu = set()
        self.streaming = False
        self.rate_hz = 350.0

    def _seq(self) -> int:
        s = self.seq
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        return s

    # signals --------------------------------------------------------------
    def _mag(self, k: int, loc: str, cfg: dict, t: float, roll: float, pitch: float) -> MagRaw:
        bx0, by0, bz0 = REST_FIELD_UT.get(loc, (0.0, 0.0, 27000.0))
        period, offset, dur, peak = PRESS_SCHEDULE.get(loc, (4.0, 1.0, 0.6, 7000.0))
        p = _press(t, period, offset, dur)
        bz = bz0 + peak * p
        bx = bx0 + 1500.0 * p * math.sin(2 * math.pi * 0.7 * t)
        by = by0 + 600.0 * p
        cycle = int((t - offset) // period)
        ph = ((t - offset) % period) / dur
        if loc == "index_tip" and cycle % 2 == 1 and 0.3 < ph < 0.45:     # slip burst
            bx += 140.0 * math.sin(2 * math.pi * 90.0 * t)
            by += 90.0 * math.sin(2 * math.pi * 120.0 * t)
        bx += 20.0 * math.sin(roll)          # Earth's field seen through hand rotation
        by += 20.0 * math.sin(pitch)
        bz += 45.0 * math.cos(roll) * math.cos(pitch)
        bx += self.rng.gauss(0, 5.0)
        by += self.rng.gauss(0, 5.0)
        bz += self.rng.gauss(0, 9.0)
        sx, sy, sz = cfg["scale_ut_per_lsb"]
        rail = cfg.get("rail_lsb", [32767] * 3)
        counts, flags = [], 0
        for v, s, r in zip((bx, by, bz), (sx, sy, sz), rail):
            c = int(round(v / s))
            if abs(c) >= r:
                c = r if c > 0 else -r
                flags |= 0x08
            counts.append(c)
        t_raw = int(round(46109 + 45.2 * (0.002 * t + 0.2 * k) + self.rng.gauss(0, 0.6)))
        return MagRaw(cfg["ch"], 0x03, flags, counts[0], counts[1], counts[2], t_raw)

    def _imu(self, k: int, loc: str, cfg: dict, t: float, t_rel: int) -> ImuRaw:
        def angles(tt):
            period, offset, dur, _ = PRESS_SCHEDULE.get(loc, (4.0, 1.0, 0.6, 0.0))
            flex = 0.45 * _press(tt, period, offset, dur) if loc.endswith("_tip") else 0.0
            roll = 0.25 * math.sin(2 * math.pi * 0.21 * tt + k)
            pitch = 0.18 * math.sin(2 * math.pi * 0.13 * tt + 0.5 * k) + flex
            return roll, pitch
        roll, pitch = angles(t)
        r0, p0 = angles(t - 0.001)
        wx, wy = (roll - r0) / 0.001, (pitch - p0) / 0.001
        gx_ = -G * math.sin(pitch)
        gy_ = G * math.sin(roll) * math.cos(pitch)
        gz_ = G * math.cos(roll) * math.cos(pitch)
        a, w = cfg["acc_scale_mps2_per_lsb"], cfg["gyr_scale_rads_per_lsb"]
        clip = lambda v: max(-32768, min(32767, int(round(v))))
        acc = tuple(clip(v / a + self.rng.gauss(0, 6)) for v in (gx_, gy_, gz_))
        gyr = tuple(clip(v / w + self.rng.gauss(0, 3)) for v in (wx, wy, 0.0))
        temp = int(round(540 + 4 * k + 0.5 * t))
        st = (int(t * 1e6 / 39.0625) + 1000 * k) & 0xFFFFFF
        return ImuRaw(cfg["ch"], 0, t_rel, acc, gyr, temp, st), (roll, pitch)

    def frame_bytes(self, t_us: int, extra_flags: int = 0, junk_mag: Optional[int] = None) -> bytes:
        t = t_us / 1e6
        pods = {p["mag_ch"]: p for p in self.info["pods"] if p.get("mag_ch") is not None}
        imu_pods = {p["imu_ch"]: p for p in self.info["pods"] if p.get("imu_ch") is not None}
        mags, imus, attitude = [], [], {}
        for k, cfg in enumerate(self.info["imu"]):
            loc = imu_pods.get(cfg["ch"], {}).get("location", f"imu{cfg['ch']}")
            r, att = self._imu(k, loc, cfg, t, 210 + 219 * k + self.rng.randint(0, 20))
            if cfg["ch"] in self.recovered_imu:
                r.flags |= 0x20
            imus.append(r)
            attitude[loc] = att
        self.recovered_imu.clear()
        for k, cfg in enumerate(self.info["mag"]):
            loc = pods.get(cfg["ch"], {}).get("location", f"mag{cfg['ch']}")
            roll, pitch = attitude.get(loc, (0.0, 0.0))
            m = self._mag(k, loc, cfg, t, roll, pitch)
            if junk_mag == cfg["ch"]:
                m = MagRaw(cfg["ch"], 0x01, 0x02, -20432, -20432, -20432, 0)
            if cfg["ch"] in self.recovered_mag:
                m.flags |= 0x40
            mags.append(m)
        self.recovered_mag.clear()
        flags = self.pending_flags | extra_flags
        self.pending_flags = 0
        acq = 1600 + self.rng.randint(0, 120)
        pkt = msg_frame(self.frame_counter, self.info["config_id"], flags, acq, mags, imus,
                        self._seq(), t_us)
        self.frame_counter = (self.frame_counter + 1) & 0xFFFFFFFF
        return pkt

    # commands -------------------------------------------------------------
    def handle(self, pkt: Packet, t_us: int) -> list:
        """Replies (wire bytes) to one host command, like the firmware would."""
        out = []
        ack = lambda status, text="": msg_ack(pkt.seq, pkt.type, status, self._seq(), t_us, text)
        t = pkt.type
        if t == Msg.SYNC_PING:
            if len(pkt.payload) != CMD_SYNC_PING.size:
                return [ack(AckStatus.BAD_PAYLOAD, "SYNC_PING needs 8 bytes")]
            (host_time,) = CMD_SYNC_PING.unpack(pkt.payload)
            return [msg_sync_pong(host_time, t_us, self._seq(), t_us + 30)]
        if t == Msg.GET_INFO:
            return [ack(AckStatus.OK), msg_device_info(self.info, self._seq(), t_us)]
        if t == Msg.START:
            rate = CMD_START.unpack(pkt.payload)[0] if len(pkt.payload) == CMD_START.size else 0
            self.rate_hz = float(rate) if rate else 350.0
            self.streaming = True
            return [ack(AckStatus.OK), msg_event(EventCode.STREAM_STARTED, self._seq(), t_us,
                                                f"rate_hz={self.rate_hz:g}")]
        if t == Msg.STOP:
            self.streaming = False
            return [ack(AckStatus.OK), msg_event(EventCode.STREAM_STOPPED, self._seq(), t_us)]
        if t == Msg.SET_CONFIG:
            if self.streaming:
                return [ack(AckStatus.BAD_STATE, "stop streaming before SET_CONFIG")]
            try:
                patch = json.loads(pkt.payload.decode("utf-8"))
                new = apply_set_config(self.info, patch)
            except (ValueError, KeyError, TypeError) as e:
                return [ack(AckStatus.BAD_PAYLOAD, str(e))]
            self.info = new
            self.pending_flags |= 0x01
            return [ack(AckStatus.OK), msg_device_info(self.info, self._seq(), t_us),
                    msg_event(EventCode.CONFIG_CHANGED, self._seq(), t_us, str(self.info["config_id"]))]
        if t == Msg.RESET_SENSOR:
            if len(pkt.payload) != CMD_RESET.size:
                return [ack(AckStatus.BAD_PAYLOAD)]
            kind, ch = CMD_RESET.unpack(pkt.payload)
            mags = [m["ch"] for m in self.info["mag"]] if ch == CH_NONE else [ch]
            imus = [m["ch"] for m in self.info["imu"]] if ch == CH_NONE else [ch]
            if kind in (0, 2):
                self.recovered_mag.update(mags)
            if kind in (1, 2):
                self.recovered_imu.update(imus)
            return [ack(AckStatus.OK)]
        if t == Msg.SELFTEST:
            if self.streaming:
                return [ack(AckStatus.BAD_STATE, "stop streaming before SELFTEST")]
            test = pkt.payload[0] if pkt.payload else 0
            result = {"test": test, "pass": True,
                      "mag": [{"ch": m["ch"], "ok": True} for m in self.info["mag"]],
                      "imu": [{"ch": m["ch"], "ok": True} for m in self.info["imu"]]}
            return [ack(AckStatus.OK), msg_event(EventCode.SELFTEST_RESULT, self._seq(), t_us,
                                                json.dumps(result, separators=(",", ":")))]
        if t in (Msg.SET_MODE, Msg.REBOOT):
            return [ack(AckStatus.OK)]
        return [ack(AckStatus.UNKNOWN_TYPE)]


WRITABLE_MAG = {"gain_sel", "res_xyz", "osr", "osr2", "dig_filt", "hallconf", "mount"}
WRITABLE_IMU = {"acc_range_g", "acc_odr_hz", "gyr_range_dps", "gyr_odr_hz", "mount"}


def mount_matrix(mount: str) -> list:
    """Rows = pod axes (distal, lateral, normal) in chip coordinates. Rejects reflections."""
    if len(mount) != 6:
        raise ValueError(f"bad mount '{mount}'")
    rows = []
    for i in range(3):
        sign, axis = mount[2 * i], mount[2 * i + 1]
        if sign not in "+-" or axis not in "XYZ":
            raise ValueError(f"bad mount '{mount}'")
        v = [0, 0, 0]
        v["XYZ".index(axis)] = 1 if sign == "+" else -1
        rows.append(v)
    (a, b, c), (d, e, f), (g, h, i_) = rows
    det = a * (e * i_ - f * h) - b * (d * i_ - f * g) + c * (d * h - e * g)
    if det != 1:
        raise ValueError(f"mount '{mount}' is not a rotation (axes repeat or reflect)")
    return rows


def apply_set_config(info: dict, patch: dict) -> dict:
    """Validate and apply a SET_CONFIG patch the way the firmware should (all or nothing)."""
    if not isinstance(patch, dict) or not patch:
        raise ValueError("SET_CONFIG needs a non-empty JSON object")
    allowed = {"rate_hz", "role", "hand", "pods", "mag", "imu"}
    unknown = set(patch) - allowed
    if unknown:
        raise ValueError(f"not writable: {sorted(unknown)}")
    new = copy.deepcopy(info)
    if "rate_hz" in patch:
        new["rate_hz"] = patch["rate_hz"]
    if "role" in patch:
        if patch["role"] not in ("glove", "robot", "bench"):
            raise ValueError("role must be glove, robot or bench")
        new["role"] = patch["role"]
    if "hand" in patch:
        if patch["hand"] not in ("left", "right", "none"):
            raise ValueError("hand must be left, right or none")
        new["hand"] = patch["hand"]
    if "pods" in patch:
        for p in patch["pods"]:
            if p["location"] not in LOCATIONS:
                raise ValueError(f"unknown location {p['location']}")
        new["pods"] = patch["pods"]
    for key, writable in (("mag", WRITABLE_MAG), ("imu", WRITABLE_IMU)):
        for item in patch.get(key, []):
            target = next((m for m in new[key] if m["ch"] == item["ch"]), None)
            if target is None:
                raise ValueError(f"no {key} channel {item['ch']}")
            bad = set(item) - writable - {"ch"}
            if bad:
                raise ValueError(f"{key}[{item['ch']}]: not writable: {sorted(bad)}")
            if "mount" in item:
                mount_matrix(item["mount"])
            target.update(item)
            if key == "imu" and "acc_range_g" in item:
                target["acc_scale_mps2_per_lsb"] = round(G * item["acc_range_g"] / 32768.0, 10)
            if key == "imu" and "gyr_range_dps" in item:
                lsb = {125: 262.4, 250: 131.2, 500: 65.6, 1000: 32.8, 2000: 16.4}[item["gyr_range_dps"]]
                target["gyr_scale_rads_per_lsb"] = round(DEG / lsb, 10)
    new["config_id"] = (info["config_id"] + 1) & 0xFFFF
    return new


def simulate_capture(seconds: float = 2.0, rate_hz: float = 350.0, seed: int = 7) -> bytes:
    """An offline byte stream as a host would receive it, with a few deliberate faults:
    one BAD_LENGTH mag record, one dropped frame (TX_DROP) and one marker."""
    sim = Sim(seed=seed)
    out = bytearray()
    t_us = 1_000_000
    out += msg_event(EventCode.BOOT, sim._seq(), 20_000, "power-on")
    out += msg_ack(0, Msg.GET_INFO, AckStatus.OK, sim._seq(), t_us)
    out += msg_device_info(sim.info, sim._seq(), t_us)
    out += msg_ack(1, Msg.START, AckStatus.OK, sim._seq(), t_us + 200)
    out += msg_event(EventCode.STREAM_STARTED, sim._seq(), t_us + 210, f"rate_hz={rate_hz:g}")
    period_us = 1e6 / rate_hz
    n = int(seconds * rate_hz)
    t0 = t_us + 1000
    next_stats = t0 + 1_000_000
    for i in range(n):
        ft = int(t0 + i * period_us + sim.rng.randint(-15, 15))
        if i == int(0.8 * rate_hz):            # pretend the TX buffer was full: frame produced, not sent
            sim.frame_counter += 1
            sim.pending_flags |= 0x04
            continue
        extra = 0x08 if i == int(1.2 * rate_hz) else 0
        if extra:
            out += msg_event(EventCode.MARKER, sim._seq(), ft, "button")
        junk = 3 if i == int(1.5 * rate_hz) else None
        out += sim.frame_bytes(ft, extra, junk)
        if ft >= next_stats:
            stats = dict(EXAMPLE_STATS, uptime_ms=ft // 1000, frames_produced=sim.frame_counter,
                         frames_sent=sim.frame_counter - 1, tx_dropped=1, rate_hz=rate_hz)
            out += msg_stats(stats, sim._seq(), ft + 50)
            next_stats += 1_000_000
    end = int(t0 + n * period_us)
    out += msg_ack(2, Msg.STOP, AckStatus.OK, sim._seq(), end)
    out += msg_event(EventCode.STREAM_STOPPED, sim._seq(), end + 10)
    return bytes(out)


def serve_pty(rate_hz: float = 350.0) -> None:
    """Fake hub on a virtual serial port. Open the printed /dev/ttys... path like a real port."""
    import tty
    master, slave = os.openpty()
    tty.setraw(slave)
    os.set_blocking(master, False)
    print(f"Fake Dermis hub on {os.ttyname(slave)}  (Ctrl-C to stop)", flush=True)
    sim = Sim()
    sim.rate_hz = rate_hz
    dec = StreamDecoder()
    t0 = time.monotonic()
    now_us = lambda: int((time.monotonic() - t0) * 1e6) + 1_000_000
    drops = 0

    def send(data: bytes) -> bool:
        try:
            os.write(master, data)
            return True
        except BlockingIOError:
            return False

    send(msg_event(EventCode.BOOT, sim._seq(), now_us(), "power-on"))
    next_frame = time.monotonic()
    try:
        while True:
            timeout = max(0.0, next_frame - time.monotonic()) if sim.streaming else 0.1
            readable, _, _ = select.select([master], [], [], timeout)
            if readable:
                try:
                    data = os.read(master, 4096)
                except (BlockingIOError, OSError):
                    data = b""
                for pkt in dec.feed(data):
                    was_streaming = sim.streaming
                    for reply in sim.handle(pkt, now_us()):
                        send(reply)
                    if sim.streaming and not was_streaming:
                        next_frame = time.monotonic()
                    print(f"<- {pkt.name} seq={pkt.seq}", flush=True)
            if sim.streaming and time.monotonic() >= next_frame:
                frame = sim.frame_bytes(now_us())
                if not send(frame):
                    drops += 1
                    sim.pending_flags |= 0x04
                next_frame += 1.0 / sim.rate_hz
                if time.monotonic() - next_frame > 0.1:      # fell far behind: resync, don't burst
                    next_frame = time.monotonic()
    except KeyboardInterrupt:
        print(f"\nstopped; {sim.frame_counter} frames produced, {drops} dropped (host not reading)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _read_file(path: str, chunk: int = 4096) -> Iterator[bytes]:
    with open(path, "rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                return
            yield data


def _read_port(port: str) -> Iterator[bytes]:
    try:
        import serial  # pyserial
    except ImportError:
        sys.exit("dump --port needs pyserial:  pip install pyserial")
    ser = serial.Serial(port, timeout=0.05)
    host_ns = time.monotonic_ns
    ser.write(b"\x00" + cmd_get_info(0) + cmd_sync_ping(1, host_ns()) + cmd_start(2, 0))
    seq = 3
    last_ping = time.monotonic()
    try:
        while True:
            data = ser.read(4096)
            if data:
                yield data
            if time.monotonic() - last_ping > 1.0:
                ser.write(cmd_sync_ping(seq, host_ns()))
                seq += 1
                last_ping = time.monotonic()
    except KeyboardInterrupt:
        ser.write(cmd_stop(seq))
        ser.close()


def dump(source: Iterable[bytes], csv_path: Optional[str] = None, every: int = 350, quiet: bool = False) -> dict:
    dec = StreamDecoder()
    info: Optional[DeviceInfo] = None
    writer = None
    fh = None
    stats = {"frames": 0, "lost_frames": 0, "invalid_mag": 0, "invalid_imu": 0, "waiting_for_info": 0}
    last_fc = None
    try:
        for chunk in source:
            for pkt in dec.feed(chunk):
                if pkt.type == Msg.DEVICE_INFO:
                    info = DeviceInfo.from_packet(pkt)
                    if not quiet:
                        print(f"DEVICE_INFO {info.device_id} config_id={info.config_id} "
                              f"pods={[p['location'] for p in info.raw['pods']]}")
                    if csv_path and writer is None:
                        fh = open(csv_path, "w", newline="")
                        writer = csv.DictWriter(fh, fieldnames=csv_columns(info))
                        writer.writeheader()
                    continue
                if pkt.type != Msg.FRAME:
                    if not quiet:
                        print(describe(pkt))
                    continue
                fr = parse_frame(pkt)
                if info is None or fr.config_id != info.config_id:
                    stats["waiting_for_info"] += 1
                    continue
                if last_fc is not None and fr.frame_counter != (last_fc + 1) & 0xFFFFFFFF:
                    stats["lost_frames"] += (fr.frame_counter - last_fc - 1) & 0xFFFFFFFF
                last_fc = fr.frame_counter
                stats["frames"] += 1
                stats["invalid_mag"] += sum(1 for m in fr.mags if m.flags & MAG_INVALID_MASK)
                stats["invalid_imu"] += sum(1 for r in fr.imus if r.flags & IMU_INVALID_MASK)
                if writer:
                    writer.writerow(frame_to_row(fr, info))
                if not quiet and (stats["frames"] % every == 1 or fr.flags):
                    d = decode_frame(fr, info)
                    parts = []
                    for key, pod in d["pods"].items():
                        b = pod["mag"]["b_ut"] if pod["mag"] and pod["mag"]["b_ut"] else None
                        parts.append(f"{key}: " + (f"|B|={math.sqrt(sum(v * v for v in b)) / 1000:.2f} mT"
                                                   if b else "invalid"))
                    print(f"frame {fr.frame_counter} t={fr.t_us / 1e6:.4f}s {d['flags'] or ''} " + "  ".join(parts))
    finally:
        if fh:
            fh.close()
    stats["packets"] = dec.packets
    stats["stream_errors"] = dec.errors
    if not quiet:
        print(json.dumps(stats))
    return stats


def check_vectors(lines: Iterable[str]) -> int:
    expected = {k: v.hex() for k, v in vector_packets().items()}
    seen, bad = 0, 0
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        name, hexs = parts
        if name not in expected:
            print(f"?    {name}: no such vector")
            bad += 1
            continue
        seen += 1
        if hexs.lower() == expected[name]:
            print(f"ok   {name}")
        else:
            bad += 1
            print(f"FAIL {name}\n  got      {hexs}\n  expected {expected[name]}")
    if seen == 0:
        print("no vectors found in input")
        return 1
    return 1 if bad else 0


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Dermis Wire Schema v1.0 reference tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("vectors", help="print test vectors as JSON")
    cv = sub.add_parser("check-vectors", help="compare 'name hex' lines to the vectors")
    cv.add_argument("path", help="file with 'name hex' lines, or - for stdin")
    sm = sub.add_parser("simulate", help="write a simulated capture")
    sm.add_argument("--seconds", type=float, default=2.0)
    sm.add_argument("--rate", type=float, default=350.0)
    sm.add_argument("--out", required=True)
    dp = sub.add_parser("dump", help="decode a capture file or a live port")
    src = dp.add_mutually_exclusive_group(required=True)
    src.add_argument("--file")
    src.add_argument("--port")
    dp.add_argument("--csv")
    dp.add_argument("--quiet", action="store_true")
    sp = sub.add_parser("serve-pty", help="fake hub on a virtual serial port (macOS/Linux)")
    sp.add_argument("--rate", type=float, default=350.0)
    args = ap.parse_args(argv)

    if args.cmd == "vectors":
        print(json.dumps(build_vectors(), indent=2))
    elif args.cmd == "check-vectors":
        fh = sys.stdin if args.path == "-" else open(args.path)
        return check_vectors(fh)
    elif args.cmd == "simulate":
        data = simulate_capture(args.seconds, args.rate)
        with open(args.out, "wb") as fh:
            fh.write(data)
        print(f"wrote {len(data)} bytes to {args.out}")
    elif args.cmd == "dump":
        source = _read_file(args.file) if args.file else _read_port(args.port)
        dump(source, args.csv, quiet=args.quiet)
    elif args.cmd == "serve-pty":
        serve_pty(args.rate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
