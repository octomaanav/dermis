"""Development TCP sender: emits decoded-frame NDJSON to `dermis teleop --source tcp`.

    python -m dermis send-test --mode fixture                    # the example frame, once
    python -m dermis send-test --mode replay --path rec.jsonl    # a recorded decoded JSONL stream
    python -m dermis send-test --mode synthetic --rate 30        # SYNTHETIC pinch/tilt motion
    (also: python -m dermis.server.dev_sender ...)

Modes
    fixture    firmware/v1/examples/decoded_frame.json, content unchanged, re-serialized onto
               one line (NDJSON needs one object per line; the pretty-printed file has newlines).
               Sent once. Repeating it (--repeat) sends the same frame number again: the receiver
               drops those as duplicates, and constant values could not show motion anyway.
    replay     .jsonl / .json (one object, a list, or one per line) of decoded frames, sent
               unchanged, paced by their t_device_us differences / --speed (or --rate if the
               timestamps are unusable). A .bin Dermis Wire capture is decoded first with the
               reference decoder (firmware/v1/host/python/dermis_wire.py).
    synthetic  SYNTHETIC development data built on the example frame's structure, pods, pod IDs,
               units and per-sensor timestamp offsets: monotonically increasing frame numbers,
               a stationary unloaded interval for calibration, then smooth index/thumb magnetic
               changes (pinch) and an optional coherent gravity vector under slow tilt.
               --profile updown instead holds the gripper open and pitches the index pod
               forward/back (+-0.45 rad, 6 s period): with `teleop --tilt-z` the arm moves
               down and up. --profile punch makes a fist, then punches PUNCH_REACH_M forward along
               chip +x (minimum-jerk out in 0.25 s, ~3 g peak; back in 0.4 s) every 2.5 s: the IMUs
               carry that linear acceleration on top of gravity, as a real punch would.
               --write PATH saves the frames as JSONL (no network) for replay.
               The middle_tip field follows the same pinch 1 s later, so each claw of the hand
               sim moves on its own. --format draft emits the proposed (not yet v1) shape: no
               imu key on middle_tip/thenar_apb and a bare "wrist_imu" record without
               valid/flags (see server/compat.py).
               device_id is "synthetic-dermis-dev" so the receiver can label the source;
               t_host_ns stays null (there is no host clock-sync fit). It proves software
               integration only, not sensor behaviour.

Faults (--faults name@t[:dur],...  t and dur in seconds since the first frame; --faults ci = preset)
    malformed  one truncated-JSON line and one invalid-UTF-8 line
    fragment   the next frame written in 3 pieces with short sleeps in between
    combine    the next 3 frames in one write
    duplicate  the next frame sent twice
    reorder    two frames swapped (n+1 before n)
    oversize   one line larger than --oversize-bytes
    nonfinite  one frame with NaN in index_tip acc (non-standard JSON "NaN")
    burst      for dur: 4 frames per tick (4x the control rate)
    null_index for dur: index_tip mag b_ut null, valid false, COMM_FAIL
    stale_thumb for dur: thumb_tip mag values repeated with STALE
    pause      for dur: transmit nothing (the device keeps counting frames)
    disconnect close the socket, wait dur, reconnect (same frame numbering)
    reboot     like disconnect, but frame numbers and the device clock restart from 0
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ..telemetry.kit import find_kit_file

SYNTHETIC_DEVICE_ID = "synthetic-dermis-dev"
PROFILES = ("pinch", "updown", "punch")
PUNCH_PERIOD_S, PUNCH_REACH_M = 2.5, 0.30
CI_FAULTS = ("malformed@3.0,fragment@3.3,combine@3.6,duplicate@4.0,reorder@4.3,oversize@4.6,nonfinite@4.9,"
             "burst@5.2:1.0,null_index@6.6:0.4,stale_thumb@7.4:0.4,pause@9.0:2.5,reboot@13.0:0.5")
G = 9.80665


def env_default(name: str, fallback):
    v = os.environ.get(name)
    return type(fallback)(v) if v not in (None, "") else fallback


def compact(obj: dict) -> bytes:
    return (json.dumps(obj, separators=(",", ":"), allow_nan=True) + "\n").encode("utf-8")


# --------------------------------------------------------------------------
# Synthetic frames
# --------------------------------------------------------------------------

def _minjerk(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5


def _minjerk_acc(x: float) -> float:
    """d2/dx2 of _minjerk: acceleration shape of a minimum-jerk move (integrates to 0 velocity at both ends)."""
    return 60 * x - 180 * x ** 2 + 120 * x ** 3


def _rot_matrix(pitch: float, roll: float) -> list:
    """Matrix of _rot: Ry(pitch) @ Rx(roll)."""
    return [_rot(e, pitch, roll) for e in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]      # columns


def _body_rate(pitch: float, roll: float, pitch2: float, roll2: float, dt: float) -> list:
    """Body-frame angular rate (rad/s) of a hand whose sensor reads acc = _rot(a0, pitch, roll).
    With M = Ry(pitch) Rx(roll) the body->world rotation is R = M^T, so [w]x = R^T dR/dt = M dM^T/dt."""
    c1, c2 = _rot_matrix(pitch, roll), _rot_matrix(pitch2, roll2)          # columns of M(t), M(t+dt)
    M = [[c1[j][i] for j in range(3)] for i in range(3)]
    dMt = [[(c2[i][j] - c1[i][j]) / dt for j in range(3)] for i in range(3)]   # (dM/dt)^T: rows = columns of dM
    S = [[sum(M[i][k] * dMt[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return [S[2][1], S[0][2], S[1][0]]


def _rot(v, pitch: float, roll: float):
    """Rotate a chip-axis vector by roll about x, then pitch about y."""
    x, y, z = v
    cr, sr = math.cos(roll), math.sin(roll)
    y, z = cr * y - sr * z, sr * y + cr * z
    cp, sp = math.cos(pitch), math.sin(pitch)
    x, z = cp * x + sp * z, -sp * x + cp * z
    return [x, y, z]


class SyntheticGlove:
    """SYNTHETIC frames with the example frame's structure. frame_at(t) is deterministic in t."""

    def __init__(self, stationary_s: float = 2.5, grip_period_s: float = 4.0, tilt: bool = True,
                 noise_ut: float = 4.0, seed: int = 1, fixture: dict | None = None, profile: str = "pinch"):
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}")
        self.profile = profile
        self.base = fixture or json.loads(find_kit_file("decoded_frame.json").read_text())
        self.stationary_s, self.grip_period_s, self.tilt = stationary_s, grip_period_s, tilt
        self.noise_ut = noise_ut
        self.rng = random.Random(seed)
        self.peak_dbz = {"index_tip": 8000.0, "thumb_tip": 7000.0, "middle_tip": 7500.0}
        self.lag_s = {"middle_tip": 1.0}
        b = self.base
        self._mag_off = {k: p["mag"]["t_us"] - b["t_device_us"] for k, p in b["pods"].items() if p["mag"]}
        self._imu_off = {k: p["imu"]["t_us"] - b["t_device_us"] for k, p in b["pods"].items() if p["imu"]}
        self._st0 = {k: p["imu"]["sensortime"] for k, p in b["pods"].items() if p["imu"]}
        self.frame0, self.t0_us = 0, 1_000_000
        self._last_thumb = None

    def reboot(self) -> None:
        """Device restart: frame numbers and device clock start over."""
        self.frame0, self.t0_us = 0, 1_000_000

    def closure(self, t: float) -> float:
        if t < self.stationary_s or self.profile == "updown":
            return 0.0
        if self.profile == "punch":                   # fist before the punch, open after the return
            ph = (t - self.stationary_s) % PUNCH_PERIOD_S
            if ph < 0.3:
                return _minjerk(ph / 0.3)
            return 1.0 if ph < 1.3 else 1.0 - _minjerk(min(1.0, (ph - 1.3) / 0.3))
        return 0.5 - 0.5 * math.cos(2 * math.pi * (t - self.stationary_s) / self.grip_period_s)

    def attitude(self, t: float) -> tuple[float, float]:
        if not self.tilt or t < self.stationary_s:
            return 0.0, 0.0
        s = t - self.stationary_s
        if self.profile == "punch":
            return 0.0, 0.0
        if self.profile == "updown":                      # pitch only: arm down, then up (with --tilt-z)
            return 0.45 * math.sin(2 * math.pi * s / 6.0), 0.0
        return 0.30 * math.sin(2 * math.pi * s / 7.0), 0.25 * math.sin(2 * math.pi * s / 9.0)

    def lin_acc(self, t: float) -> list:
        """Linear acceleration of the hand (m/s^2, chip axes = hand axes when level). Punch profile only."""
        if self.profile != "punch" or t < self.stationary_s:
            return [0.0, 0.0, 0.0]
        ph = (t - self.stationary_s) % PUNCH_PERIOD_S
        ax = 0.0
        if 0.40 <= ph < 0.65:
            ax = PUNCH_REACH_M * _minjerk_acc((ph - 0.40) / 0.25) / 0.25 ** 2
        elif 0.75 <= ph < 1.15:
            ax = -PUNCH_REACH_M * _minjerk_acc((ph - 0.75) / 0.40) / 0.40 ** 2
        return [ax, 0.0, 0.0]

    def frame_at(self, t: float, index: int, t_dev_s: float | None = None, faults: frozenset = frozenset()) -> dict:
        """Frame number `index` sampled at motion time t (s). t_dev_s: device clock time if different."""
        f = copy.deepcopy(self.base)
        t_dev_s = t if t_dev_s is None else t_dev_s
        t_us = self.t0_us + int(round(t_dev_s * 1e6))
        f.update(device_id=SYNTHETIC_DEVICE_ID, frame=self.frame0 + index, t_device_us=t_us, t_host_ns=None, flags=[])
        pitch, roll = self.attitude(t)
        dt = 0.01
        p2, r2 = self.attitude(t + dt)
        rates = _body_rate(pitch, roll, p2, r2, dt)        # gyro consistent with the acc rotation
        for loc, pod in f["pods"].items():
            m, imu = pod["mag"], pod["imu"]
            if m is not None:
                m["t_us"] = t_us + self._mag_off[loc]
                if m["b_ut"] is not None:
                    b0 = self.base["pods"][loc]["mag"]["b_ut"]
                    lag = 0.0 if self.profile == "punch" else self.lag_s.get(loc, 0.0)
                    c = self.closure(t - lag) if t - lag >= self.stationary_s else 0.0
                    dz = self.peak_dbz.get(loc, 0.0) * c
                    shear = 0.12 * dz * math.sin(2 * math.pi * 0.3 * t)
                    m["b_ut"] = [round(b0[0] + shear + self.rng.gauss(0, self.noise_ut), 3),
                                 round(b0[1] + 0.05 * dz + self.rng.gauss(0, self.noise_ut), 3),
                                 round(b0[2] + dz + self.rng.gauss(0, self.noise_ut), 3)]
                    # temperature converts at half the field rate: TEMP_STALE on odd frames (field stays fresh)
                    m["flags"] = ["TEMP_STALE"] if index % 2 else []
            if imu is not None:
                imu["t_us"] = t_us + self._imu_off[loc]
                imu["sensortime"] = (self._st0[loc] + int(t_dev_s * 1e6 / 39.0625)) & 0xFFFFFF
                imu["flags"] = []
                if imu["acc_mps2"] is not None:
                    a = _rot(self.base["pods"][loc]["imu"]["acc_mps2"], pitch, roll)
                    a = [u + w for u, w in zip(a, self.lin_acc(t))]
                    imu["acc_mps2"] = [round(v + self.rng.gauss(0, 0.02), 6) for v in a]
                    imu["gyr_rads"] = [round(w + self.rng.gauss(0, 0.003), 6) for w in rates]
        if "null_index" in faults:
            m = f["pods"]["index_tip"]["mag"]
            m.update(b_ut=None, temp_c=None, flags=["COMM_FAIL"], valid=False, status=0)
        if "stale_thumb" in faults and self._last_thumb is not None:
            m = f["pods"]["thumb_tip"]["mag"]
            m.update(b_ut=self._last_thumb, flags=["STALE"])
        elif f["pods"]["thumb_tip"]["mag"]["b_ut"] is not None:
            self._last_thumb = f["pods"]["thumb_tip"]["mag"]["b_ut"]
        if "nonfinite" in faults:
            f["pods"]["index_tip"]["imu"]["acc_mps2"][0] = float("nan")
        return f

    def draft_frame_at(self, *a, **kw) -> dict:
        """Same data in the proposed draft shape (not v1): middle/thenar lose their imu key and the
        hand-level IMU arrives as a bare "wrist_imu" record with no valid / flags fields."""
        f = self.frame_at(*a, **kw)
        pods = f["pods"]
        wrist = copy.deepcopy(pods["thenar_apb"]["imu"])
        t_ = a[0] if a else kw["t"]
        a_ = [u + w for u, w in zip(_rot([0.0, 0.0, G], *self.attitude(t_)), self.lin_acc(t_))]
        wrist["acc_mps2"] = [round(v + self.rng.gauss(0, 0.02), 6) for v in a_]
        for k in ("flags", "valid"):
            wrist.pop(k, None)
        for loc in ("middle_tip", "thenar_apb"):
            pods[loc].pop("imu", None)
        pods["wrist_imu"] = wrist
        return f


# --------------------------------------------------------------------------
# Connection
# --------------------------------------------------------------------------

class Link:
    def __init__(self, host: str, port: int, connect_timeout: float, log):
        self.host, self.port, self.connect_timeout, self.log = host, port, connect_timeout, log
        self.sock: socket.socket | None = None
        self.bytes = 0

    def connect(self) -> None:
        deadline = time.monotonic() + self.connect_timeout
        while True:
            try:
                s = socket.create_connection((self.host, self.port), timeout=2.0)
                break
            except OSError as e:
                if time.monotonic() > deadline:
                    raise ConnectionError(f"cannot connect to {self.host}:{self.port}: {e}. "
                                          f"Is `python -m dermis teleop --source tcp` running?") from None
                time.sleep(0.2)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(1.0)
        try:
            hello = s.recv(4096).decode("utf-8", "replace")
        except socket.timeout:
            hello = ""
        if "controller_busy" in hello:
            s.close()
            raise ConnectionRefusedError("receiver already has a controlling client (controller_busy)")
        s.settimeout(5.0)
        self.sock = s
        self.log(f"[sender] connected to {self.host}:{self.port}")

    def write(self, data: bytes) -> None:
        self.sock.sendall(data)
        self.bytes += len(data)

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
            self.sock = None


@dataclass
class Fault:
    name: str
    at: float
    dur: float = 0.0
    done: bool = False

    def active(self, t: float) -> bool:
        return self.at <= t < self.at + self.dur


ONE_SHOT = {"malformed", "fragment", "combine", "duplicate", "reorder", "oversize", "nonfinite", "disconnect", "reboot"}
WINDOW = {"burst", "null_index", "stale_thumb", "pause"}


def parse_faults(spec: str) -> list[Fault]:
    if spec.strip() == "ci":
        spec = CI_FAULTS
    out = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, _, when = item.partition("@")
        if name not in ONE_SHOT | WINDOW:
            raise ValueError(f"unknown fault '{name}'; known: {sorted(ONE_SHOT | WINDOW)}")
        at, _, dur = when.partition(":")
        out.append(Fault(name, float(at or 0.0), float(dur or (0.5 if name in WINDOW | {"disconnect", "reboot"} else 0.0))))
    return out


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

def run_fixture(link: Link, a, log) -> dict:
    obj = json.loads(find_kit_file("decoded_frame.json").read_text())
    for i in range(a.repeat):
        link.write(compact(obj))
        if i == 0 and a.repeat > 1:
            log("[sender] note: repeats reuse frame number "
                f"{obj['frame']}; the receiver counts them as duplicates")
        time.sleep(1.0 / a.rate)
    log(f"[sender] FIXTURE: sent decoded_frame.json x{a.repeat} (frame {obj['frame']}, unchanged content)")
    return {"sent": a.repeat}


def load_recording(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".bin":
        return _decode_wire(p)
    text = p.read_text()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _decode_wire(p: Path) -> list[dict]:
    sys.path.insert(0, str(find_kit_file("dermis_wire.py").parent))
    import dermis_wire as w
    dec, info, out = w.StreamDecoder(), None, []
    for pkt in dec.feed(p.read_bytes()):
        if pkt.type == w.Msg.DEVICE_INFO:
            info = w.DeviceInfo.from_packet(pkt)
        elif pkt.type == w.Msg.FRAME and info is not None:
            fr = w.parse_frame(pkt)
            if fr.config_id == info.config_id:
                out.append(w.decode_frame(fr, info))
    return out


def run_replay(link: Link, a, log) -> dict:
    frames = load_recording(a.path)
    if not frames:
        raise ValueError(f"{a.path}: no frames")
    log(f"[sender] REPLAY of {a.path}: {len(frames)} frames at {a.speed}x (content unchanged)")
    sent, t_start = 0, time.monotonic()
    while True:
        t0_dev = frames[0].get("t_device_us")
        for i, f in enumerate(frames):
            due = None
            td = f.get("t_device_us")
            if isinstance(td, int) and isinstance(t0_dev, int) and td >= t0_dev:
                due = (td - t0_dev) / 1e6 / a.speed
            else:
                due = i / a.rate
            delay = t_start + due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            link.write(compact(f))
            sent += 1
        if not a.loop:
            break
        t_start = time.monotonic()
    return {"sent": sent}


def write_synthetic(a, log) -> int:
    """Save --duration s of SYNTHETIC frames at --rate as JSONL (one decoded frame per line), no network."""
    gen = SyntheticGlove(stationary_s=a.stationary_s, grip_period_s=a.grip_period_s,
                         tilt=a.profile == "updown" or not a.no_tilt, seed=a.seed, profile=a.profile)
    n = int(a.duration * a.rate)
    Path(a.write).parent.mkdir(parents=True, exist_ok=True)
    with open(a.write, "wb") as fh:
        make = gen.draft_frame_at if getattr(a, "format", "v1") == "draft" else gen.frame_at
        for i in range(n):
            fh.write(compact(make(i / a.rate, i)))
    log(f"[sender] wrote {n} SYNTHETIC '{a.profile}' frames ({a.duration:g} s at {a.rate:g} Hz) to {a.write}; "
        f"replay with: --mode replay --path {a.write}")
    return 0


def run_synthetic(link: Link, a, log) -> dict:
    gen = SyntheticGlove(stationary_s=a.stationary_s, grip_period_s=a.grip_period_s,
                         tilt=a.profile == "updown" or not a.no_tilt, seed=a.seed, profile=a.profile)
    faults = parse_faults(a.faults) if a.faults else []
    log(f"[sender] SYNTHETIC '{a.profile}' development data at {a.rate:g} Hz for {a.duration:g} s "
        f"(stationary {a.stationary_s:g} s first); proves software integration only")
    if faults:
        log("[sender] faults: " + ", ".join(f"{f.name}@{f.at:g}" + (f":{f.dur:g}" if f.dur else "") for f in faults))
    period = 1.0 / a.rate
    stats = {"sent": 0, "faults": []}
    index = 0
    held_reorder = None
    pending_combine = []
    mode = {"fragment": False, "duplicate": False, "reorder": False, "combine": 0}
    t_start = time.monotonic()
    dev_offset = 0.0                     # device clock offset after a reboot
    k = 0
    while True:
        t = k * period
        if t >= a.duration:
            break
        delay = t_start + t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        k += 1
        active = frozenset(f.name for f in faults if f.name in WINDOW and f.active(t))
        for f in faults:
            if f.name in ONE_SHOT and not f.done and t >= f.at:
                f.done = True
                stats["faults"].append(f"{f.name}@{t:.2f}")
                if f.name == "malformed":
                    link.write(b'{"device_id": "synthetic-dermis-dev", "frame": 12, "pods": {\n')
                    link.write(b'\xff\xfe\xfd not utf-8\n')
                elif f.name == "oversize":
                    link.write(b'{"pad":"' + b"x" * a.oversize_bytes + b'"}\n')
                elif f.name in ("disconnect", "reboot"):
                    link.close()
                    log(f"[sender] {f.name}: closed connection for {f.dur:g} s")
                    time.sleep(f.dur)
                    if f.name == "reboot":
                        gen.reboot()
                        index, dev_offset = 0, -t
                    link.connect()
                elif f.name == "combine":
                    mode["combine"] = 3
                elif f.name in mode:
                    mode[f.name] = True
                elif f.name == "nonfinite":
                    active = active | {"nonfinite"}
        if "pause" in active:
            index += 1                   # the device keeps producing frames; they are just not sent
            continue
        n_sub = 4 if "burst" in active else 1
        for j in range(n_sub):
            ts = t + j * period / n_sub
            frame = (gen.draft_frame_at if a.format == "draft" else gen.frame_at)(ts, index, ts + dev_offset, active)
            index += 1
            data = compact(frame)
            if held_reorder is not None:
                link.write(data + held_reorder)          # n+1 before n
                held_reorder = None
                stats["sent"] += 2
                continue
            if mode["reorder"]:
                mode["reorder"] = False
                held_reorder = data
                continue
            if mode["combine"]:
                pending_combine.append(data)
                mode["combine"] -= 1
                if mode["combine"] == 0:
                    link.write(b"".join(pending_combine))
                    stats["sent"] += len(pending_combine)
                    pending_combine = []
                continue
            if mode["fragment"]:
                mode["fragment"] = False
                cut = [len(data) // 3, 2 * len(data) // 3]
                for piece in (data[:cut[0]], data[cut[0]:cut[1]], data[cut[1]:]):
                    link.write(piece)
                    time.sleep(0.004)
            else:
                link.write(data)
            if mode["duplicate"]:
                mode["duplicate"] = False
                link.write(data)
                stats["sent"] += 1
            stats["sent"] += 1
    return stats


def add_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--host", default=env_default("DERMIS_TCP_HOST", "127.0.0.1"),
                    help="receiver address: 127.0.0.1 locally, the Mac's LAN IP from another machine")
    sp.add_argument("--port", type=int, default=env_default("DERMIS_TCP_PORT", 8765))
    sp.add_argument("--mode", choices=["fixture", "replay", "synthetic"], default="synthetic")
    sp.add_argument("--profile", choices=PROFILES, default="pinch",
                    help="synthetic: pinch (grip open/close + tilt), updown (pitch only; arm --tilt-z) or "
                         "punch (fist + 30 cm punch every 2.5 s)")
    sp.add_argument("--format", choices=["v1", "draft"], default="v1",
                    help="synthetic: v1 decoded frames, or the proposed draft shape (wrist_imu, no imu on middle/thenar)")
    sp.add_argument("--write", metavar="PATH", help="synthetic: save frames to a JSONL file instead of sending")
    sp.add_argument("--rate", type=float, default=30.0, help="synthetic frame rate / replay fallback, Hz")
    sp.add_argument("--duration", type=float, default=60.0, help="synthetic: seconds to run")
    sp.add_argument("--stationary-s", type=float, default=3.0, help="synthetic: still, unloaded start (calibrate here)")
    sp.add_argument("--grip-period-s", type=float, default=4.0)
    sp.add_argument("--no-tilt", action="store_true", help="synthetic: keep the gravity vector fixed")
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--faults", default="", help="e.g. 'pause@10:3,malformed@5' or 'ci' (see module doc)")
    sp.add_argument("--oversize-bytes", type=int, default=40000)
    sp.add_argument("--path", help="replay: .jsonl/.json decoded frames, or a .bin wire capture")
    sp.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    sp.add_argument("--loop", action="store_true", help="replay: loop forever")
    sp.add_argument("--repeat", type=int, default=1, help="fixture: send N times (duplicates after the first)")
    sp.add_argument("--connect-timeout", type=float, default=15.0)


def cmd_send_test(a) -> int:
    log = lambda s: print(s, flush=True)
    if a.write:
        return write_synthetic(a, log)
    link = Link(a.host, a.port, a.connect_timeout, log)
    try:
        link.connect()
        if a.mode == "fixture":
            stats = run_fixture(link, a, log)
        elif a.mode == "replay":
            if not a.path:
                log("[sender] --mode replay needs --path")
                return 2
            stats = run_replay(link, a, log)
        else:
            stats = run_synthetic(link, a, log)
    except ConnectionRefusedError as e:
        log(f"[sender] refused: {e}")
        return 3
    except (ConnectionError, BrokenPipeError, OSError) as e:
        log(f"[sender] connection error: {e}")
        return 4
    except KeyboardInterrupt:
        log("[sender] interrupted")
        return 130
    finally:
        link.close()
    log(f"[sender] done: {json.dumps(stats)} ({link.bytes} bytes)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dermis send-test", description=__doc__.split("\n")[0])
    add_args(ap)
    return cmd_send_test(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
