"""Real-time pick-up app: glove frames -> PickupSession (physics, DIRECT) -> display-only PyBullet GUI window.

    python -m dermis glass pickup                              # live glove Wi-Fi (ws://192.168.4.1:81/ws)
    python -m dermis glass pickup --source replay --file rec.jsonl
    python -m dermis glass pickup --source synthetic --profile cycle   # no hardware: SYNTHETIC grasps

Glove: firmware v0 hotspot "Dermis-S3-XXXX", 10 Hz frames over a WebSocket. Check the stream with
`python -m dermis glass glove-check`.

Grip per pod: the on-device `force_n` when the venue calibration found it responds to a real grip,
otherwise the field signal from dermis.glass.glove_cal.GripSensor: movement of b_ut along each pad's load
axis (from the glove calibration export) relative to the open-hand rest taken over the first second, ZERO
unless it clears that pod's noise-based activation level, scaled by the venue calibration. The rest
follows slow drift while the hand is open, so a glove at rest does not creep into a grip.
The window always shows which one is in use.
Physics runs at a fixed 50 Hz control clock on the latest frame; if no frame arrives for STALE_S the
session keeps its last command and the window says STALE (nothing is invented).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from . import hand3
from .glove_cal import GripSensor, pod_b
from .pickup import PickupConfig, PickupSession, combined_grip
from .sensors import SyntheticGlove

PODS = ("thumb_tip", "index_tip", "middle_tip", "thenar_apb")
FINGERS = ("thumb_tip", "index_tip", "middle_tip")
STALE_S = 0.5                            # firmware streams at 10 Hz; allow a few late Wi-Fi frames
BASELINE_S = 1.0
DEFAULT_URL = "ws://192.168.4.1:81/ws"


# -- grip from a raw glove frame ----------------------------------------------------------------------
POD_ALIASES = {"thenar_apb": ("thenar_apb", "palm_center")}   # firmware v3 names the palm pod palm_center
MIN_FORCE_GRIP_N = 0.3                   # a normal hold must read at least this on-device to trust force_n
GRIP_MEASURE = "span_projection_active_top2"   # calibration files made with an older measure must be redone
FREE_MARGIN = 1.5                        # activation = this x the largest signal free hand movement produced


def _mag(frame: dict, pod: str) -> dict:
    pods = frame.get("pods", {})
    for name in POD_ALIASES.get(pod, (pod,)):
        if name in pods:
            return (pods[name] or {}).get("mag") or {}
    return {}


def frame_b(frame: dict) -> dict:
    out = {}
    for k in PODS:
        mag = _mag(frame, k)
        b = mag.get("b_ut")
        out[k] = np.array(b, float) if mag.get("valid") and isinstance(b, list) and len(b) == 3 else None
    return out


def device_force(frame: dict) -> dict | None:
    """Per-pod on-device force_n (N), or None unless every finger pod reports a valid one."""
    out = {}
    for k in PODS:
        m = _mag(frame, k)
        f = m.get("force_n")
        ok = m.get("valid") and m.get("force_valid") and isinstance(f, (int, float)) and not isinstance(f, bool)
        out[k] = float(f) if ok else None
    return out if all(out[k] is not None for k in FINGERS) else None


def field_change(frame: dict, baseline: dict | None) -> dict:
    """Per-pod |b_ut - open-hand baseline| (uT), None where the pod or its baseline is missing."""
    b = frame_b(frame)
    return {k: (None if b[k] is None or baseline is None or baseline.get(k) is None
                else float(np.linalg.norm(b[k] - baseline[k]))) for k in PODS}


def frame_imu(frame: dict) -> dict | None:
    imu = frame.get("imu") or {}
    a, g = imu.get("acc_mps2"), imu.get("gyr_rads")
    if not imu.get("valid") or not (isinstance(a, list) and len(a) == 3 and isinstance(g, list) and len(g) == 3):
        return None
    return {"acc": np.array(a, float), "gyr": np.array(g, float)}


def pod_forces(frame: dict, baseline: dict | None, ut_per_n: float, mode: str = "auto") -> tuple[dict, str]:
    """Per-pod grip (N or None if the pod is invalid) and the mode used ('force_n' or 'proxy').
    mode 'auto' uses on-device force_n when valid; 'proxy' always uses the raw-field change (chosen by the
    venue calibration when force_n does not respond to a real grip)."""
    if mode != "proxy":
        f = device_force(frame)
        if f is not None:
            return f, "force_n"
    return {k: (None if v is None else v / ut_per_n) for k, v in field_change(frame, baseline).items()}, "proxy"


# -- sources (each: poll() -> list of new raw frames) --------------------------------------------------
def _ws_worker(url: str, conn):
    import asyncio

    from ..server.websocket_bridge import _frame, _open_websocket, _read_message

    async def run():
        while True:
            writer = None
            try:
                reader, writer = await _open_websocket(url, [])
                conn.send(("status", f"connected {url}"))
                while True:
                    op, raw = await _read_message(reader)
                    if op == 0x8:
                        raise ConnectionError("glove closed the stream")
                    if op == 0x9:
                        writer.write(_frame(0xA, raw, mask=True))
                        await writer.drain()
                    elif op == 0x1:
                        conn.send(("frame", raw.decode("utf-8", errors="replace")))
            except (ConnectionError, OSError, asyncio.IncompleteReadError, ValueError) as e:
                conn.send(("status", f"glove unavailable ({e}); retrying"))
                await asyncio.sleep(1.0)
            finally:
                if writer is not None:
                    writer.close()
    asyncio.run(run())


class WebSocketSource:
    """Glove Wi-Fi stream read in a child process (PyBullet holds the GIL while stepping). A thread in this
    process drains the pipe continuously, so frames never back up while the main thread is busy or waiting
    at a prompt; poll() returns what arrived since the last call. Undecodable frames are counted in stats."""
    label = "LIVE GLOVE"

    def __init__(self, url: str = DEFAULT_URL):
        self.conn, child = mp.Pipe(duplex=False)
        self.proc = mp.Process(target=_ws_worker, args=(url, child), daemon=True)
        self.proc.start()
        self.status = f"connecting {url}"
        self.stats = {"frames": 0, "bad_json": 0, "connects": 0, "last_bad": None}
        self.q = deque(maxlen=2000)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        while True:
            try:
                kind, payload = self.conn.recv()
            except (EOFError, OSError):
                return
            if kind == "status":
                self.status = payload
                self.stats["connects"] += payload.startswith("connected")
                print(payload)
                continue
            try:
                self.q.append(json.loads(payload))
                self.stats["frames"] += 1
            except json.JSONDecodeError as e:
                self.stats["bad_json"] += 1
                self.stats["last_bad"] = f"{e}: {payload[:160]!r}"

    def poll(self) -> list:
        out = []
        while self.q:
            out.append(self.q.popleft())
        return out

    def close(self):
        self.proc.terminate()


class ReplaySource:
    """Recorded raw glove frames (JSONL), paced by their device timestamps. Loops."""
    label = "REPLAY"

    def __init__(self, path: str):
        self.frames = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        assert self.frames, f"no frames in {path}"
        self.i, self.t0, self.status = 0, None, f"replay {path}"
        self.dev0 = self.frames[0]["t_device_us"]

    def poll(self) -> list:
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
        out = []
        while self.i < len(self.frames) and (self.frames[self.i]["t_device_us"] - self.dev0) / 1e6 <= now - self.t0:
            out.append(self.frames[self.i])
            self.i += 1
        if self.i >= len(self.frames):
            self.i, self.t0 = 0, now + 2.0
        return out

    def close(self):
        pass


# SYNTHETIC grips for the default grip_gain (normal hold of ~1.9 N = the calibrated normal): 'light' is below
# the grasp threshold (never grips), 'crush' is ~5x normal (passes the simulated crack level).
PROFILES = {"normal": {"thumb_tip": 2.0, "index_tip": 1.8, "middle_tip": 1.6, "thenar_apb": 0.8},
            "light": {"thumb_tip": 0.1, "index_tip": 0.08, "middle_tip": 0.06, "thenar_apb": 0.04},
            "crush": {"thumb_tip": 9.81, "index_tip": 9.81, "middle_tip": 9.81, "thenar_apb": 6.0}}


class SyntheticSource:
    """SYNTHETIC grasp-hold-release cycles as frames in the provisional-force schema (flag 'SYNTHETIC')."""
    label = "SYNTHETIC"

    def __init__(self, profile: str = "cycle", hz: int = 50):
        self.order = ["normal", "light", "crush"] if profile == "cycle" else [profile]
        self.glove = SyntheticGlove(0)
        self.glove_period = 1.0 / hz
        self.t0, self.n, self.status = None, 0, f"SYNTHETIC profile {profile}"

    def level(self, t: float) -> dict:
        cyc = 9.0
        k, u = int(t // cyc), t % cyc
        lv = PROFILES[self.order[k % len(self.order)]]
        a = 0.0 if (u < 1.5 or u > 7.0) else min(1.0, (u - 1.5) / 0.3)
        return {pod: v * a for pod, v in lv.items()}

    @staticmethod
    def hand_acc(t: float) -> float:
        """SYNTHETIC vertical hand acceleration: lift at 2.8 s (push up, then brake), set down at 5.8 s."""
        u = t % 9.0
        for t0, sign in ((2.8, 1.0), (5.8, -1.0)):
            if t0 <= u < t0 + 0.25:
                return 3.0 * sign
            if t0 + 0.25 <= u < t0 + 0.5:
                return -3.0 * sign
        return 0.0

    def poll(self) -> list:
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
        out = []
        while self.n * self.glove_period <= now - self.t0:
            lv = self.level(self.n * self.glove_period)
            out.append(self.glove.make({k: {"normal": v, "shear_z": 0.0, "shear_h": 0.0} for k, v in lv.items()},
                                       self.hand_acc(self.n * self.glove_period)))
            self.n += 1
        return out

    def close(self):
        pass


# -- display ------------------------------------------------------------------------------------------
BAR_SEG, BAR_N_PER_SEG = 12, 1.0
BAR_LABELS = {"thumb_tip": "thumb", "index_tip": "index", "middle_tip": "middle", "thenar_apb": "palm", "grip": "GRIP"}
TEXT_X = 0.40                            # the camera looks along -y, so +x is the left of the screen


class PickupView:
    def __init__(self, sim, source_label: str, grip_label: str = ""):
        self.c = p.connect(p.GUI)
        P = self.P = dict(physicsClientId=self.c)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, **P)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), **P)
        p.loadURDF("plane.urdf", **P)
        tv = p.createVisualShape(p.GEOM_BOX, halfExtents=sim.table_half, rgbaColor=(0.55, 0.4, 0.28, 1), **P)
        p.createMultiBody(0, -1, tv, p.getBasePositionAndOrientation(sim.table, **sim._p)[0], **P)
        cv = p.createVisualShape(p.GEOM_CYLINDER, radius=sim.cfg.cup_radius, length=sim.cfg.cup_height, rgbaColor=(0.95, 0.95, 0.9, 1), **P)
        self.cup = p.createMultiBody(0, -1, cv, (0, 0, -1), **P)
        self.hand, self.joints, _ = hand3.build(self.c, tuple(sim.home), visual=True)
        for j in range(-1, p.getNumJoints(self.hand, **P)):
            p.setCollisionFilterGroupMask(self.hand, j, 0, 0, **P)
        self.crack_rgba = False
        seg = p.createVisualShape(p.GEOM_BOX, halfExtents=(0.013, 0.004, 0.006), rgbaColor=(0.23, 0.5, 0.95, 1), **P)
        hot = p.createVisualShape(p.GEOM_BOX, halfExtents=(0.013, 0.004, 0.006), rgbaColor=(0.9, 0.2, 0.15, 1), **P)
        top = sim.cfg.table_top
        self.bars = []
        for i, k in enumerate(BAR_LABELS):
            x = 0.37 - 0.066 * i
            p.addUserDebugText(BAR_LABELS[k], (x + 0.018, 0.0, top + 0.004), (0.1, 0.1, 0.1), 0.75, **P)
            self.bars.append([(p.createMultiBody(0, -1, hot if n >= 9 else seg, (0, 0, -1), **P), (x, 0.0, top + 0.02 + 0.015 * n))
                              for n in range(BAR_SEG)])
        p.addUserDebugText(f"{source_label}   {grip_label}", (TEXT_X, 0.05, top + 0.33), (0.1, 0.1, 0.5), 1.1, **P)
        p.addUserDebugText("bars: grip per pod in N (red above 9 N); GRIP = mean of the two strongest.  R reset  Q quit",
                           (TEXT_X, 0.0, top + 0.215), (0.3, 0.3, 0.3), 0.8, **P)
        p.addUserDebugText("SIMULATION. Lift height is fixed: the glove senses grip and motion, not position.",
                           (TEXT_X, 0.05, top - 0.05), (0.3, 0.3, 0.3), 0.8, **P)
        p.addUserDebugText("Crack = simulated force threshold.", (TEXT_X, 0.05, top - 0.075), (0.3, 0.3, 0.3), 0.8, **P)
        p.resetDebugVisualizerCamera(0.68, 180, -18, (0.13, 0.05, top + 0.1), **P)
        self.text, self.shown = None, None

    def sync(self, sim, forces: dict | None, banner: str, color):
        P = self.P
        p.resetBasePositionAndOrientation(self.hand, sim.hand_pos().tolist(), (0, 0, 0, 1), **P)
        for j, jj in zip(self.joints, sim.joints):
            p.resetJointState(self.hand, j, p.getJointState(sim.hand, jj, **sim._p)[0], **P)
        pos, orn = p.getBasePositionAndOrientation(sim.cup, **sim._p)
        p.resetBasePositionAndOrientation(self.cup, pos, orn, **P)
        if sim.cracked != self.crack_rgba:
            p.changeVisualShape(self.cup, -1, rgbaColor=(0.9, 0.2, 0.15, 1) if sim.cracked else (0.95, 0.95, 0.9, 1), **P)
            self.crack_rgba = sim.cracked
        vals = None if not forces else {**forces, "grip": combined_grip(forces)}
        for k, segs in zip(BAR_LABELS, self.bars):
            n = 0 if not vals or vals.get(k) is None else int(round(min(vals[k], BAR_SEG * BAR_N_PER_SEG) / BAR_N_PER_SEG))
            for i, (b, at) in enumerate(segs):
                p.resetBasePositionAndOrientation(b, at if i < n else (0, 0, -1), (0, 0, 0, 1), **P)
        if banner != self.shown:                                # GUI text redraws are slow: only on change
            kw = {"replaceItemUniqueId": self.text} if self.text is not None else {}
            self.text = p.addUserDebugText(banner, (TEXT_X, 0.05, sim.cfg.table_top + 0.27), color, 1.5, **P, **kw)
            self.shown = banner

    def keys(self) -> set:
        ev = p.getKeyboardEvents(**self.P)
        return {chr(k) for k, s in ev.items() if s & p.KEY_WAS_TRIGGERED and k < 256}

    def alive(self) -> bool:
        return p.isConnected(self.c)


BANNERS = {"OPEN": ("READY - grasp the cup", (0.1, 0.1, 0.1)), "CLOSING": ("GRASPING", (0.1, 0.3, 0.7)),
           "GRIPPED": ("GRIPPED - now lift the cup", (0.1, 0.3, 0.7)),
           "LIFTING": ("LIFTING", (0.1, 0.3, 0.7)), "HELD": ("HOLDING", (0.08, 0.55, 0.25)),
           "SETTING_DOWN": ("SETTING DOWN", (0.1, 0.3, 0.7)),
           "LOWERING": ("RELEASING", (0.1, 0.3, 0.7)), "OPENING": ("RELEASED", (0.1, 0.1, 0.1)),
           "DROPPED": ("DROPPED - grip too weak", (0.8, 0.15, 0.1)),
           "CRACKED": ("CRACKED - grip too strong (simulated)", (0.8, 0.15, 0.1))}


# -- venue calibration -----------------------------------------------------------------------------------
CALIBRATION = "runs/glass/pickup_calibration.json"
NORMAL_HOLD_N = 4.5                      # a calibrated "normal hold" reads as this many (scaled) newtons
SAFE_EFFORT_NM = 0.054                   # middle of the sim's hold window (fixed-effort sweep: 0.04-0.08 held)


def _collect(source, seconds: float) -> list:
    source.poll()                                               # discard anything queued before this window
    out, end = [], time.monotonic() + seconds
    while time.monotonic() < end:
        out += source.poll()
        time.sleep(0.01)
    return out


def _no_stream(source, got: int, seconds: float) -> RuntimeError:
    st = getattr(source, "stats", {})
    return RuntimeError(f"only {got} glove frames in {seconds:.0f} s. Source status: {getattr(source, 'status', '?')}; "
                        f"stats: {st}. Run `python -m dermis glass glove-check` for details.")


def calibrate(source, out: str = CALIBRATION, open_s: float = 2.0, hold_s: float = 3.0, prompt=input,
              wait_s: float = 5.0, export: dict | None = None, free_s: float = 4.0) -> dict:
    """Open hand, a normal hold of the real cup, then free hand movement touching nothing.
    The hold sets the scale (normal hold -> mid hold-window in the sim): valid on-device force_n sets
    grip_gain, otherwise the field signal's ut_per_n. The free movement sets each pod's activation level
    FREE_MARGIN above the largest signal that movement alone produced, so moving the glove is not a grip."""
    print("waiting for the glove stream...")
    got = _collect(source, wait_s)
    if len(got) < 5:
        raise _no_stream(source, len(got), wait_s)
    print(f"glove streaming ({len(got) / wait_s:.1f} frames/s)")
    prompt("Step 1/3: hold the hand OPEN and still, then press Enter (records 2 s)... ")
    opened = _collect(source, open_s)
    prompt("Step 2/3: pick up the real cup and HOLD it normally, then press Enter (records 3 s)... ")
    held = _collect(source, hold_s)
    prompt(f"Step 3/3: put the cup down. Move the hand around, wave it, open and close the fingers - TOUCH NOTHING. "
           f"Press Enter and keep moving (records {free_s:.0f} s)... ")
    free = _collect(source, free_s)
    if min(len(opened), len(held), len(free)) < 5:
        raise _no_stream(source, min(len(opened), len(held), len(free)), min(open_s, hold_s, free_s))
    sensor = GripSensor(export)                                 # the same signal the live run uses
    sensor.set_rest(opened)

    def med(rows):
        return {k: (round(float(np.median([r[k] for r in rows if r[k] is not None])), 3)
                    if any(r[k] is not None for r in rows) else None) for k in PODS}

    def signals(frames):
        return [{k: (None if (b := pod_b(f, k, sensor.info(k)["ch"])) is None else sensor.raw_signal(b, k)) for k in PODS}
                for f in frames]
    forces = [device_force(f) for f in held]
    force_hold = med([f for f in forces if f is not None]) if sum(f is not None for f in forces) >= len(held) // 2 else None
    sig_hold, sig_open = med(signals(held)), med(signals(opened))
    free_sig = signals(free)
    free_max = {k: (round(float(max(r[k] for r in free_sig if r[k] is not None)), 1)
                    if any(r[k] is not None for r in free_sig) else None) for k in PODS}
    on_ut = {k: round(max(sensor.thresholds(k)[0], FREE_MARGIN * (free_max[k] or 0.0)), 1) for k in PODS}
    registered = [k for k in PODS if (sig_hold[k] or 0.0) > on_ut[k]]
    for k in PODS:
        if k not in registered and free_max[k] is not None:
            print(f"  {k}: hold {sig_hold[k]} uT is not above its activation {on_ut[k]} uT "
                  f"(free movement reached {free_max[k]} uT) - this pod cannot tell a grip from movement; ignored")
    flags = sorted({fl for f in held for k in PODS for fl in (_mag(f, k).get("flags") or [])})
    top2 = lambda d: combined_grip({k: d[k] for k in registered}) if d and registered else 0.0
    cal = {"grip_measure": GRIP_MEASURE, "glove_export": bool(export), "force_hold_n": force_hold,
           "signal_hold_ut": sig_hold, "signal_open_ut": sig_open, "free_motion_max_ut": free_max,
           "activation_ut": on_ut, "pods_registered": registered, "flags_during_hold": flags,
           "frames": [len(opened), len(held), len(free)], "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(f"pods that registered the hold (signal above activation): {registered or 'none'}")
    if not registered:
        raise RuntimeError(f"no pod's hold signal cleared its activation level. Hold {sig_hold} uT, activation {on_ut} uT "
                           f"(free movement {free_max} uT, open hand {sig_open} uT). Hold the cup firmly with the pads "
                           "on it, and keep step 3 to moving in the air.")
    if len(registered) == 1:
        print("  only one pod registered: grasps will need that pod pressed firmly on its own")
    if force_hold and top2(force_hold) >= MIN_FORCE_GRIP_N:
        cal.update(mode="force_n", grip_gain=SAFE_EFFORT_NM / top2(force_hold), ut_per_n=None)
    else:
        why = ("on-device force_n read %.2f N during the hold" % top2(force_hold)) if force_hold else "no valid on-device force_n"
        print(f"{why} (flags {flags}); using the calibrated field signal instead.")
        cal.update(mode="proxy", grip_gain=SAFE_EFFORT_NM / NORMAL_HOLD_N, ut_per_n=top2(sig_hold) / NORMAL_HOLD_N,
                   note=f"{why}; field signal chosen")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(cal, indent=1))
    return cal


def glove_check(source, seconds: float = 5.0) -> dict:
    """What the glove is actually sending: rate, decode failures, config, per-pod validity and force state."""
    _collect(source, 1.5)                                       # let the connection come up
    t0 = time.monotonic()
    frames = _collect(source, seconds)
    dt = time.monotonic() - t0
    rep = {"frames": len(frames), "rate_hz": round(len(frames) / dt, 1), "status": getattr(source, "status", None),
           "stats": getattr(source, "stats", None)}
    if frames:
        last = frames[-1]
        rep["config_id"] = last.get("config_id")
        rep["frame_flags"] = sorted({fl for f in frames for fl in f.get("flags", [])})
        ts = [f.get("t_device_us") for f in frames if isinstance(f.get("t_device_us"), int)]
        if len(ts) > 1:
            rep["device_rate_hz"] = round((len(ts) - 1) / ((ts[-1] - ts[0]) / 1e6), 1)
        rep["pod_names_sent"] = sorted({n for f in frames for n in f.get("pods", {})})
        pods = {}
        for k in PODS:
            mags = [_mag(f, k) for f in frames]
            pods[k] = {"valid": f"{sum(bool(m.get('valid')) for m in mags)}/{len(mags)}",
                       "force_valid": f"{sum(bool(m.get('force_valid')) for m in mags)}/{len(mags)}",
                       "flags": sorted({fl for m in mags for fl in (m.get("flags") or [])}),
                       "last_b_ut": mags[-1].get("b_ut"), "last_force_n": mags[-1].get("force_n"),
                       "last_load_signal_ut": mags[-1].get("load_signal_ut")}
        rep["pods"] = pods
        imus = [frame_imu(f) for f in frames]
        ok = [m for m in imus if m is not None]
        rep["imu"] = {"valid": f"{len(ok)}/{len(imus)}"}
        if ok:
            an = np.array([np.linalg.norm(m["acc"]) for m in ok])
            gn = np.array([m["gyr"] for m in ok])
            rep["imu"].update(acc_norm_mean=round(float(an.mean()), 3), acc_norm_std=round(float(an.std()), 3),
                              gyro_std=round(float(np.linalg.norm(gn.std(0))), 4),
                              hint="keep the hand still: std well below the motion thresholds (0.8 m/s^2, 0.5 rad/s)")
        rep["grip_mode"] = "force_n" if device_force(last) is not None else "proxy (on-device force_n not valid yet)"
    return rep


# -- main loop ----------------------------------------------------------------------------------------
def run(source, cfg: PickupConfig | None = None, gui: bool = True, record: str | None = None,
        ut_per_n: float = 160.0, duration_s: float | None = None, grip_mode: str = "auto",
        export: dict | None = None, activation_ut: dict | None = None) -> dict:
    sess = PickupSession(cfg)
    sensor = GripSensor(export, ut_per_n, activation_ut=activation_ut)
    grip_label = ("grip: field signal" + (", glove calibration" if export else ", NO glove calibration")
                  if grip_mode == "proxy" else "grip: force_n if valid, else field signal")
    view = PickupView(sess.sim, source.label, grip_label) if gui else None
    rec = open(record, "w", buffering=1) if record else None
    dt = 1.0 / sess.cfg.control_hz
    latest, last_rx, rest_frames, field_forces, mode = None, None, [], None, "?"
    start = time.monotonic()
    next_tick, next_draw = start, start
    n_frames, n_rezero = 0, 0
    try:
        while True:
            now = time.monotonic()
            if duration_s is not None and now - start >= duration_s:
                break
            for fr in source.poll():
                latest, last_rx = fr, now
                n_frames += 1
                if rec:
                    rec.write(json.dumps({"rx": round(now - start, 4), "frame": fr}) + "\n")
                if field_forces is None and rest_frames is not None:
                    rest_frames.append(fr)                         # open-hand rest, first BASELINE_S
                    if now - start >= BASELINE_S and len(rest_frames) >= 5:
                        sensor.set_rest(rest_frames)
                        rest_frames = None
                if rest_frames is None:                            # once per frame: activation + drift tracking
                    field_forces = sensor.update(fr, gripping=sess.trig.state != "OPEN")
                    if len(sensor.rezeroed) > n_rezero:
                        sess.trig.events += [{"t": round(sess.t, 3), "event": "rezero", "pod": k} for k in sensor.rezeroed[n_rezero:]]
                        n_rezero = len(sensor.rezeroed)
            calibrating = latest is not None and field_forces is None
            ticks = 0
            while now >= next_tick and ticks < 3:                 # fixed 50 Hz control clock
                stale = last_rx is None or now - last_rx > STALE_S
                forces = imu = None
                if not stale and not calibrating:
                    dev = device_force(latest) if grip_mode != "proxy" else None
                    forces, mode = (dev, "force_n") if dev is not None else (field_forces, "proxy")
                    imu = frame_imu(latest)
                r = sess.tick(forces, imu)
                if rec:
                    rec.write(json.dumps({"tick": r, "stale": stale, "mode": mode}) + "\n")
                next_tick += dt
                ticks += 1
            if now > next_tick + 0.5:                              # fell far behind (window dragged etc.): resync
                next_tick = now
            if view:
                if now >= next_draw:
                    if calibrating or latest is None:
                        banner = ("WAITING FOR GLOVE - " + getattr(source, "status", ""), (0.5, 0.5, 0.5)) if latest is None \
                            else ("CALIBRATING - keep the hand open", (0.5, 0.4, 0.0))
                    elif last_rx is None or now - last_rx > STALE_S:
                        banner = ("GLOVE STALE - holding last command", (0.8, 0.5, 0.0))
                    else:
                        banner = BANNERS.get(sess.trig.state, (sess.trig.state, (0, 0, 0)))
                    shown = None if (calibrating or latest is None) else (device_force(latest) if mode == "force_n" else field_forces)
                    view.sync(sess.sim, shown, *banner)
                    next_draw = now + 1 / 30
                keys = view.keys()
                if "q" in keys or not view.alive():
                    break
                if "r" in keys:
                    sess.sim.reset()
                    sess.trig.reset()
                    sess.trig.events.append({"t": round(sess.t, 3), "event": "reset (key)"})
            sleep = min(next_tick, next_draw if view else next_tick) - time.monotonic()
            if sleep > 0:
                time.sleep(min(sleep, 0.005))
    finally:
        source.close()
        if rec:
            rec.close()
        events = list(sess.trig.events)
        sess.close()
        if view and view.alive():
            p.disconnect(view.c)
    return {"events": events, "frames": n_frames, "mode": mode, "resets": sess.resets}
