"""Real-time pick-up app: glove frames -> PickupSession (physics, DIRECT) -> display-only PyBullet GUI window.

    python -m dermis glass pickup                              # live glove Wi-Fi (ws://192.168.4.1:81/ws)
    python -m dermis glass pickup --source replay --file rec.jsonl
    python -m dermis glass pickup --source synthetic --profile cycle   # no hardware: SYNTHETIC grasps

Glove: firmware v0 hotspot "Dermis-S3-XXXX", 10 Hz frames over a WebSocket. Check the stream with
`python -m dermis glass glove-check`.

Grip per pod: the on-device `force_n` when every finger pod reports a valid one (the firmware sends null
until the glove has been tared and spanned on the device), otherwise an UNCALIBRATED proxy,
|b_ut - open-hand baseline| / ut_per_n, with the baseline captured over the first second while the judge
holds the hand open. The window always shows which one is in use.
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
from .pickup import PickupConfig, PickupSession
from .sensors import SyntheticGlove

PODS = ("thumb_tip", "index_tip", "middle_tip", "thenar_apb")
FINGERS = ("thumb_tip", "index_tip", "middle_tip")
STALE_S = 0.5                            # firmware streams at 10 Hz; allow a few late Wi-Fi frames
BASELINE_S = 1.0
DEFAULT_URL = "ws://192.168.4.1:81/ws"


# -- grip from a raw glove frame ----------------------------------------------------------------------
def frame_b(frame: dict) -> dict:
    out = {}
    for k in PODS:
        mag = (frame.get("pods", {}).get(k) or {}).get("mag") or {}
        b = mag.get("b_ut")
        out[k] = np.array(b, float) if mag.get("valid") and isinstance(b, list) and len(b) == 3 else None
    return out


def device_force(frame: dict) -> dict | None:
    """Per-pod on-device force_n (N), or None unless every finger pod reports a valid one."""
    out = {}
    for k in PODS:
        m = (frame.get("pods", {}).get(k) or {}).get("mag") or {}
        f = m.get("force_n")
        ok = m.get("valid") and m.get("force_valid") and isinstance(f, (int, float)) and not isinstance(f, bool)
        out[k] = float(f) if ok else None
    return out if all(out[k] is not None for k in FINGERS) else None


def pod_forces(frame: dict, baseline: dict | None, ut_per_n: float) -> tuple[dict, str]:
    """Per-pod grip (N or None if the pod is invalid) and the mode used ('force_n' or 'proxy')."""
    f = device_force(frame)
    if f is not None:
        return f, "force_n"
    b = frame_b(frame)
    return {k: (None if b[k] is None or baseline is None or baseline.get(k) is None
                else float(np.linalg.norm(b[k] - baseline[k])) / ut_per_n) for k in PODS}, "proxy"


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


PROFILES = {"normal": {"thumb_tip": 5.0, "index_tip": 4.5, "middle_tip": 4.0, "thenar_apb": 2.0},
            "weak": {"thumb_tip": 1.6, "index_tip": 1.6, "middle_tip": 1.6, "thenar_apb": 0.5},
            "crush": {"thumb_tip": 9.81, "index_tip": 9.81, "middle_tip": 9.81, "thenar_apb": 6.0}}


class SyntheticSource:
    """SYNTHETIC grasp-hold-release cycles as frames in the provisional-force schema (flag 'SYNTHETIC')."""
    label = "SYNTHETIC"

    def __init__(self, profile: str = "cycle", hz: int = 50):
        self.order = ["normal", "weak", "crush"] if profile == "cycle" else [profile]
        self.glove = SyntheticGlove(0)
        self.glove_period = 1.0 / hz
        self.t0, self.n, self.status = None, 0, f"SYNTHETIC profile {profile}"

    def level(self, t: float) -> dict:
        cyc = 9.0
        k, u = int(t // cyc), t % cyc
        lv = PROFILES[self.order[k % len(self.order)]]
        a = 0.0 if (u < 1.5 or u > 7.0) else min(1.0, (u - 1.5) / 0.3)
        return {pod: v * a for pod, v in lv.items()}

    def poll(self) -> list:
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
        out = []
        while self.n * self.glove_period <= now - self.t0:
            lv = self.level(self.n * self.glove_period)
            out.append(self.glove.make({k: {"normal": v, "shear_z": 0.0, "shear_h": 0.0} for k, v in lv.items()}))
            self.n += 1
        return out

    def close(self):
        pass


# -- display ------------------------------------------------------------------------------------------
BAR_SEG, BAR_N_PER_SEG = 12, 1.0


class PickupView:
    def __init__(self, sim, source_label: str):
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
        for i, k in enumerate(PODS):
            x = 0.13 + 0.045 * i
            p.addUserDebugText(k.replace("_tip", ""), (x + 0.015, 0.0, top + 0.004), (0.1, 0.1, 0.1), 0.9, **P)
            self.bars.append([(p.createMultiBody(0, -1, hot if n >= 9 else seg, (0, 0, -1), **P), (x, 0.0, top + 0.02 + 0.015 * n))
                              for n in range(BAR_SEG)])
        p.addUserDebugText(f"{source_label}", (-0.28, 0.05, top + 0.30), (0.1, 0.1, 0.5), 1.3, **P)
        p.addUserDebugText("SIMULATION: lift is triggered by the grasp (height not tracked); grip strength is live per finger. "
                           "Crack = simulated force threshold.", (-0.28, 0.05, top - 0.07), (0.3, 0.3, 0.3), 0.9, **P)
        p.addUserDebugText("bars: glove grip per pod (N, red above 9 N)   keys: R reset  Q quit", (0.28, 0.0, top + 0.22),
                           (0.3, 0.3, 0.3), 0.8, **P)
        p.resetDebugVisualizerCamera(0.55, 180, -18, (0.07, 0.05, top + 0.09), **P)
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
        for k, segs in zip(PODS, self.bars):
            n = 0 if not forces or forces.get(k) is None else int(round(min(forces[k], BAR_SEG * BAR_N_PER_SEG) / BAR_N_PER_SEG))
            for i, (b, at) in enumerate(segs):
                p.resetBasePositionAndOrientation(b, at if i < n else (0, 0, -1), (0, 0, 0, 1), **P)
        if banner != self.shown:                                # GUI text redraws are slow: only on change
            kw = {"replaceItemUniqueId": self.text} if self.text is not None else {}
            self.text = p.addUserDebugText(banner, (-0.28, 0.05, sim.cfg.table_top + 0.25), color, 1.6, **P, **kw)
            self.shown = banner

    def keys(self) -> set:
        ev = p.getKeyboardEvents(**self.P)
        return {chr(k) for k, s in ev.items() if s & p.KEY_WAS_TRIGGERED and k < 256}

    def alive(self) -> bool:
        return p.isConnected(self.c)


BANNERS = {"OPEN": ("READY - grasp the cup", (0.1, 0.1, 0.1)), "CLOSING": ("GRASPING", (0.1, 0.3, 0.7)),
           "LIFTING": ("LIFTING", (0.1, 0.3, 0.7)), "HELD": ("HOLDING", (0.08, 0.55, 0.25)),
           "LOWERING": ("SETTING DOWN", (0.1, 0.3, 0.7)), "OPENING": ("RELEASED", (0.1, 0.1, 0.1)),
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
              wait_s: float = 5.0) -> dict:
    """Open hand, then a normal hold of the real cup -> scale so that hold sits mid hold-window in the sim.
    Valid on-device force_n: sets grip_gain. Otherwise sets the proxy's ut_per_n (grip_gain stays default)."""
    print("waiting for the glove stream...")
    got = _collect(source, wait_s)
    if len(got) < 5:
        raise _no_stream(source, len(got), wait_s)
    print(f"glove streaming ({len(got) / wait_s:.1f} frames/s)")
    prompt("Step 1/2: hold the hand OPEN and still, then press Enter (records 2 s)... ")
    opened = _collect(source, open_s)
    prompt("Step 2/2: pick up the real cup and HOLD it normally, then press Enter (records 3 s)... ")
    held = _collect(source, hold_s)
    if len(opened) < 5 or len(held) < 5:
        raise _no_stream(source, min(len(opened), len(held)), min(open_s, hold_s))
    force = all(device_force(f) is not None for f in held)
    base = None
    if not force:
        bs = [frame_b(f) for f in opened]
        base = {k: (np.mean([b[k] for b in bs if b[k] is not None], 0) if any(b[k] is not None for b in bs) else None)
                for k in PODS}
    vals = [pod_forces(f, base, 1.0)[0] for f in held]          # N (force_n) or uT (proxy, scale 1)
    per = {k: float(np.median([v[k] for v in vals if v[k] is not None] or [0.0])) for k in PODS}
    grip = float(np.median([per[k] for k in FINGERS]))
    if grip <= 0:
        raise RuntimeError(f"no grip measured during the hold: {per}")
    cal = {"mode": "force_n" if force else "proxy", "normal_hold": per, "frames": [len(opened), len(held)],
           "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if force:
        cal.update(grip_gain=SAFE_EFFORT_NM / grip, ut_per_n=None)
    else:
        cal.update(grip_gain=SAFE_EFFORT_NM / NORMAL_HOLD_N, ut_per_n=grip / NORMAL_HOLD_N)
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
        pods = {}
        for k in PODS:
            mags = [((f.get("pods", {}).get(k) or {}).get("mag") or {}) for f in frames]
            pods[k] = {"valid": f"{sum(bool(m.get('valid')) for m in mags)}/{len(mags)}",
                       "force_valid": f"{sum(bool(m.get('force_valid')) for m in mags)}/{len(mags)}",
                       "flags": sorted({fl for m in mags for fl in (m.get("flags") or [])}),
                       "last_b_ut": mags[-1].get("b_ut"), "last_force_n": mags[-1].get("force_n")}
        rep["pods"] = pods
        rep["grip_mode"] = "force_n" if device_force(last) is not None else "proxy (on-device force_n not valid yet)"
    return rep


# -- main loop ----------------------------------------------------------------------------------------
def run(source, cfg: PickupConfig | None = None, gui: bool = True, record: str | None = None,
        ut_per_n: float = 160.0, duration_s: float | None = None) -> dict:
    sess = PickupSession(cfg)
    view = PickupView(sess.sim, source.label) if gui else None
    rec = open(record, "w", buffering=1) if record else None
    dt = 1.0 / sess.cfg.control_hz
    latest, last_rx, baseline, base_acc, mode = None, None, None, [], "?"
    start = time.monotonic()
    next_tick, next_draw = start, start
    n_frames = 0
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
            calibrating = False
            if latest is not None and baseline is None:        # open-hand baseline (used whenever force_n is not valid)
                calibrating = True
                base_acc.append(frame_b(latest))
                if now - start >= BASELINE_S and len(base_acc) >= 5:
                    baseline = {k: (np.mean([b[k] for b in base_acc if b[k] is not None], 0)
                                    if any(b[k] is not None for b in base_acc) else None) for k in PODS}
            ticks = 0
            while now >= next_tick and ticks < 3:                 # fixed 50 Hz control clock
                stale = last_rx is None or now - last_rx > STALE_S
                forces = None
                if not stale and not calibrating:
                    forces, mode = pod_forces(latest, baseline, ut_per_n)
                r = sess.tick(forces)
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
                        b, col = BANNERS.get(sess.trig.state, (sess.trig.state, (0, 0, 0)))
                        banner = (f"{b}   [{'force_n' if mode == 'force_n' else 'UNCALIBRATED b-field proxy'}]", col)
                    view.sync(sess.sim, None if (calibrating or latest is None) else pod_forces(latest, baseline, ut_per_n)[0], *banner)
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
