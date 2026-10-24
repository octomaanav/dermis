"""Bounded live buffers between the ingestion / control threads and the dashboard.

    ingestion thread  on_snapshot(), on_line()       per frame / per line (cheap appends)
    control thread    on_command()                    per control tick
    any thread        event()                         faults, watchdog, operator actions
    HTTP threads      payload(pod, cursor)            what one browser needs for the next update

Everything is a fixed-size deque, so memory is bounded however long the dashboard runs and
however many frames arrive. Recording to disk is done at full rate elsewhere; these buffers
only feed the display, which is decimated to max_samples per update.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

import numpy as np

from .metrics import IngestMetrics, usable_channels

STATE = ("fresh", "stale", "invalid", "missing")


def _num(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def _vec(v, nd=3):
    return [_num(x, nd) for x in v]


def channel_state(present: bool, usable_now: bool, fresh: bool, stale: bool) -> str:
    """fresh: new value this frame; stale: valid but repeated (STALE flag); invalid: unusable value; missing: no pod."""
    if not present:
        return "missing"
    if fresh:
        return "fresh"
    if stale:
        return "stale"
    return "invalid"


class LiveHub:
    def __init__(self, pods: tuple, frames: int = 8000, commands: int = 3000, events: int = 400,
                 window_s: float = 5.0):
        self.pods = tuple(pods)
        self.t0 = time.monotonic_ns()
        self._lock = threading.Lock()
        self.frames: deque = deque(maxlen=frames)        # (seq, t_ns, frame_no, {pod: record})
        self.commands: deque = deque(maxlen=commands)    # (seq, t_ns, dict)
        self.events: deque = deque(maxlen=events)        # (seq, wall_s, t_ns, level, kind, text)
        self._seq = 0
        self.metrics = IngestMetrics(window_s=window_s, n_channels=2 * len(self.pods))
        self.latest = None                               # newest accepted Snapshot
        self.latest_t_ns: int | None = None

    @staticmethod
    def _after(dq: deque, cursor: int, limit: int | None = None) -> list:
        """Items with seq > cursor, newest last. Walks back from the newest item and stops at the cursor,
        so a steady-state update touches only the new items, not the whole buffer."""
        out = []
        for item in reversed(dq):
            if item[0] <= cursor or (limit is not None and len(out) >= limit):
                break
            out.append(item)
        out.reverse()
        return out

    def _next(self) -> int:
        self._seq += 1
        return self._seq

    def rel(self, t_ns: int) -> float:
        return round((t_ns - self.t0) / 1e9, 4)

    # -- feeding ---------------------------------------------------------------------

    def on_snapshot(self, snap) -> None:
        recs = {}
        for i, pod in enumerate(snap.pods):
            present = bool(snap.present[i])
            ms = channel_state(present, bool(snap.mag_valid[i]), bool(snap.mag_fresh[i]), bool(snap.mag_stale[i]))
            ims = channel_state(present, bool(snap.imu_valid[i]), bool(snap.imu_fresh[i]), bool(snap.imu_stale[i]))
            recs[pod] = (snap.mag_raw[i].copy(), snap.acc_raw[i].copy(), snap.gyr_raw[i].copy(), ms, ims)
        usable, total = usable_channels(snap)
        t = snap.frame.t_recv_ns
        with self._lock:
            self.frames.append((self._next(), t, snap.frame.frame, recs))
            self.latest, self.latest_t_ns = snap, t
        self.metrics.accepted(snap.frame.frame, t, usable, total)

    def on_line(self, verdict: str, frame, t_ns: int, processing_us=None, reason=None) -> None:
        self.metrics.line(verdict, frame, t_ns, processing_us)
        if verdict not in ("accepted", "duplicate", "out_of_order"):
            self.event("warn", f"rejected:{verdict}", (reason or "")[:200])

    def on_command(self, cmd: dict, t_ns: int | None = None) -> None:
        with self._lock:
            self.commands.append((self._next(), t_ns or time.monotonic_ns(), cmd))

    def event(self, level: str, kind: str, text: str = "") -> None:
        with self._lock:
            self.events.append((self._next(), time.time(), time.monotonic_ns(), level, kind, text))

    # -- reading ----------------------------------------------------------------------

    def pod_summary(self, baseline) -> list[dict]:
        """One entry per configured pod from the newest accepted frame (empty before any frame)."""
        s = self.latest
        if s is None:
            return [{"pod": p, "health": "missing", "reason": "no frame received yet"} for p in self.pods]
        out = []
        for i, pod in enumerate(s.pods):
            present = bool(s.present[i])
            fp = s.frame.pods.get(pod)
            has_imu = fp is not None and fp.imu is not None
            ms = channel_state(present, bool(s.mag_valid[i]), bool(s.mag_fresh[i]), bool(s.mag_stale[i]))
            ims = (channel_state(present, bool(s.imu_valid[i]), bool(s.imu_fresh[i]), bool(s.imu_stale[i]))
                   if has_imu else "absent")
            mag_ok = bool(s.mag_held_mask[i] and np.all(np.isfinite(s.mag_held[i])))
            imu_ok = bool(s.imu_held_mask[i] and np.all(np.isfinite(s.acc_held[i])))
            if not present:
                health = "missing"
            elif ms == "fresh" and ims in ("fresh", "absent"):
                health = "healthy"
            elif not mag_ok and not imu_ok:
                health = "invalid"
            else:
                health = "degraded"
            base = baseline(pod)
            b = s.mag_held[i] if mag_ok else None
            out.append({
                "pod": pod, "pod_id": getattr(fp, "pod_id", None), "health": health, "mag": ms, "imu": ims,
                "mag_flags": sorted(fp.mag.flags) if fp is not None and fp.mag is not None else [],
                "imu_flags": sorted(fp.imu.flags) if has_imu else [],
                "b_abs_ut": None if b is None else _num(np.linalg.norm(b), 1),
                "db_ut": None if b is None or base is None else _num(np.linalg.norm(b - base), 1),
                "db_vec_ut": None if b is None or base is None else _vec(b - base, 1),
                "mag_age_ms": _num(s.mag_age_us[i] / 1000.0, 1),
                "imu_age_ms": _num(s.imu_age_us[i] / 1000.0, 1) if has_imu else None,
                "saturated": bool(s.mag_saturated[i]),
            })
        return out

    def samples(self, pod: str, cursor: int, baseline, max_samples: int = 60) -> tuple[list, int]:
        """New plot samples for one pod since `cursor`, decimated to at most max_samples."""
        with self._lock:
            rows = self._after(self.frames, cursor, limit=4000)       # first payload: at most the newest 4000
        if not rows:
            return [], cursor
        last = rows[-1][0]
        if len(rows) > max_samples:
            idx = np.linspace(0, len(rows) - 1, max_samples).round().astype(int)
            rows = [rows[k] for k in idx]
        base = baseline(pod)
        out = []
        for seq, t, fno, recs in rows:
            r = recs.get(pod)
            if r is None:
                continue
            b, a, g, ms, ims = r
            out.append({"t": self.rel(t), "f": fno, "b": _vec(b, 2), "a": _vec(a, 4), "g": _vec(g, 4),
                        "db": None if base is None else _vec(b - base, 2), "ms": ms, "is": ims})
        return out, last

    def commands_since(self, cursor: int, max_n: int = 60) -> tuple[list, int]:
        with self._lock:
            rows = self._after(self.commands, cursor, limit=max_n)
        if not rows:
            return [], cursor
        last = rows[-1][0]
        rows = rows[-max_n:]
        return [{"t": self.rel(t), **c} for _, t, c in rows], last

    def events_since(self, cursor: int) -> tuple[list, int]:
        with self._lock:
            rows = self._after(self.events, cursor)
        if not rows:
            return [], cursor
        return [{"wall": w, "t": self.rel(t), "level": lv, "kind": k, "text": tx} for _, w, t, lv, k, tx in rows], rows[-1][0]
