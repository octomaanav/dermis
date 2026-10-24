"""Ingestion metrics for the dashboard, computed from what the pipeline actually saw.

Every number has a definition (DEFINITIONS, shown as tooltips) and a scope:
    session     since this process started
    connection  since the current TCP connection (or source) began; reset on reconnect
    window      the last `window_s` seconds of host time (rolling)

No end-to-end latency is reported: device and host clocks are not synchronized (t_host_ns is
null), so only host-side quantities are measured: frame age, inter-arrival timing and the
host processing time of each line.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque, OrderedDict

U32 = 1 << 32
MAX_GAP_TRACKED = 10_000          # larger forward jumps are counted, but not stored id-by-id

DEFINITIONS = {
    "received_fps": "Lines received per second (any verdict) over the rolling window, host clock.",
    "accepted_fps": "Frames accepted by the pipeline per second over the rolling window (after schema, "
                    "duplicate and ordering checks).",
    "rejected": "Lines not used: malformed JSON / bad UTF-8, oversized, schema violations, adapter errors. "
                "Duplicates and late (out-of-order) frames are counted separately.",
    "missing_ids": "Unique frame IDs never received, inferred from forward gaps in the device frame counter. "
                   "A late frame that fills a gap is removed again. Counting restarts on reconnect, device "
                   "change or a frame-counter reset. Gaps can arise anywhere upstream (sensor bus, firmware "
                   "TX buffer, bridge), so this is NOT network packet loss.",
    "duplicates": "Frames whose number equals the last accepted frame (dropped).",
    "out_of_order": "Frames older than the last accepted one, within the reorder window (dropped, never applied).",
    "usable_pct": "Share of configured sensor channels usable per accepted frame, averaged over the window. "
                  "Denominator = configured pods x 2 sensors (magnetometer + IMU). Usable = valid, finite and "
                  "either fresh or STALE for at most the 50 ms hold. TEMP_STALE does not make a channel unusable.",
    "frame_age_ms": "Host time since the newest accepted frame arrived (host monotonic clock). Not latency.",
    "usable_age_ms": "Host time since the newest frame whose required control channels were usable.",
    "interval_ms": "Mean host receive interval between consecutive accepted frames over the window.",
    "jitter_ms": "Standard deviation of host receive intervals between consecutive accepted frames over the "
                 "window. TCP may deliver several frames in one read, which shows up as near-zero intervals.",
    "processing_us": "Host time to parse, validate and ingest one line (median / 95th percentile over the window). "
                     "Measured in the TCP receiver; not available for other sources.",
    "recording": "Raw telemetry recording in progress: frames written and elapsed host time.",
}


class MissingTracker:
    """Unique missing frame IDs from forward gaps; late frames that fill a gap are taken back out."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.last: int | None = None
        self.missing: set[int] = set()
        self.untracked = 0                       # ids inside gaps too large to store one by one
        self.late_filled = 0
        self.seen_late: OrderedDict = OrderedDict()   # late frames seen before their gap was registered (bounded)

    def accept(self, frame: int) -> None:
        if self.last is None:
            self.last = frame
            return
        delta = (frame - self.last) % U32
        if 0 < delta < U32 // 2:
            gap = delta - 1
            if gap > MAX_GAP_TRACKED:
                self.untracked += gap
            else:
                for k in range(1, gap + 1):
                    fid = (self.last + k) % U32
                    if self.seen_late.pop(fid, None) is not None:
                        self.late_filled += 1    # it already arrived (late): not missing
                    else:
                        self.missing.add(fid)
            self.last = frame

    def late(self, frame: int) -> None:
        """Order-independent: works whether the late frame is reported before or after its gap."""
        if frame in self.missing:
            self.missing.discard(frame)
            self.late_filled += 1
        else:
            self.seen_late[frame] = True
            while len(self.seen_late) > 512:
                self.seen_late.popitem(last=False)

    @property
    def count(self) -> int:
        return len(self.missing) + self.untracked


class IngestMetrics:
    """Thread-safe. Feed it from the ingestion thread; read summary() from any thread."""

    def __init__(self, window_s: float = 5.0, n_channels: int = 8, t_start_ns: int | None = None):
        self.window_s = window_s
        self.n_channels = n_channels
        self._lock = threading.Lock()
        self.t_start = time.monotonic_ns() if t_start_ns is None else t_start_ns
        self.lines = deque()                      # (t_ns, verdict)
        self.accepts = deque()                    # (t_ns, usable, total)
        self.proc = deque()                       # (t_ns, us)
        self.session = {"received": 0, "accepted": 0, "rejected": 0, "duplicates": 0, "out_of_order": 0}
        self.rejected_by = {}
        self.conn = dict(self.session)
        self.conn_started_ns = self.t_start
        self.missing_conn = MissingTracker()
        self.missing_session_total = 0            # missing ids from completed connections
        self.last_accept_ns: int | None = None
        self.resets: list = []

    # -- feeding ------------------------------------------------------------------

    def _trim(self, now: int) -> None:
        lo = now - int(self.window_s * 1e9)
        for d in (self.lines, self.accepts, self.proc):
            while d and d[0][0] <= lo:
                d.popleft()

    def line(self, verdict: str, frame: int | None, t_ns: int, processing_us: float | None = None) -> None:
        with self._lock:
            self.lines.append((t_ns, verdict))
            if processing_us is not None:
                self.proc.append((t_ns, processing_us))
            for scope in (self.session, self.conn):
                scope["received"] += 1
                if verdict == "accepted":
                    scope["accepted"] += 1
                elif verdict == "duplicate":
                    scope["duplicates"] += 1
                elif verdict == "out_of_order":
                    scope["out_of_order"] += 1
                else:
                    scope["rejected"] += 1
            if verdict not in ("accepted", "duplicate", "out_of_order"):
                self.rejected_by[verdict] = self.rejected_by.get(verdict, 0) + 1
            if verdict == "out_of_order" and frame is not None:
                self.missing_conn.late(frame)
            self._trim(t_ns)

    def late_frame(self, frame: int) -> None:
        """A frame arrived after a newer one (seen at the source); if its id was counted missing, it was not."""
        with self._lock:
            self.missing_conn.late(frame)

    def accepted(self, frame: int, t_ns: int, usable: int, total: int) -> None:
        with self._lock:
            self.accepts.append((t_ns, usable, total))
            self.missing_conn.accept(frame)
            self.last_accept_ns = t_ns
            self._trim(t_ns)

    def stream_reset(self, reason: str, t_ns: int | None = None, new_connection: bool = False) -> None:
        """Frame counter restarted / device changed / new connection: gaps across it are not 'missing'."""
        with self._lock:
            self.missing_session_total += self.missing_conn.count
            self.missing_conn.reset()
            if new_connection:
                self.conn = {k: 0 for k in self.session}
                self.conn_started_ns = t_ns or time.monotonic_ns()
            self.resets.append({"t_ns": t_ns or time.monotonic_ns(), "reason": reason})
            self.resets = self.resets[-50:]

    # -- reading --------------------------------------------------------------------

    def summary(self, now_ns: int | None = None) -> dict:
        now = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            self._trim(now)
            span = min(self.window_s, max(1e-9, (now - self.t_start) / 1e9))
            recv = len(self.lines)
            acc = [a for a in self.accepts]
            ts = [a[0] for a in acc]
            iv = [(b - a) / 1e6 for a, b in zip(ts, ts[1:])]
            mean_iv = sum(iv) / len(iv) if iv else None
            jitter = math.sqrt(sum((x - mean_iv) ** 2 for x in iv) / len(iv)) if len(iv) >= 2 else None
            usable = sum(a[1] for a in acc)
            total = sum(a[2] for a in acc)
            proc = sorted(p[1] for p in self.proc)
            q = lambda f: None if not proc else round(proc[min(len(proc) - 1, int(f * len(proc)))], 1)
            return {
                "window_s": self.window_s,
                "received_fps": round(recv / span, 1),
                "accepted_fps": round(len(acc) / span, 1),
                "interval_ms": None if mean_iv is None else round(mean_iv, 2),
                "jitter_ms": None if jitter is None else round(jitter, 2),
                "usable_pct": None if not total else round(100.0 * usable / total, 1),
                "usable_denominator": f"{self.n_channels} channels per frame x {len(acc)} frames in window",
                "frame_age_ms": None if self.last_accept_ns is None else round((now - self.last_accept_ns) / 1e6, 1),
                "processing_us": {"median": q(0.5), "p95": q(0.95), "n": len(proc)},
                "session": dict(self.session), "connection": dict(self.conn),
                "rejected_by": dict(self.rejected_by),
                "connection_age_s": round((now - self.conn_started_ns) / 1e9, 1),
                "missing_ids": {"connection": self.missing_conn.count,
                                "session": self.missing_session_total + self.missing_conn.count,
                                "late_filled": self.missing_conn.late_filled},
                "stream_resets": list(self.resets[-5:]),
            }


def usable_channels(snap) -> tuple[int, int]:
    """(usable, total) magnetometer + IMU channels in one pipeline Snapshot (configured pods x 2)."""
    import numpy as np
    total = 2 * len(snap.pods)
    mag = snap.mag_held_mask & np.all(np.isfinite(snap.mag_held), axis=1)
    imu = snap.imu_held_mask & np.all(np.isfinite(snap.acc_held), axis=1)
    return int(mag.sum() + imu.sum()), total
