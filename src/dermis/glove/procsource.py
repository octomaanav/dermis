"""Telemetry ingestion in a separate PROCESS, for use next to a running PyBullet simulation.

Why: PyBullet's C calls hold Python's GIL. Measured on an M2: an in-process ingestion thread fed by
a 100 Hz source accepted ~73 frames/s with an idle main thread, but only ~7 frames/s while the main
thread ran the headless physics loop and ~3 frames/s with the GUI viewer. A live glove would then
back up in the socket and trip the watchdog. Here the child process owns the source, the pipeline
and the raw recording at full rate; the parent drains the newest Snapshot once per control tick.

Interface matches what GripControl / GloveGrip use from TelemetryRunner: latest(), source_stale(),
stop_recording(), pipeline_stats(), label. Host receive times (time.monotonic_ns) are taken in the
child; the monotonic clock is system-wide on macOS/Linux, so ages computed in the parent are valid.
"""
from __future__ import annotations

import multiprocessing as mp
import time


def _child(conn, source_kw: dict, pod_order: list, record_path: str | None, stop) -> None:
    """Ingest + record at full rate; never blocks on the parent. A sender thread ships only the
    NEWEST snapshot (a one-item slot that newer frames overwrite) whenever the parent is reading."""
    import json
    import threading
    from ..telemetry.pipeline import TelemetryPipeline
    from ..telemetry.sources import open_source
    from .session import Stamped
    src = Stamped(open_source(**source_kw))
    pipe = TelemetryPipeline(pod_order=pod_order)
    fh = open(record_path, "a", buffering=1) if record_path else None
    slot = {"snap": None, "stats": None, "error": None}
    lock, ready, done = threading.Lock(), threading.Event(), threading.Event()
    n_rec = 0

    def sender():
        while not (done.is_set() and not ready.is_set()):
            if not ready.wait(0.2):
                continue
            with lock:
                items = [(k, slot[k]) for k in ("snap", "stats", "error") if slot[k] is not None]
                slot.update(snap=None, stats=None, error=None)
                ready.clear()
            for k, v in items:
                try:
                    conn.send((k, v))
                except (BrokenPipeError, EOFError, OSError):      # parent has gone
                    stop.set()
                    return

    th = threading.Thread(target=sender, daemon=True)
    th.start()

    def post(kind, payload):
        with lock:
            slot[kind] = payload
        ready.set()

    last_stats = 0.0
    try:
        for msg in src:
            if stop.is_set():
                break
            if fh and msg.obj is not None:
                fh.write(json.dumps(msg.obj) + "\n")
                n_rec += 1
            snap = pipe.ingest(msg, src.label)
            if snap is not None:
                post("snap", snap)
            now = time.monotonic()
            if now - last_stats > 0.5:
                post("stats", {**pipe.stats(), "recorded": n_rec, "seen_pods": pipe.seen_pods})
                last_stats = now
    except Exception as e:                               # surfaced to the parent
        post("error", repr(e))
    finally:
        if fh:
            fh.close()
        post("stats", {**pipe.stats(), "recorded": n_rec, "seen_pods": pipe.seen_pods, "finished": True})
        done.set()
        th.join(timeout=1.0)


class ProcessTelemetry:
    def __init__(self, source_kw: dict, pod_order, record_path: str | None = None, label: str = "",
                 timeout_s: float = 0.25):
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe(duplex=False)
        self._stop = ctx.Event()
        self.rec_path, self.label, self.timeout_s = record_path, label, timeout_s
        self._latest = None
        self.stats: dict = {}
        self.error: str | None = None
        self.proc = ctx.Process(target=_child, args=(child, source_kw, list(pod_order), record_path, self._stop),
                                daemon=True)
        self.proc.start()

    def _drain(self) -> None:
        while self._conn.poll():
            try:
                kind, payload = self._conn.recv()
            except EOFError:
                break
            if kind == "snap":
                self._latest = payload
            elif kind == "stats":
                self.stats = payload
            elif kind == "error":
                self.error = payload

    def latest(self):
        self._drain()
        return self._latest

    def source_stale(self) -> bool:
        s = self.latest()
        return s is None or (time.monotonic_ns() - s.frame.t_recv_ns) > self.timeout_s * 1e9

    def pipeline_stats(self) -> dict:
        self._drain()
        return self.stats

    def stop_recording(self) -> dict:
        self._stop.set()
        self.proc.join(timeout=2.0)
        if self.proc.is_alive():
            self.proc.terminate()
        self._drain()
        return {"path": self.rec_path, "frames": self.stats.get("recorded")}
