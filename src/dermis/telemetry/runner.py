"""Background reader: any source -> TelemetryPipeline, with latest snapshot, a ring buffer for
plots, and optional raw recording (the decoded-frame dicts exactly as received, JSONL)."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

from .pipeline import Snapshot, TelemetryPipeline


class TelemetryRunner:
    def __init__(self, source, pipeline: TelemetryPipeline | None = None, history: int = 600):
        self.source = source
        self.pipeline = pipeline or TelemetryPipeline()
        self.label = getattr(source, "label", "UNKNOWN")
        self.history: deque = deque(maxlen=history)
        self._lock = threading.Lock()
        self._latest: Snapshot | None = None
        self._rec_fh = None
        self.rec_path: str | None = None
        self.rec_frames = 0
        self.error: BaseException | None = None
        self.finished = False
        self._seq = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "TelemetryRunner":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            for msg in self.source:
                snap = self.pipeline.ingest(msg, self.label)
                with self._lock:
                    if self._rec_fh and msg.obj is not None:
                        self._rec_fh.write(json.dumps(msg.obj) + "\n")
                        self.rec_frames += 1
                    if snap is not None:
                        self._latest = snap
                        self._seq += 1
                        self.history.append((self._seq, snap))
        except BaseException as e:            # surfaced to the caller via .error
            self.error = e
        finally:
            self.finished = True

    def latest(self) -> Snapshot | None:
        with self._lock:
            return self._latest

    def since(self, seq: int) -> list:
        with self._lock:
            return [(s, snap) for s, snap in self.history if s > seq]

    def source_stale(self) -> bool:
        return self.pipeline.source_stale()

    def start_recording(self, path: str) -> str:
        with self._lock:
            if self._rec_fh:
                raise RuntimeError(f"already recording to {self.rec_path}")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._rec_fh = open(path, "a", buffering=1)      # line-buffered: survives an abrupt exit
            self.rec_path, self.rec_frames = path, 0
        return path

    def stop_recording(self) -> dict:
        with self._lock:
            if self._rec_fh:
                self._rec_fh.close()
            out = {"path": self.rec_path, "frames": self.rec_frames}
            self._rec_fh = None
            self.rec_path = None
        return out

    def wait_for_frame(self, timeout: float = 2.0) -> Snapshot | None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            s = self.latest()
            if s is not None:
                return s
            if self.finished:
                break
            time.sleep(0.01)
        return self.latest()
