"""TCP receiver for decoded-frame NDJSON (one decoded_frame.schema.json object per line).

This is NOT the binary Dermis Wire protocol. A glove, ESP32 or USB bridge must send the decoded
JSON form, one compact object per line, UTF-8, terminated by "\\n". See docs/tcp_teleop.md.

Threads
    The asyncio event loop runs in its own daemon thread. It parses, validates and runs every
    line through the TelemetryPipeline (pure numpy, no PyBullet), then publishes the newest
    accepted Snapshot into a LatestSlot. The simulation thread only ever reads that slot: it
    never waits for the network, and a backlog cannot queue up stale robot commands because the
    slot holds one item and newer frames overwrite older ones.

Per line:  bytes -> NDJSONFramer -> UTF-8/JSON -> JSON Schema -> adapter -> pipeline -> LatestSlot

Clocks
    t_recv_ns is host time.monotonic_ns() at the moment the line was parsed. It is the only
    clock used for stream age and watchdogs. t_device_us is the device's own clock and is never
    compared with host time. t_host_ns is passed through exactly as received (null unless the
    sender has a real clock-sync fit); the server never invents it.

One controller
    The first connected client is the controller. Further clients get one error line
    ({"type":"error","error":"controller_busy"}) and are closed. When the controller
    disconnects, connection_epoch increments and the next client becomes the controller.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from pathlib import Path

from .compat import normalize_frame
from .ndjson import Line, MessageError, NDJSONFramer, Oversized, parse_line
from ..telemetry.pipeline import RawMessage, Snapshot, TelemetryPipeline
from .validation import frame_validator, schema_errors

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_MAX_MESSAGE_BYTES = 16384


class LatestSlot:
    """Single-item, overwrite-on-put handoff between the network thread and the sim thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._item = None
        self._seq = 0
        self._taken_seq = 0
        self.overwritten = 0            # items replaced before anyone read them (burst traffic)

    def put(self, item) -> None:
        with self._lock:
            if self._item is not None and self._taken_seq != self._seq:
                self.overwritten += 1
            self._item = item
            self._seq += 1

    def get(self) -> tuple[int, object]:
        """(sequence number, newest item). Never blocks on the network; seq changes when new."""
        with self._lock:
            self._taken_seq = self._seq
            return self._seq, self._item


class TelemetryServer:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES, pipeline: TelemetryPipeline | None = None,
                 source_label: str | None = None, record_path: str | None = None, rate_window_s: float = 1.0,
                 compat: bool = True, assume_valid: bool = False, log=print):
        self.host, self.port = host, port
        self.max_message_bytes = max_message_bytes
        self.pipeline = pipeline or TelemetryPipeline()
        self.source_label = source_label
        self.record_path = record_path
        self.rate_window_s = rate_window_s
        self.log = log
        self.compat, self.assume_valid = compat, assume_valid     # see compat.py
        self.compat_notes: list[str] = []
        self.slot = LatestSlot()
        self.observers: list = []           # callables(snapshot), run on EVERY accepted frame in this thread
        self._lock = threading.Lock()
        self._recv_times: deque = deque(maxlen=4096)
        self.counters = {k: 0 for k in ("bytes", "lines", "accepted", "malformed", "oversized", "schema_rejected",
                                        "adapter_rejected", "duplicates", "out_of_order", "non_finite_messages",
                                        "truncated_at_disconnect", "connections", "rejected_clients",
                                        "disconnects", "normalized")}
        self.connection_state = "stopped"
        self.connection_epoch = 0
        self.peer: str | None = None
        self.last_error: str | None = None
        self.last_accept_ns: int | None = None
        self.detected_source: str | None = None
        self.peak_buffered = 0
        self._active = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self._rec = self._rec_meta = None

    # -- lifecycle (called from the main thread) -----------------------------

    def start(self, timeout: float = 5.0) -> tuple[str, int]:
        """Bind and serve in a background thread. Returns the bound (host, port); port=0 picks a free one."""
        frame_validator()                     # import + compile the schema now, not on the first frame
        if self.record_path:
            Path(self.record_path).parent.mkdir(parents=True, exist_ok=True)
            self._rec = open(self.record_path, "a", encoding="utf-8")
            self._rec_meta = open(str(self.record_path) + ".recv.jsonl", "a", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="dermis-tcp", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("TCP server did not start")
        if self._start_error:
            raise self._start_error
        return self.host, self.port

    def stop(self) -> None:
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._shutdown)
        if self._thread:
            self._thread.join(timeout=3.0)
        for fh in (self._rec, self._rec_meta):
            if fh:
                fh.close()
        self._rec = self._rec_meta = None

    def _shutdown(self) -> None:
        if self._server:
            self._server.close()
        if self._active:
            self._active.close()
        for t in asyncio.all_tasks(self._loop):
            t.cancel()
        self._loop.call_later(0.05, self._loop.stop)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._server = self._loop.run_until_complete(
                asyncio.start_server(self._handle, self.host, self.port, limit=self.max_message_bytes))
        except OSError as e:
            self._start_error = OSError(e.errno, f"cannot listen on {self.host}:{self.port}: {e.strerror}. "
                                                 f"Is another `dermis teleop` running? Try --port or "
                                                 f"`lsof -nP -iTCP:{self.port} -sTCP:LISTEN`.")
            self._ready.set()
            return
        self.port = self._server.sockets[0].getsockname()[1]
        with self._lock:
            self.connection_state = "listening"
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()
            with self._lock:
                self.connection_state = "stopped"

    # -- connections (event-loop thread) -----------------------------------

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        peer_s = f"{peer[0]}:{peer[1]}" if peer else "?"
        if self._active is not None:
            with self._lock:
                self.counters["rejected_clients"] += 1
            self.log(f"[tcp] rejected {peer_s}: a controller is already connected ({self.peer})")
            try:
                writer.write(b'{"type":"error","error":"controller_busy","detail":"one controlling client at a time"}\n')
                await writer.drain()
            except (ConnectionError, OSError):
                pass
            writer.close()
            return
        self._active = writer
        self.pipeline.reset_stream("new_connection")        # a new connection may restart frame numbering
        with self._lock:
            self.counters["connections"] += 1
            self.connection_epoch += 1
            self.connection_state = "connected"
            self.peer = peer_s
        self.log(f"[tcp] controller connected from {peer_s} (epoch {self.connection_epoch})")
        framer = NDJSONFramer(self.max_message_bytes)
        try:
            writer.write((json.dumps({"type": "hello", "role": "controller", "format": "decoded_frame NDJSON",
                                      "max_message_bytes": self.max_message_bytes}) + "\n").encode())
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                with self._lock:
                    self.counters["bytes"] += len(data)
                for item in framer.feed(data):
                    self._on_item(item)
                self.peak_buffered = max(self.peak_buffered, framer.peak_buffered)
        except (ConnectionError, OSError) as e:
            self.last_error = f"connection error: {e}"
        except asyncio.CancelledError:
            pass
        finally:
            if framer.pending_partial():
                with self._lock:
                    self.counters["truncated_at_disconnect"] += 1
            self._active = None
            with self._lock:
                self.counters["disconnects"] += 1
                self.connection_state = "disconnected"
            self.log(f"[tcp] controller {peer_s} disconnected")
            writer.close()

    def _on_item(self, item) -> None:
        t_recv_ns = time.monotonic_ns()
        if isinstance(item, Oversized):
            self._reject("oversized", f"message of >= {item.size} bytes exceeds {self.max_message_bytes}")
            return
        assert isinstance(item, Line)
        if not item.data.strip():
            return                                          # blank keep-alive lines are allowed
        with self._lock:
            self.counters["lines"] += 1
        try:
            obj, non_finite = parse_line(item.data)
        except MessageError as e:
            self._reject("malformed", str(e))
            return
        if non_finite:
            with self._lock:
                self.counters["non_finite_messages"] += 1
        if self.compat:
            obj, notes = normalize_frame(obj, self.assume_valid)
            if notes:
                with self._lock:
                    self.counters["normalized"] += 1
                if self.counters["normalized"] == 1:
                    self.compat_notes = notes
                    self.log("[tcp] frame is not v1-shaped; normalizing (compat.py): " + "; ".join(notes[:6]))
        errs = schema_errors(obj)
        if errs:
            self._reject("schema_rejected", "; ".join(errs))
            return
        label = self._label(obj)
        p = self.pipeline
        before = dict(p.counters)
        snap = p.ingest(RawMessage(obj, t_recv_ns), label)
        self._record(item.data, t_recv_ns, snap is not None)
        if snap is None:
            if p.counters["duplicates"] > before["duplicates"]:
                self._reject("duplicates", f"duplicate frame {obj.get('frame')}", quiet=True)
            elif p.counters["out_of_order"] > before["out_of_order"]:
                self._reject("out_of_order", f"late frame {obj.get('frame')}", quiet=True)
            else:
                self._reject("adapter_rejected", str(p.events[-1].get("error")) if p.events else "adapter error")
            return
        with self._lock:
            self.counters["accepted"] += 1
            self.last_accept_ns = t_recv_ns
            self._recv_times.append(t_recv_ns)
        self.slot.put(snap)
        for cb in self.observers:
            try:
                cb(snap)
            except Exception as e:                       # an observer bug must not stop ingestion
                self._reject("adapter_rejected", f"observer error: {e!r}", quiet=True)

    def _label(self, obj: dict) -> str:
        if self.source_label:
            return self.source_label
        dev = str(obj.get("device_id", ""))
        label = "SYNTHETIC" if dev.startswith("synthetic") else "TCP-UNVERIFIED"
        self.detected_source = label
        return label

    def _reject(self, counter: str, why: str, quiet: bool = False) -> None:
        with self._lock:
            self.counters[counter] += 1
            self.last_error = f"{counter}: {why[:300]}"
        if not quiet and self.counters[counter] <= 5:
            self.log(f"[tcp] {counter}: {why[:200]}")

    def _record(self, data: bytes, t_recv_ns: int, accepted: bool) -> None:
        """Raw schema-valid lines exactly as received (replayable), plus a host-receive sidecar."""
        if self._rec is None:
            return
        self._rec.write(data.decode("utf-8") + "\n")
        self._rec_meta.write(json.dumps({"t_recv_ns": t_recv_ns, "accepted": accepted,
                                         "epoch": self.connection_epoch}) + "\n")

    # -- status (any thread) -------------------------------------------------

    def receive_rate_hz(self, now_ns: int | None = None) -> float:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        lo = now_ns - int(self.rate_window_s * 1e9)
        with self._lock:
            n = sum(1 for t in self._recv_times if t >= lo)
        return n / self.rate_window_s

    def status(self) -> dict:
        now = time.monotonic_ns()
        with self._lock:
            st = {"state": self.connection_state, "epoch": self.connection_epoch, "peer": self.peer,
                  "host": self.host, "port": self.port, **self.counters,
                  "last_accept_age_ms": None if self.last_accept_ns is None
                  else round((now - self.last_accept_ns) / 1e6, 1),
                  "last_error": self.last_error, "slot_overwritten": self.slot.overwritten,
                  "peak_buffered_bytes": self.peak_buffered, "source": self.source_label or self.detected_source}
        st["rate_hz"] = round(self.receive_rate_hz(now), 1)
        st["pipeline"] = {k: self.pipeline.counters[k] for k in ("lost_frames", "restarts", "clock_resets",
                                                               "device_changes", "config_changes", "stream_resets")}
        return st
