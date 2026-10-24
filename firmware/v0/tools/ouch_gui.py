#!/usr/bin/env python3
"""Live desktop GUI for the Dermis fingertip "OUCH!" demo.

Run the BLE receiver in one terminal, then this program in another:

    python tools/ble_receive.py --output data/ouch_live.jsonl
    python tools/ouch_gui.py --file data/ouch_live.jsonl

The first valid readings after opening the GUI establish an unloaded magnetic
baseline. This is a deformation demonstration, not a calibrated pressure or
force measurement.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import tkinter as tk
from pathlib import Path

try:  # direct script execution and package-style tests
    from .ouch import baseline, deformation, vector_from_frame
except ImportError:
    from ouch import baseline, deformation, vector_from_frame
try:  # direct script execution and package-style tests
    from .ble_receive import _is_decoded_v1, to_decoded_frame
except ImportError:
    from ble_receive import _is_decoded_v1, to_decoded_frame
from dermis.server.websocket_bridge import _frame, _open_websocket, _read_message


BACKGROUND = "#10131b"
PANEL = "#1a2030"
TEXT = "#f5f7fb"
MUTED = "#aeb8cc"
GREEN = "#35d07f"
RED = "#ff4d5f"
YELLOW = "#f6bd3b"


def _put_latest(incoming: queue.Queue, item) -> None:
    """Keep UI latency bounded if frames arrive faster than Tk can repaint."""
    while True:
        try:
            incoming.put_nowait(item)
            return
        except queue.Full:
            try:
                incoming.get_nowait()
            except queue.Empty:
                pass


async def websocket_frames(url: str, incoming: queue.Queue) -> None:
    """Receive and normalize glove WebSocket frames in a worker thread."""
    while True:
        writer = None
        try:
            _put_latest(incoming, ("status", f"Connecting to {url}…"))
            reader, writer = await _open_websocket(url, [])
            _put_latest(incoming, ("status", "Connected — waiting for valid fingertip readings"))
            while True:
                opcode, raw = await _read_message(reader)
                if opcode == 0x8:
                    raise ConnectionError("glove closed the WebSocket stream")
                if opcode == 0x9:
                    writer.write(_frame(0xA, raw, mask=True))
                    await writer.drain()
                    continue
                if opcode != 0x1:
                    continue
                try:
                    packet = json.loads(raw)
                    decoded = packet if _is_decoded_v1(packet) else to_decoded_frame(packet)
                except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    _put_latest(incoming, ("status", f"Ignoring malformed frame: {exc}"))
                    continue
                _put_latest(incoming, ("frame", decoded))
        except (ConnectionError, OSError, asyncio.IncompleteReadError, ValueError) as exc:
            _put_latest(incoming, ("status", f"Disconnected — retrying in 1 s ({exc})"))
            await asyncio.sleep(1)
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except (ConnectionError, OSError):
                    pass


class OuchWindow:
    def __init__(self, root: tk.Tk, path: Path | None, url: str | None, finger: str, baseline_frames: int, threshold: float):
        self.root, self.path, self.url, self.finger = root, path, url, finger
        self.baseline_frames = baseline_frames
        self.samples: list[tuple[float, float, float]] = []
        self.reference: tuple[float, float, float] | None = None
        self.stream = None
        self.incoming: queue.Queue = queue.Queue(maxsize=1)
        self.last_active = False
        self.valid_frames = 0
        self.threshold = tk.DoubleVar(value=threshold)
        self.status = tk.StringVar(value="WAITING FOR LIVE DATA")
        self.detail = tk.StringVar(value=(f"Watching {path.name}" if path else f"Connecting to {url}") + f" · {finger}")
        self.value = tk.StringVar(value="— µT")
        self._build()
        if self.url:
            threading.Thread(target=lambda: asyncio.run(websocket_frames(self.url, self.incoming)), daemon=True).start()
        self.root.after(50, self.poll)

    def _build(self) -> None:
        root = self.root
        root.title("Dermis — OUCH! demo")
        root.configure(bg=BACKGROUND)
        root.minsize(720, 500)
        root.geometry("900x640")

        frame = tk.Frame(root, bg=BACKGROUND, padx=42, pady=32)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text="DERMIS FINGERTIP DEMO", bg=BACKGROUND, fg=TEXT,
                 font=("Helvetica", 18, "bold")).pack(anchor="w")
        tk.Label(frame, text="Magnetic deformation demo · not calibrated physical pressure", bg=BACKGROUND,
                 fg=MUTED, font=("Helvetica", 12)).pack(anchor="w", pady=(3, 25))

        self.alert = tk.Label(frame, textvariable=self.status, bg=PANEL, fg=YELLOW,
                              font=("Helvetica", 46, "bold"), height=2)
        self.alert.pack(fill="x")
        tk.Label(frame, textvariable=self.detail, bg=BACKGROUND, fg=MUTED,
                 font=("Helvetica", 12)).pack(pady=(12, 18))

        row = tk.Frame(frame, bg=BACKGROUND)
        row.pack(fill="x")
        tk.Label(row, text="DEFORMATION", bg=BACKGROUND, fg=MUTED,
                 font=("Helvetica", 12, "bold")).pack(side="left")
        tk.Label(row, textvariable=self.value, bg=BACKGROUND, fg=TEXT,
                 font=("Helvetica", 25, "bold")).pack(side="right")

        self.gauge = tk.Canvas(frame, height=42, bg="#2b3448", highlightthickness=0)
        self.gauge.pack(fill="x", pady=(8, 30))
        self.gauge.bind("<Configure>", lambda _event: self.draw_gauge(0.0, False))

        controls = tk.Frame(frame, bg=BACKGROUND)
        controls.pack(fill="x")
        tk.Label(controls, text="OUCH THRESHOLD", bg=BACKGROUND, fg=MUTED,
                 font=("Helvetica", 12, "bold")).pack(anchor="w")
        slider = tk.Scale(controls, from_=1, to=3000, resolution=1, orient="horizontal",
                          variable=self.threshold, bg=BACKGROUND, fg=TEXT, troughcolor="#2b3448",
                          activebackground=GREEN, highlightthickness=0, font=("Helvetica", 11))
        slider.pack(fill="x", pady=(2, 16))
        tk.Button(controls, text="RESET BASELINE", command=self.reset_baseline, bg="#34415a", fg=TEXT,
                  activebackground="#4a5c7e", activeforeground=TEXT, relief="flat", padx=18, pady=12,
                  font=("Helvetica", 12, "bold")).pack()

    def reset_baseline(self) -> None:
        self.samples.clear()
        self.reference = None
        self.last_active = False
        self.status.set("HOLD FINGER UNLOADED")
        self.detail.set(f"Capturing {self.baseline_frames} valid {self.finger} readings…")
        self.value.set("— µT")
        self.set_theme(YELLOW)
        self.draw_gauge(0.0, False)

    def set_theme(self, color: str) -> None:
        self.alert.configure(fg=color)

    def draw_gauge(self, delta: float, active: bool) -> None:
        width = max(self.gauge.winfo_width(), 1)
        threshold = max(self.threshold.get(), 1.0)
        fraction = min(delta / (threshold * 1.25), 1.0)
        color = RED if active else GREEN
        self.gauge.delete("all")
        self.gauge.create_rectangle(0, 0, width * fraction, 42, fill=color, width=0)
        marker = min(width * threshold / (threshold * 1.25), width - 2)
        self.gauge.create_line(marker, 0, marker, 42, fill=YELLOW, width=3)

    def process(self, frame: dict) -> None:
        vector = vector_from_frame(frame, self.finger)
        if vector is None:
            return
        self.valid_frames += 1
        if self.reference is None:
            self.samples.append(vector)
            count = len(self.samples)
            self.status.set("HOLD FINGER UNLOADED")
            self.detail.set(f"Baseline {count}/{self.baseline_frames} · waiting for a steady {self.finger}")
            self.value.set("— µT")
            self.set_theme(YELLOW)
            if count < self.baseline_frames:
                return
            self.reference = baseline(self.samples)
            self.status.set("READY — PRESS THE FINGERTIP")
            self.detail.set(f"Baseline captured from {self.baseline_frames} readings")
        delta = deformation(vector, self.reference)
        active = delta >= self.threshold.get()
        self.status.set("OUCH!" if active else "COMFORTABLE")
        self.detail.set(f"{self.finger} · live frame {frame.get('frame', '—')} · threshold {self.threshold.get():.0f} µT")
        self.value.set(f"{delta:,.0f} µT")
        self.set_theme(RED if active else GREEN)
        self.draw_gauge(delta, active)
        if active and not self.last_active:
            self.root.bell()
        self.last_active = active

    def poll(self) -> None:
        if self.url:
            self.poll_network()
            self.root.after(50, self.poll)
            return
        try:
            if self.stream is None:
                self.stream = self.path.open(encoding="utf-8")
                # Live mode begins with samples that arrive after the UI opens.
                self.stream.seek(0, 2)
                self.reset_baseline()
            for _ in range(50):  # drain a short burst without freezing the UI
                line = self.stream.readline()
                if not line:
                    break
                try:
                    self.process(json.loads(line))
                except json.JSONDecodeError:
                    pass
        except FileNotFoundError:
            self.status.set("WAITING FOR CAPTURE FILE")
            self.detail.set(f"Start ble_receive.py --output {self.path}")
            self.set_theme(YELLOW)
        except OSError as exc:
            self.status.set("FILE ERROR")
            self.detail.set(str(exc))
            self.set_theme(RED)
        self.root.after(50, self.poll)

    def poll_network(self) -> None:
        """Drain network-worker events on Tk's main thread."""
        for _ in range(50):
            try:
                kind, value = self.incoming.get_nowait()
            except queue.Empty:
                return
            if kind == "frame":
                self.process(value)
            else:
                self.status.set(value.upper())
                self.detail.set(f"{self.finger} · Wi-Fi WebSocket source")
                self.set_theme(YELLOW)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", type=Path, help="JSONL file to follow instead of connecting directly")
    source.add_argument("--url", default="ws://192.168.4.1:81/ws", help="glove WebSocket URL")
    parser.add_argument("--finger", default="index_tip", help="pod location to monitor")
    parser.add_argument("--baseline-frames", type=int, default=25)
    parser.add_argument("--threshold-ut", type=float, default=100.0)
    args = parser.parse_args()
    if args.baseline_frames < 1 or args.threshold_ut <= 0:
        parser.error("--baseline-frames and --threshold-ut must be positive")
    root = tk.Tk()
    OuchWindow(root, args.file, None if args.file else args.url, args.finger, args.baseline_frames, args.threshold_ut)
    root.mainloop()


if __name__ == "__main__":
    main()
