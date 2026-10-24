#!/usr/bin/env python3
"""Live Dermis plot: four magnitudes plus XYZ for a selected sensor."""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial

from protocol import BAUD, SENSOR_COUNT, parse_frame, saturation_sensors


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", default=BAUD, type=int)
    parser.add_argument("--window", default=8.0, type=float, help="visible history in seconds")
    parser.add_argument("--sensor", default=1, type=int, choices=range(1, 5))
    return parser.parse_args()


def main() -> int:
    args = arguments()
    selected = args.sensor - 1
    capacity = max(400, int(args.window * 600))
    times: deque[float] = deque(maxlen=capacity)
    magnitudes = [deque(maxlen=capacity) for _ in range(SENSOR_COUNT)]
    axes = [[deque(maxlen=capacity) for _ in range(3)] for _ in range(SENSOR_COUNT)]
    start_us: int | None = None
    last_flags = 0

    port = serial.Serial(args.port, args.baud, timeout=0.01)
    port.reset_input_buffer()
    port.write(b"s")
    port.flush()

    plt.style.use("seaborn-v0_8-darkgrid")
    figure, (mag_ax, axis_ax) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    magnitude_lines = [
        mag_ax.plot([], [], color=colors[i], linewidth=2.0, label=f"Patch S{i + 1}")[0]
        for i in range(SENSOR_COUNT)
    ]
    axis_lines = [
        axis_ax.plot([], [], linewidth=1.5, label=axis)[0] for axis in ("X", "Y", "Z")
    ]
    mag_ax.set_ylabel("|B| (mT)")
    axis_ax.set_ylabel("Field (mT)")
    axis_ax.set_xlabel("Time (s)")
    mag_ax.legend(ncol=4, loc="upper left")
    axis_ax.legend(ncol=3, loc="upper left")

    def on_key(event: object) -> None:
        nonlocal selected
        key = getattr(event, "key", None)
        if key in {"1", "2", "3", "4"}:
            selected = int(key) - 1

    figure.canvas.mpl_connect("key_press_event", on_key)

    def update(_animation_frame: int) -> list[object]:
        nonlocal start_us, last_flags
        for _ in range(250):
            if port.in_waiting <= 0:
                break
            line = port.readline().decode("ascii", errors="replace")
            if line.startswith("#RATE"):
                print(line.strip())
                continue
            frame = parse_frame(line)
            if frame is None:
                continue
            if start_us is None:
                start_us = frame.t_us
            times.append((frame.t_us - start_us) / 1_000_000.0)
            for sensor in range(SENSOR_COUNT):
                x, y, z, _temperature = frame.sensor(sensor)
                magnitudes[sensor].append(math.sqrt(x * x + y * y + z * z) / 1000.0)
                axes[sensor][0].append(x / 1000.0)
                axes[sensor][1].append(y / 1000.0)
                axes[sensor][2].append(z / 1000.0)
            last_flags = frame.flags

        if times:
            x_values = list(times)
            for sensor, line in enumerate(magnitude_lines):
                line.set_data(x_values, list(magnitudes[sensor]))
                line.set_linewidth(3.5 if sensor == selected else 1.8)
            for axis, line in enumerate(axis_lines):
                line.set_data(x_values, list(axes[selected][axis]))
            left = max(0.0, x_values[-1] - args.window)
            right = max(args.window, x_values[-1])
            mag_ax.set_xlim(left, right)
            mag_ax.relim()
            mag_ax.autoscale_view(scalex=False, scaley=True)
            axis_ax.relim()
            axis_ax.autoscale_view(scalex=False, scaley=True)

        saturated = saturation_sensors(last_flags)
        warning = f"  SATURATION RISK: {saturated}" if saturated else ""
        color = "crimson" if saturated else "black"
        figure.suptitle(
            f"Press each patch: only its colored |B| trace should move.  "
            f"Selected S{selected + 1} (keys 1-4){warning}",
            color=color,
            fontweight="bold",
        )
        return [*magnitude_lines, *axis_lines]

    animation = FuncAnimation(figure, update, interval=40, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        # Keep a reference until the GUI exits and tell firmware to leave stream mode.
        _ = animation
        port.write(b"\n")
        port.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

