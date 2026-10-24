#!/usr/bin/env python3
"""Analyze a Dermis CSV log and save its sample-interval jitter histogram."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from protocol import HEADER, SENSOR_COUNT, estimate_dropped


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--no-plot", action="store_true", help="do not write a PNG histogram")
    return parser.parse_args()


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    values: list[list[float]] = []
    flags: list[int] = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != HEADER:
            raise ValueError(f"unexpected CSV header: {reader.fieldnames!r}")
        for row in reader:
            timestamps.append(int(row["t_us"]))
            values.append([float(row[name]) for name in HEADER[1:-1]])
            flags.append(int(row["flags"], 0))
    if not timestamps:
        raise ValueError("CSV contains no data frames")
    return np.asarray(timestamps, dtype=np.int64), np.asarray(values), np.asarray(flags, dtype=np.uint32)


def summarize(timestamps: np.ndarray, values: np.ndarray, flags: np.ndarray) -> list[str]:
    lines: list[str] = []
    intervals = np.diff(timestamps).astype(float)
    positive = intervals[intervals > 0]
    nominal = float(np.median(positive)) if positive.size else math.nan
    dropped = estimate_dropped(timestamps.tolist())
    lines.append(
        f"frames={len(timestamps)} duration={(timestamps[-1] - timestamps[0]) / 1e6:.3f}s "
        f"median_interval={nominal:.1f}us estimated_dropped={dropped}"
    )
    if positive.size:
        lines.append(
            f"interval mean={positive.mean():.1f}us std={positive.std(ddof=1):.1f}us "
            f"min={positive.min():.1f}us max={positive.max():.1f}us"
        )

    baseline_count = max(1, min(len(values), int(round(len(values) * 0.10))))
    for sensor in range(SENSOR_COUNT):
        start = sensor * 4
        xyz = values[:, start : start + 3]
        temperature = values[:, start + 3]
        magnitude = np.linalg.norm(xyz, axis=1)
        sat_mask = np.uint32(1 << (4 + sensor))
        saturation_count = int(np.count_nonzero(flags & sat_mask))
        lines.append(f"S{sensor + 1}:")
        for axis, label in enumerate("XYZ"):
            column = xyz[:, axis]
            lines.append(
                f"  {label}: mean={column.mean():.3f}uT std={column.std(ddof=1):.3f}uT "
                f"min={column.min():.3f}uT max={column.max():.3f}uT"
            )
        baseline = float(np.median(magnitude[:baseline_count]))
        peak = float(np.percentile(magnitude, 99.5))
        delta = peak - baseline
        lines.append(
            f"  T: mean={temperature.mean():.2f}C; saturation_frames={saturation_count}"
        )
        lines.append(
            f"  press test: baseline |B|={baseline / 1000:.3f}mT, "
            f"99.5% peak={peak / 1000:.3f}mT, delta={delta / 1000:.3f}mT"
        )
    return lines


def save_histogram(path: Path, timestamps: np.ndarray) -> Path:
    intervals = np.diff(timestamps).astype(float)
    intervals = intervals[intervals > 0]
    if not intervals.size:
        raise ValueError("at least two increasing timestamps are required for a histogram")
    median = float(np.median(intervals))
    jitter = intervals - median
    output = path.with_name(f"{path.stem}.jitter.png")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(jitter, bins=min(100, max(10, int(math.sqrt(jitter.size)))), color="#377eb8")
    axis.axvline(0, color="black", linewidth=1)
    axis.set(title=f"Sample-interval jitter (median {median:.1f} us)", xlabel="Jitter (us)", ylabel="Count")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def main() -> int:
    args = arguments()
    try:
        timestamps, values, flags = load_csv(args.csv_path)
        print("\n".join(summarize(timestamps, values, flags)))
        if not args.no_plot:
            print(f"jitter_histogram={save_histogram(args.csv_path, timestamps)}")
    except (OSError, ValueError, KeyError) as error:
        print(f"analyze: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

