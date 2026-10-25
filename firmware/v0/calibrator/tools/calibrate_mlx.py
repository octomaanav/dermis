#!/usr/bin/env python3
"""Interactively fit one MLX90393 patch against the calibrated load cell."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import serial

HEADER = ["t_us", "frame", "load_raw", "load_delta", "force_N"]
for sensor in range(1, 5):
    HEADER.extend(
        [
            f"s{sensor}x_uT",
            f"s{sensor}y_uT",
            f"s{sensor}z_uT",
            f"s{sensor}temp_C",
            f"s{sensor}valid",
        ]
    )


def read_sample(device: serial.Serial) -> list[float] | None:
    line = device.readline().decode("utf-8", errors="replace").strip()
    if not line or line.startswith("#"):
        return None
    fields = line.split(",")
    if len(fields) != len(HEADER):
        return None
    try:
        values = [float(value) for value in fields]
    except ValueError:
        return None
    if not all(math.isfinite(value) for value in values[:5]):
        raise RuntimeError("load cell is not calibrated; force_N is not finite")
    return values


def capture(device: serial.Serial, seconds: float) -> list[list[float]]:
    samples: list[list[float]] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        sample = read_sample(device)
        if sample is not None:
            samples.append(sample)
            print(f"samples={len(samples)} force={sample[4]:7.3f} N", end="\r")
    print()
    return samples


def features(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    delta = values[:, :3] - baseline[:3]
    dt = values[:, 3:4] - baseline[3]
    x, y, z = delta[:, 0], delta[:, 1], delta[:, 2]
    return np.column_stack(
        [x, y, z, x*x, y*y, z*z, x*y, x*z, y*z, dt]
    )


def fit_model(samples: list[list[float]], sensor: int, baseline: np.ndarray) -> dict:
    data = np.asarray(samples, dtype=float)
    start = 5 + (sensor - 1) * 5
    valid = data[:, start + 4] == 1
    valid &= np.isfinite(data[:, 4])
    data = data[valid]
    if len(data) < 100:
        raise RuntimeError(f"only {len(data)} valid samples; collect at least 100")

    force = data[:, 4]
    if np.ptp(force) < 0.5:
        raise RuntimeError("force range is below 0.5 N; apply a wider loading range")
    sensor_values = data[:, start : start + 4]
    raw_features = features(sensor_values, baseline)
    mean = raw_features.mean(axis=0)
    scale = raw_features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (raw_features - mean) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])

    indices = np.arange(len(data))
    validation = indices % 5 == 0
    training = ~validation
    ridge = np.eye(design.shape[1]) * 1e-3
    ridge[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design[training].T @ design[training] + ridge,
        design[training].T @ force[training],
    )
    predicted = design @ coefficients
    residual = predicted[validation] - force[validation]
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    span = float(np.ptp(force[validation]))
    r2_denominator = float(np.sum((force[validation] - force[validation].mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / r2_denominator if r2_denominator else float("nan")

    return {
        "schema_version": 1,
        "sensor": sensor,
        "feature_order": ["dx", "dy", "dz", "dx2", "dy2", "dz2", "dxdy", "dxdz", "dydz", "dtemp"],
        "baseline": {"x_uT": baseline[0], "y_uT": baseline[1], "z_uT": baseline[2], "temp_C": baseline[3]},
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "force_range_N": [float(force.min()), float(force.max())],
        "samples": int(len(data)),
        "validation": {"rmse_N": rmse, "mae_N": mae, "r2": r2, "span_N": span},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--sensor", type=int, required=True, choices=range(1, 5))
    parser.add_argument("--baseline-seconds", type=float, default=5.0)
    parser.add_argument("--capture-seconds", type=float, default=45.0)
    parser.add_argument("--output-dir", type=Path, default=Path("calibration_data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with serial.Serial(args.port, 115200, timeout=1) as device:
        time.sleep(1)
        device.reset_input_buffer()
        device.write(b"csv\n")

        input(
            f"S{args.sensor}: remove all force, keep the patch still, then press Enter "
            f"for a {args.baseline_seconds:g} s baseline..."
        )
        baseline_samples = capture(device, args.baseline_seconds)
        if not baseline_samples:
            raise RuntimeError("no baseline samples received")
        baseline_data = np.asarray(baseline_samples, dtype=float)
        start = 5 + (args.sensor - 1) * 5
        baseline = np.median(baseline_data[:, start : start + 4], axis=0)

        input(
            f"S{args.sensor}: press Enter, then perform at least five slow 0-to-maximum-to-0 "
            f"force cycles through this patch for {args.capture_seconds:g} s..."
        )
        samples = capture(device, args.capture_seconds)
        device.write(b"stop\n")

    csv_path = args.output_dir / f"mlx_s{args.sensor}_samples.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(HEADER)
        writer.writerows(baseline_samples + samples)

    model = fit_model(samples, args.sensor, baseline)
    model_path = args.output_dir / f"mlx_s{args.sensor}_model.json"
    model_path.write_text(json.dumps(model, indent=2) + "\n")
    validation = model["validation"]
    print(f"Saved samples: {csv_path}")
    print(f"Saved model:   {model_path}")
    print(
        f"S{args.sensor} validation: RMSE={validation['rmse_N']:.4f} N, "
        f"MAE={validation['mae_N']:.4f} N, R²={validation['r2']:.4f}, "
        f"range={model['force_range_N'][0]:.3f}..{model['force_range_N'][1]:.3f} N"
    )


if __name__ == "__main__":
    main()
