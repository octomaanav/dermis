"""Shared parsing helpers for Dermis serial CSV frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

SENSOR_COUNT = 4
BAUD = 921_600

HEADER = ["t_us"]
for _sensor in range(1, SENSOR_COUNT + 1):
    HEADER.extend(
        [
            f"s{_sensor}x_uT",
            f"s{_sensor}y_uT",
            f"s{_sensor}z_uT",
            f"s{_sensor}t_C",
        ]
    )
HEADER.append("flags")


@dataclass(frozen=True)
class Frame:
    t_us: int
    values: tuple[float, ...]
    flags: int

    def sensor(self, index: int) -> tuple[float, float, float, float]:
        """Return x, y, z, temperature for zero-based sensor *index*."""
        start = index * 4
        return self.values[start : start + 4]  # type: ignore[return-value]

    def csv_row(self) -> list[str]:
        return [str(self.t_us), *(f"{value:.6g}" for value in self.values), hex(self.flags)]


def parse_frame(line: str) -> Frame | None:
    """Parse a firmware frame, returning None for menus and # metadata."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = stripped.split(",")
    if len(parts) != len(HEADER):
        return None
    try:
        timestamp = int(parts[0])
        values = tuple(float(value) for value in parts[1:-1])
        flags = int(parts[-1], 0)
    except ValueError:
        return None
    return Frame(timestamp, values, flags)


def saturation_sensors(flags: int) -> list[int]:
    """Return one-based sensor numbers with firmware saturation flags."""
    return [sensor + 1 for sensor in range(SENSOR_COUNT) if flags & (1 << (4 + sensor))]


def estimate_dropped(timestamps: Iterable[int]) -> int:
    """Estimate missing frames from timestamp gaps using the median interval."""
    times = list(timestamps)
    if len(times) < 3:
        return 0
    intervals = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not intervals:
        return 0
    ordered = sorted(intervals)
    nominal = ordered[len(ordered) // 2]
    if nominal <= 0:
        return 0
    return sum(max(0, round(interval / nominal) - 1) for interval in intervals if interval > nominal * 1.5)

