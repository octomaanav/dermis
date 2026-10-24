"""Load-proxy schedule: piecewise ramps and pauses toward a peak downward load, plus bounded
short disturbances. Represents "contents being poured in" as an external force; not a fluid.

Time t is measured from the moment pouring starts (Pour button, or pre_pour_s in evaluation).
Load is never negative and never exceeds peak + disturbance_amp.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LoadSchedule:
    segments: list          # [(kind, t_start, t_end, load_start, load_end)], kind in {"ramp", "pause"}
    disturbances: list      # [(t_start, t_end, amplitude)]

    @classmethod
    def sample(cls, rng: np.random.Generator, cfg) -> "LoadSchedule":
        peak = float(rng.uniform(*cfg.max_load_range))
        n = int(rng.integers(cfg.n_ramps_range[0], cfg.n_ramps_range[1] + 1))
        cuts = np.sort(rng.uniform(0, 1, n - 1))
        levels = np.concatenate([[0.0], cuts, [1.0]]) * peak
        segs, t = [], 0.0
        for k in range(n):
            rate = float(rng.uniform(*cfg.ramp_rate_range))
            dur = (levels[k + 1] - levels[k]) / rate
            segs.append(("ramp", t, t + dur, float(levels[k]), float(levels[k + 1])))
            t += dur
            if k < n - 1:
                pause = float(rng.uniform(*cfg.pause_range))
                segs.append(("pause", t, t + pause, float(levels[k + 1]), float(levels[k + 1])))
                t += pause
        dist = []
        for _ in range(int(rng.integers(cfg.disturbance_count_range[0], cfg.disturbance_count_range[1] + 1))):
            t0 = float(rng.uniform(0, t))
            length = float(rng.uniform(*cfg.disturbance_len_range))
            amp = float(rng.uniform(-1, 1) * cfg.disturbance_amp)
            dist.append((t0, t0 + length, amp))
        return cls(segs, sorted(dist))

    @property
    def duration(self) -> float:
        return self.segments[-1][2] if self.segments else 0.0

    @property
    def peak(self) -> float:
        return self.segments[-1][4] if self.segments else 0.0

    def base(self, t: float) -> float:
        if t <= 0 or not self.segments:
            return 0.0
        for kind, t0, t1, l0, l1 in self.segments:
            if t < t1:
                return l0 + (l1 - l0) * (t - t0) / max(1e-9, t1 - t0)
        return self.peak

    def load(self, t: float) -> float:
        extra = sum(a for t0, t1, a in self.disturbances if t0 <= t < t1)
        return max(0.0, self.base(t) + extra)

    def segment(self, t: float) -> str:
        for kind, t0, t1, *_ in self.segments:
            if t < t1:
                return kind
        return "final_hold"

    def to_dict(self) -> dict:
        return {"segments": [list(s) for s in self.segments], "disturbances": [list(d) for d in self.disturbances]}

    @classmethod
    def from_dict(cls, d: dict) -> "LoadSchedule":
        return cls([tuple(s) for s in d["segments"]], [tuple(x) for x in d["disturbances"]])
