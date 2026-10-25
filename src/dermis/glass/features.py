"""Network inputs, computed ONLY from glove frames in the dermis.glass.schema layout (+ the previous effort
command). Per pod (POD_ORDER): b_ut minus the unloaded baseline frame (3), provisional force_n (1), and the
FORCE_CLAMPED_HIGH flag (1). The IMU is not used: the hand is fixed in this demo, so it reads gravity only.
"""
from __future__ import annotations

import numpy as np

from .schema import POD_ORDER, parse

PER_POD = ("db_x", "db_y", "db_z", "force_n", "clamped_high")
NAMES = [f"{pod}.{f}" for pod in POD_ORDER for f in PER_POD] + ["prev_effort"]


def frame_features(frame: dict, baseline: dict, prev_effort: float) -> np.ndarray:
    f, b = parse(frame), parse(baseline)
    db = f.b_ut - b.b_ut
    force = np.nan_to_num(f.force_n, nan=0.0)
    per = np.concatenate([db, force[:, None], f.clamped_high[:, None].astype(float)], axis=1)
    return np.concatenate([per.ravel(), [prev_effort]]).astype(np.float32)
