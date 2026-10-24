"""Episode-level splits, history windows and normalization for behavioral cloning.

Rules enforced here (and tested in tests/test_bc_dataset.py):
  * train/val split by whole episode; rows of one episode never land in both.
  * a history window only contains ticks of its own episode. Ticks before the start of the
    episode are filled by repeating the episode's first observation (the same padding the
    runtime policy uses), never with rows from another episode.
  * normalization statistics come from training episodes only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.episodes import Episode
from ..sim.env import ACTION_HIGH, ACTION_LOW, OBS_NAMES, obs_indices


def split_episodes(ids: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    ids = sorted(ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    n_val = max(1, int(round(len(ids) * val_fraction))) if len(ids) > 1 else 0
    val = sorted(ids[i] for i in order[:n_val])
    train = sorted(ids[i] for i in order[n_val:])
    return train, val


def history_windows(obs: np.ndarray, history: int) -> np.ndarray:
    """(T, D) -> (T, history * D); row t = [obs[t-h+1], ..., obs[t]], padded with obs[0]."""
    T = len(obs)
    idx = np.arange(T)[:, None] - np.arange(history - 1, -1, -1)[None, :]
    idx = np.clip(idx, 0, None)
    return obs[idx].reshape(T, -1)


def feature_names(groups: list[str], history: int) -> list[str]:
    base = [OBS_NAMES[i] for i in obs_indices(groups)]
    return [f"{n}@t-{k}" if k else f"{n}@t" for k in range(history - 1, -1, -1) for n in base]


def scale_actions(a: np.ndarray) -> np.ndarray:
    return 2.0 * (a - ACTION_LOW) / (ACTION_HIGH - ACTION_LOW) - 1.0


def unscale_actions(a: np.ndarray) -> np.ndarray:
    return ACTION_LOW + (np.asarray(a) + 1.0) * 0.5 * (ACTION_HIGH - ACTION_LOW)


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, rows: np.ndarray, std_floor: float = 1e-3) -> "Normalizer":
        return cls(rows.mean(0).astype(np.float32), np.maximum(rows.std(0), std_floor).astype(np.float32))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(np.array(d["mean"], np.float32), np.array(d["std"], np.float32))


def label_array(ep: Episode, label: str) -> np.ndarray:
    if label == "expert_actions" and "expert_actions" in ep.arrays:
        return ep.arrays["expert_actions"]
    if label not in ("actions", "expert_actions"):
        raise ValueError(f"unknown label {label!r}")
    return ep.arrays["actions"]          # keyboard/glove episodes: the issued action is the only label


def build_arrays(episodes: list[Episode], groups: list[str], history: int, norm: Normalizer, label: str):
    """Stacked (X, Y, episode_index) for a set of episodes; X normalized, Y scaled to [-1, 1]."""
    cols = obs_indices(groups)
    xs, ys, owner = [], [], []
    for k, ep in enumerate(episodes):
        o = norm(ep.obs[:, cols])
        xs.append(history_windows(o, history))
        ys.append(scale_actions(label_array(ep, label)))
        owner.append(np.full(len(ep), k))
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32),
            np.concatenate(owner))


def file_digest(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]
