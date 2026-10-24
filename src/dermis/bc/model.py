"""MLP policy and the checkpoint format.

A checkpoint is a directory:
    model.pt        state_dict + everything needed to rebuild inputs/outputs (see save_checkpoint)
    metadata.json   the same metadata, human-readable (no weights)
    train_log.csv   per-epoch losses
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..sim.env import ACTION_HIGH, ACTION_LOW, obs_indices
from .dataset import Normalizer, history_windows, unscale_actions

CHECKPOINT_VERSION = 1


class MLPPolicy(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: list[int], dropout: float = 0.0):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))          # actions live in [-1, 1], unscaled to bounds afterwards


def save_checkpoint(path: Path, model: MLPPolicy, meta: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoint_version": CHECKPOINT_VERSION,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "meta": meta}, path / "model.pt")
    (path / "metadata.json").write_text(json.dumps(meta, indent=1))


class BCPolicy:
    """Runtime wrapper. Consumes only simulator/robot observations (OBS_NAMES vectors); it has no
    glove input and runs without any telemetry source attached."""

    label = "AUTONOMOUS"

    def __init__(self, model: MLPPolicy, meta: dict):
        self.model = model.eval()
        self.meta = meta
        self.cols = obs_indices(meta["feature_groups"])
        self.history = int(meta["history"])
        self.norm = Normalizer.from_dict(meta["normalizer"])
        self._buf: deque = deque(maxlen=self.history)

    @classmethod
    def load(cls, path: str | Path) -> "BCPolicy":
        path = Path(path)
        ckpt = torch.load(path / "model.pt" if path.is_dir() else path, map_location="cpu", weights_only=False)
        meta = ckpt["meta"]
        m = meta["model"]
        model = MLPPolicy(meta["n_inputs"], meta["n_outputs"], m["hidden"], m.get("dropout", 0.0))
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, meta)

    def reset(self) -> None:
        self._buf.clear()

    def features(self, obs: np.ndarray) -> np.ndarray:
        x = self.norm(np.asarray(obs, np.float32)[self.cols])
        if not self._buf:                       # same start-of-episode padding as training
            self._buf.extend([x] * (self.history - 1))
        self._buf.append(x)
        return np.concatenate(list(self._buf)).astype(np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(self.features(obs))[None]
        a = unscale_actions(self.model(x)[0].numpy())
        return np.clip(a, ACTION_LOW, ACTION_HIGH).astype(np.float32)

    @torch.no_grad()
    def predict_batch(self, obs_seq: np.ndarray) -> np.ndarray:
        """Offline prediction for one whole episode (T, D) using the training-time windowing."""
        x = history_windows(self.norm(obs_seq[:, self.cols]), self.history)
        return unscale_actions(self.model(torch.from_numpy(x.astype(np.float32))).numpy())
