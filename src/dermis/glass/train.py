"""Behavioral cloning for glass_hold: MLP on a history of glove-frame features (dermis.glass.features).

Episode-level train/val split; normalizer fitted on training episodes only; history windows never cross
episodes. Checkpoint = model.pt + metadata.json (feature names, normalizer, history, action bounds, task
config, provenance) + train_log.csv.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .env import EFFORT_HIGH, EFFORT_LOW, GlassConfig
from .features import NAMES, frame_features


def _windows(X: np.ndarray, H: int) -> np.ndarray:
    pad = np.repeat(X[:1], H - 1, axis=0)
    Xp = np.concatenate([pad, X], 0)
    return np.stack([Xp[i:i + H].ravel() for i in range(len(X))])


def _mlp(n_in: int, hidden=(128, 128)) -> nn.Module:
    layers, k = [], n_in
    for h in hidden:
        layers += [nn.Linear(k, h), nn.ReLU()]
        k = h
    return nn.Sequential(*layers, nn.Linear(k, 1))


def train(data_dir: str, out_dir: str, history: int = 4, epochs: int = 60, seed: int = 0, val_frac: float = 0.2,
          name: str | None = None) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    files = sorted(Path(data_dir).glob("ep_*.npz"))
    assert files, f"no episodes in {data_dir}"
    order = rng.permutation(len(files))
    n_val = max(1, int(len(files) * val_frac))
    val_f, tr_f = [files[i] for i in order[:n_val]], [files[i] for i in order[n_val:]]

    def load(fs):
        Xs, ys = [], []
        for f in fs:
            z = np.load(f)
            Xs.append(_windows(z["X"], history))
            ys.append(z["label"])
        return np.concatenate(Xs).astype(np.float32), np.concatenate(ys).astype(np.float32)[:, None]
    Xtr, ytr = load(tr_f)
    Xva, yva = load(val_f)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    net = _mlp(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    tX, ty = torch.tensor((Xtr - mu) / sd), torch.tensor(ytr)
    vX, vy = torch.tensor((Xva - mu) / sd), torch.tensor(yva)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log, best, best_state = ["epoch,train_mse,val_mae_nm"], 1e9, None
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(tX))
        tot = 0.0
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = nn.functional.mse_loss(net(tX[b]), ty[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            mae = float((net(vX) - vy).abs().mean())
        log.append(f"{ep},{tot / len(tX):.6g},{mae:.6g}")
        if mae < best:
            best, best_state = mae, {k: v.clone() for k, v in net.state_dict().items()}
    torch.save(best_state, out / "model.pt")
    (out / "train_log.csv").write_text("\n".join(log) + "\n")
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:16]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    meta = {"name": name or out.name, "task": "glass_hold", "task_config": GlassConfig().to_dict(),
            "feature_names": [f"{n}@t-{history - 1 - h}" if h < history - 1 else f"{n}@t" for h in range(history) for n in NAMES],
            "frame_features": NAMES, "history": history, "normalizer": {"mean": mu.tolist(), "std": sd.tolist()},
            "normalizer_fitted_on": "training episodes only", "action_names": ["grip_effort_nm"],
            "action_low": EFFORT_LOW, "action_high": EFFORT_HIGH, "hidden": [128, 128],
            "inputs": "SYNTHETIC glove frames (schema dermis-4mlx-1imu-force-provisional-v1) + previous effort; no simulator ground truth",
            "best_val_mae_nm": round(best, 5), "train_s": round(time.time() - t0, 1),
            "provenance": {"data_dir": str(data_dir), "data_digest": digest, "sources": ["scripted"], "demonstrator": "load_expert",
                           "train_episodes": [f.stem for f in tr_f], "val_episodes": [f.stem for f in val_f],
                           "seed": seed, "git_commit": commit, "torch": torch.__version__}}
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))
    return meta


class GlassPolicy:
    def __init__(self, net: nn.Module, meta: dict):
        self.net, self.meta = net.eval(), meta
        self.mu = np.array(meta["normalizer"]["mean"], np.float32)
        self.sd = np.array(meta["normalizer"]["std"], np.float32)
        self.H = meta["history"]

    @classmethod
    def load(cls, ckpt: str) -> "GlassPolicy":
        meta = json.loads(Path(ckpt, "metadata.json").read_text())
        assert meta["task"] == "glass_hold" and meta["frame_features"] == NAMES, "incompatible checkpoint"
        net = _mlp(len(meta["feature_names"]), tuple(meta["hidden"]))
        net.load_state_dict(torch.load(Path(ckpt, "model.pt"), weights_only=True))
        return cls(net, meta)

    def reset(self, baseline_frame: dict):
        self.baseline = baseline_frame
        self.hist = deque(maxlen=self.H)

    def act(self, frame: dict, prev_effort: float) -> float:
        x = frame_features(frame, self.baseline, prev_effort)
        if not self.hist:
            self.hist.extend([x] * self.H)
        else:
            self.hist.append(x)
        z = (np.concatenate(self.hist) - self.mu) / self.sd
        with torch.no_grad():
            return float(self.net(torch.tensor(z[None], dtype=torch.float32))[0, 0])
