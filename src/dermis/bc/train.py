"""Train a behavioral-cloning MLP from a directory of episodes.

Config keys (see configs/bc_contact.yaml):
    name, data, out, feature_groups, history, label, success_only, sources, max_episodes, val_fraction,
    split_seed, seed, device, model: {hidden, dropout}, train: {epochs, batch_size, lr, weight_decay}
"""
from __future__ import annotations

import csv
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..data.episodes import EpisodeStore
from ..sim.env import ACTION_HIGH, ACTION_LOW, ACTION_NAMES, OBS_NAMES, obs_indices
from .dataset import Normalizer, build_arrays, feature_names, file_digest, split_episodes
from .model import MLPPolicy, save_checkpoint

DEFAULTS = {
    "feature_groups": ["proprio", "object", "target", "contact"],
    "history": 2,
    "label": "expert_actions",
    "success_only": True,
    "sources": None,
    "max_episodes": None,          # use only the first N usable episodes (sorted by id) - data-scaling studies
    "val_fraction": 0.2,
    "split_seed": 0,
    "seed": 0,
    "device": "cpu",
    "model": {"hidden": [256, 256], "dropout": 0.0},
    "train": {"epochs": 40, "batch_size": 256, "lr": 1e-3, "weight_decay": 1e-5},
}


def merged_config(cfg: dict) -> dict:
    out = {**DEFAULTS, **cfg}
    out["model"] = {**DEFAULTS["model"], **cfg.get("model", {})}
    out["train"] = {**DEFAULTS["train"], **cfg.get("train", {})}
    return out


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def train(cfg: dict, log=print) -> Path:
    cfg = merged_config(cfg)
    _seed_everything(cfg["seed"])
    store = EpisodeStore(cfg["data"])
    metas = {i: store.load_meta(i) for i in store.ids()}
    usable = [i for i, m in metas.items()
              if (not cfg["success_only"] or m.get("success"))
              and (cfg["sources"] is None or m["source"] in cfg["sources"])]
    usable = sorted(usable)[:cfg["max_episodes"]] if cfg["max_episodes"] else usable
    if len(usable) < 2:
        raise SystemExit(f"need >= 2 usable episodes in {cfg['data']}, found {len(usable)} (of {len(metas)})")
    train_ids, val_ids = split_episodes(usable, cfg["val_fraction"], cfg["split_seed"])
    train_eps = [store.load(i) for i in train_ids]
    val_eps = [store.load(i) for i in val_ids]

    groups, H = cfg["feature_groups"], int(cfg["history"])
    cols = obs_indices(groups)
    norm = Normalizer.fit(np.concatenate([ep.obs[:, cols] for ep in train_eps]))   # training rows only
    Xtr, Ytr, _ = build_arrays(train_eps, groups, H, norm, cfg["label"])
    Xva, Yva, _ = build_arrays(val_eps, groups, H, norm, cfg["label"]) if val_eps else (None, None, None)
    log(f"train: {len(train_eps)} episodes / {len(Xtr)} rows; val: {len(val_eps)} episodes / "
        f"{0 if Xva is None else len(Xva)} rows; inputs={Xtr.shape[1]} ({', '.join(groups)}; history={H})")

    dev = _device(cfg["device"])
    model = MLPPolicy(Xtr.shape[1], Ytr.shape[1], cfg["model"]["hidden"], cfg["model"]["dropout"]).to(dev)
    tc = cfg["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    loss_fn = nn.MSELoss()
    xt, yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Ytr).to(dev)
    xv = torch.from_numpy(Xva).to(dev) if Xva is not None else None
    yv = torch.from_numpy(Yva).to(dev) if Yva is not None else None
    gen = torch.Generator().manual_seed(cfg["seed"])
    half_range = torch.tensor((ACTION_HIGH - ACTION_LOW) / 2)

    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    log_rows, best, best_state = [], float("inf"), None
    status_path = out / "status.json"
    t_start = time.time()
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(xt), generator=gen).to(dev)
        total = 0.0
        for i in range(0, len(xt), tc["batch_size"]):
            idx = perm[i:i + tc["batch_size"]]
            loss = loss_fn(model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        row = {"epoch": epoch, "train_mse": total / len(xt)}
        if xv is not None:
            model.eval()
            with torch.no_grad():
                pv = model(xv)
                row["val_mse"] = loss_fn(pv, yv).item()
                mae = ((pv - yv).abs().mean(0).cpu() * half_range).numpy()   # physical units
            row.update({f"val_mae_{n}": float(v) for n, v in zip(ACTION_NAMES, mae)})
        log_rows.append(row)
        score = row.get("val_mse", row["train_mse"])
        if score < best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        status_path.write_text(json.dumps({"state": "training", "epoch": epoch, "epochs": tc["epochs"], **row}))
        if epoch == 1 or epoch % max(1, tc["epochs"] // 10) == 0 or epoch == tc["epochs"]:
            log(f"epoch {epoch:4d}  train_mse {row['train_mse']:.5f}" +
                (f"  val_mse {row['val_mse']:.5f}" if "val_mse" in row else ""))

    model.load_state_dict(best_state)
    data_files = [Path(cfg["data"]) / f"{i}{s}" for i in usable for s in (".npz", ".json")]
    meta = {
        "name": cfg.get("name", out.name),
        "feature_groups": groups,
        "history": H,
        "obs_names_all": list(OBS_NAMES),
        "feature_names": feature_names(groups, H),
        "action_names": list(ACTION_NAMES),
        "action_low": ACTION_LOW.tolist(),
        "action_high": ACTION_HIGH.tolist(),
        "n_inputs": int(Xtr.shape[1]),
        "n_outputs": int(Ytr.shape[1]),
        "model": cfg["model"],
        "normalizer": norm.to_dict(),
        "normalizer_fitted_on": "training episodes only",
        "label": cfg["label"],
        "train_config": cfg,
        "best_score": best,
        "provenance": {
            "data_dir": str(cfg["data"]),
            "data_digest": file_digest(data_files),
            "sources": sorted({metas[i]["source"] for i in usable}),
            "n_episodes_available": len(metas),
            "train_episode_ids": train_ids,
            "val_episode_ids": val_ids,
            "train_seeds": sorted(metas[i].get("seed") for i in train_ids),
            "git_commit": _git_commit(),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "trained_unix": time.time(),
            "train_seconds": round(time.time() - t_start, 2),
        },
    }
    save_checkpoint(out, model, meta)
    with (out / "train_log.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log_rows[-1].keys()))
        w.writeheader()
        w.writerows(log_rows)
    status_path.write_text(json.dumps({"state": "done", "epoch": tc["epochs"], "epochs": tc["epochs"],
                                       "best_score": best, **log_rows[-1]}))
    log(f"saved checkpoint {out} (best {'val' if xv is not None else 'train'} mse {best:.5f})")
    return out
