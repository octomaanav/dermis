"""Behavioral cloning for fragile_cup. Reuses the original BC building blocks (episode-level split,
history windows inside one episode, train-only normalizer, MLP, checkpoint format) with the
cup's feature groups and a 1-D effort action scaled to [-1, 1] from [MIN_EFFORT, MAX_EFFORT].

Feature groups: no-contact = gripper + object; contact-aware = gripper + object + contact.
Both predict the same effort command from the same episodes and split with the same budget.

Fine-tuning (`init_from: <checkpoint dir>`): starts from that checkpoint's weights and REUSES its
normalizer, feature groups and history (inputs must keep the same scaling). Provenance records the
pretrained checkpoint and *its* training sources next to the new data, so a fine-tuned policy never
looks as if it learned only from the new demonstrations.
"""
from __future__ import annotations

import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..bc.dataset import Normalizer, file_digest, history_windows, split_episodes
from ..bc.model import MLPPolicy, save_checkpoint
from ..bc.train import _git_commit, _seed_everything
from ..data.episodes import EpisodeStore
from .env import MAX_EFFORT, MIN_EFFORT, OBS_NAMES, obs_indices

LO, HI = MIN_EFFORT, MAX_EFFORT          # fragile_cup action bounds (kept for backward compatibility)


def schema(task: str = "fragile_cup") -> dict:
    """Observation/action schema per task. fragile_cup: 1-D effort; assist_hand: [closure, effort]."""
    if task == "fragile_cup":
        return {"obs_names": OBS_NAMES, "obs_indices": obs_indices, "low": np.array([LO], np.float32),
                "high": np.array([HI], np.float32), "action_names": ["effort"]}
    if task == "assist_hand":
        from ..assist import env as A
        return {"obs_names": A.OBS_NAMES, "obs_indices": A.obs_indices, "low": A.ACTION_LOW, "high": A.ACTION_HIGH,
                "action_names": list(A.ACTION_NAMES)}
    raise ValueError(f"unknown task {task!r}")
DEFAULTS = {"feature_groups": ["gripper", "object", "contact"], "history": 4, "label": "expert_actions",
            "val_fraction": 0.2, "split_seed": 0, "seed": 0, "max_episodes": None, "success_only": False,
            "init_from": None,
            "model": {"hidden": [128, 128], "dropout": 0.0},
            "train": {"epochs": 40, "batch_size": 256, "lr": 1e-3, "weight_decay": 1e-5}}


def definitions(task: str) -> dict:
    """Human-readable observation/action definitions stored with every checkpoint."""
    if task == "fragile_cup":
        return {"actions": {"effort": "finger motor force limit, N, clipped to [0.5, 40]; the fingers are always "
                                      "commanded to close, so the squeeze is ~this value"},
                "observations": {"opening": "m", "opening_vel": "m/s", "effort_prev": "N (own previous command)",
                                 "obj_rel_*": "m, cup centre minus gripper grasp point, world frame",
                                 "obj_v*": "m/s, cup velocity relative to the gripper",
                                 "normal_l/r": "N, finger->cup normal force (SIMULATED)",
                                 "shear_z_l/r": "N, vertical tangential finger->cup force (SIMULATED)",
                                 "shear_l/r": "N, tangential force magnitude", "contact_l/r": "0/1, normal > 0.05 N"},
                "history": "observation frames stacked oldest -> newest; episode start padded with the first frame",
                "rate_hz": 30}
    if task == "assist_hand":
        from ..assist import env as A
        return {"actions": {"closure": "0..1 fraction of finger flex range", "effort": "finger motor torque limit, N m"},
                "observations": list(A.OBS_NAMES), "rate_hz": 30}
    return {}


def scale(a, lo=LO, hi=HI):
    return 2 * (np.asarray(a) - lo) / (hi - lo) - 1


def unscale(a, lo=LO, hi=HI):
    return lo + (np.asarray(a) + 1) * 0.5 * (hi - lo)


def _arrays(eps, cols, H, norm, label, lo=LO, hi=HI):
    X, Y = [], []
    for ep in eps:
        X.append(history_windows(norm(ep.obs[:, cols]), H))
        y = ep.arrays["expert_actions"] if label == "expert_actions" and "expert_actions" in ep.arrays else ep.arrays["actions"]
        Y.append(scale(y, lo, hi))
    return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)


def train(cfg: dict, log=print) -> Path:
    cfg = {**DEFAULTS, **cfg, "model": {**DEFAULTS["model"], **cfg.get("model", {})},
           "train": {**DEFAULTS["train"], **cfg.get("train", {})}}
    _seed_everything(cfg["seed"])
    store = EpisodeStore(cfg["data"])
    metas = {i: store.load_meta(i) for i in store.ids()}
    task = cfg.get("task", "fragile_cup")
    sch = schema(task)
    usable = sorted(i for i, m in metas.items() if m.get("task") == task
                    and (not cfg["success_only"] or m.get("success")))
    if cfg["max_episodes"]:
        usable = usable[:cfg["max_episodes"]]
    if not usable:
        raise SystemExit(f"no {task} episodes in {cfg['data']}")
    tr_ids, va_ids = split_episodes(usable, cfg["val_fraction"], cfg["split_seed"])
    if not va_ids:                                    # a single demonstration: train on it, validate on it
        va_ids = list(tr_ids)
    tr, va = [store.load(i) for i in tr_ids], [store.load(i) for i in va_ids]
    import hashlib
    cfgs = {json.dumps(metas[i].get("task_config"), sort_keys=True) for i in usable}
    if len(cfgs) != 1:
        raise SystemExit(f"episodes in {cfg['data']} were recorded under {len(cfgs)} different task configs; "
                         "train on one task version at a time")
    task_cfg = json.loads(next(iter(cfgs)))
    task_cfg_sha = hashlib.sha256(next(iter(cfgs)).encode()).hexdigest()[:12]
    pre = None
    if cfg["init_from"]:
        pre = torch.load(Path(cfg["init_from"]) / "model.pt", map_location="cpu", weights_only=False)
        pm = pre["meta"]
        cfg["feature_groups"], cfg["history"], cfg["model"] = pm["feature_groups"], pm["history"], pm["model"]
    cols, H = sch["obs_indices"](cfg["feature_groups"]), int(cfg["history"])
    lo, hi = sch["low"], sch["high"]
    norm = (Normalizer.from_dict(pre["meta"]["normalizer"]) if pre is not None
            else Normalizer.fit(np.concatenate([e.obs[:, cols] for e in tr])))
    Xtr, Ytr = _arrays(tr, cols, H, norm, cfg["label"], lo, hi)
    Xva, Yva = _arrays(va, cols, H, norm, cfg["label"], lo, hi)
    log(f"[cup train] {cfg.get('name')}: {len(tr)} train eps / {len(Xtr)} rows, {len(va)} val eps; "
        f"inputs {Xtr.shape[1]} ({'+'.join(cfg['feature_groups'])}, history {H})")
    model = MLPPolicy(Xtr.shape[1], Ytr.shape[1], cfg["model"]["hidden"], cfg["model"]["dropout"])
    if pre is not None:
        model.load_state_dict(pre["state_dict"])
    tc = cfg["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    xt, yt, xv, yv = map(torch.from_numpy, (Xtr, Ytr, Xva, Yva))
    gen = torch.Generator().manual_seed(cfg["seed"])
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    rows, best, best_state, t0 = [], float("inf"), None, time.time()
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        tot = 0.0
        for i in range(0, len(xt), tc["batch_size"]):
            idx = perm[i:i + tc["batch_size"]]
            loss = nn.functional.mse_loss(model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(xv)
            vm = nn.functional.mse_loss(pv, yv).item()
            mae_n = float(((pv - yv).abs().mean(0) * torch.from_numpy(np.asarray((hi - lo) / 2, np.float32))).mean())
        rows.append({"epoch": epoch, "train_mse": tot / len(xt), "val_mse": vm, "val_mae_effort_n": mae_n})
        if vm < best:
            best, best_state = vm, {k: v.clone() for k, v in model.state_dict().items()}
        (out / "status.json").write_text(json.dumps({"state": "training", "epoch": epoch, "epochs": tc["epochs"], **rows[-1]}))
    model.load_state_dict(best_state)
    secs = round(time.time() - t0, 2)
    files = [Path(cfg["data"]) / f"{i}{s}" for i in usable for s in (".npz", ".json")]
    on = sch["obs_names"]
    meta = {"name": cfg.get("name", out.name), "task": task, "feature_groups": cfg["feature_groups"],
            "obs_names_all": list(on),
            "task_version": f"{task}/{(task_cfg or {}).get('damage_model', 'n/a')}/cfg-{task_cfg_sha}",
            "task_config": task_cfg, "definitions": definitions(task),
            "history": H, "feature_names": [f"{on[c]}@t-{k}" if k else f"{on[c]}@t"
                                            for k in range(H - 1, -1, -1) for c in cols],
            "action_names": sch["action_names"], "action_low": np.asarray(lo).tolist(),
            "action_high": np.asarray(hi).tolist(), "n_inputs": int(Xtr.shape[1]),
            "n_outputs": int(Ytr.shape[1]), "model": cfg["model"], "normalizer": norm.to_dict(),
            "normalizer_fitted_on": ("reused from pretrained checkpoint" if pre is not None else "training episodes only"),
            "label": cfg["label"], "train_config": cfg,
            "best_val_mse": best, "best_val_mae_effort_n": min(r["val_mae_effort_n"] for r in rows),
            "provenance": {"data_dir": str(cfg["data"]), "data_digest": file_digest(files),
                           "sources": sorted({metas[i]["source"] for i in usable}),
                           "controllers": sorted({metas[i].get("controller", "?") for i in usable}),
                           "train_episode_ids": tr_ids, "val_episode_ids": va_ids,
                           "train_seeds": sorted(metas[i]["seed"] for i in tr_ids),
                           "git_commit": _git_commit(), "torch": torch.__version__, "trained_unix": time.time(),
                           "train_seconds": secs,
                           "pretrained_from": None if pre is None else {
                               "checkpoint": str(cfg["init_from"]), "name": pre["meta"].get("name"),
                               "sources": pre["meta"]["provenance"].get("sources"),
                               "train_episodes": len(pre["meta"]["provenance"].get("train_episode_ids", []))}}}
    save_checkpoint(out, model, meta)
    with (out / "train_log.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "status.json").write_text(json.dumps({"state": "done", "epoch": tc["epochs"], "epochs": tc["epochs"],
                                                  "best_score": best, **rows[-1]}))
    log(f"[cup train] saved {out} in {secs} s (best val mse {best:.5f}, effort MAE {meta['best_val_mae_effort_n']:.2f} N)")
    return out


class CupPolicy:
    """Runtime: observation vector -> effort. Same history padding as training. No human input."""

    def __init__(self, model: MLPPolicy, meta: dict):
        self.model, self.meta = model.eval(), meta
        sch = schema(meta.get("task", "fragile_cup"))
        self.cols = sch["obs_indices"](meta["feature_groups"])
        self.lo, self.hi = np.asarray(meta.get("action_low", [LO]), np.float32), np.asarray(meta.get("action_high", [HI]), np.float32)
        self.H = int(meta["history"])
        self.norm = Normalizer.from_dict(meta["normalizer"])
        self.buf: deque = deque(maxlen=self.H)

    @classmethod
    def load(cls, path) -> "CupPolicy":
        path = Path(path)
        ck = torch.load(path / "model.pt" if path.is_dir() else path, map_location="cpu", weights_only=False)
        m = ck["meta"]
        model = MLPPolicy(m["n_inputs"], m.get("n_outputs", 1), m["model"]["hidden"], m["model"].get("dropout", 0.0))
        model.load_state_dict(ck["state_dict"])
        return cls(model, m)

    def reset(self) -> None:
        self.buf.clear()

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        x = self.norm(np.asarray(obs, np.float32)[self.cols])
        if not self.buf:
            self.buf.extend([x] * (self.H - 1))
        self.buf.append(x)
        y = self.model(torch.from_numpy(np.concatenate(list(self.buf)).astype(np.float32))[None])[0].numpy()
        return np.clip(unscale(y, self.lo, self.hi), self.lo, self.hi).astype(np.float32)
