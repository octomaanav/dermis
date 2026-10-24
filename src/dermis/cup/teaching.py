"""Teaching workflow for fragile_cup: recorded episodes -> dataset snapshot -> background training ->
development evaluation -> explicit activation -> AUTONOMOUS.

    python -m dermis cup snapshot --from data/episodes/cup_teach --name demo1
    python -m dermis cup train-candidate --snapshot data/datasets/cup/demo1 --name cand1 [--init-from CKPT]
    python -m dermis cup candidates                     # status, timings, dev results, which one is ACTIVE
    python -m dermis cup evaluate-candidate --name cand1 # development suite only
    python -m dermis cup activate --name cand1          # refuses without a dev evaluation
    python -m dermis cup judge                          # uses the ACTIVE checkpoint

A snapshot is an immutable copy (with sha256s) of the selected episodes, so a checkpoint always
names exactly the data it saw; recording more episodes later never changes an existing snapshot or
checkpoint. Activation is a separate, explicit step: a newly trained candidate is never used until
someone activates it, and the judge display names the active checkpoint and its data.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CANDIDATES = Path("runs/cup/candidates")
ACTIVE = Path("runs/cup/active_checkpoint.json")
DATASETS = Path("data/datasets/cup")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def snapshot(sources: list[str], name: str, out_root: Path = DATASETS, ids: list[str] | None = None,
             include_sources: list[str] | None = None, success_only: bool = False, last: int | None = None,
             task: str = "fragile_cup") -> Path:
    out = out_root / name
    if out.exists():
        raise SystemExit(f"snapshot {out} already exists (snapshots are immutable; pick a new name)")
    picked = []
    for src in sources:
        for j in sorted(Path(src).glob("*.json")):
            if not j.with_suffix(".npz").exists():
                continue
            m = json.loads(j.read_text())
            if m.get("task") != task:
                continue
            if ids and m["id"] not in ids:
                continue
            if include_sources and m["source"] not in include_sources:
                continue
            if success_only and not m.get("success"):
                continue
            picked.append((m.get("created_unix", 0), j, m))
    picked.sort(key=lambda x: x[0])
    if last:
        picked = picked[-last:]
    if not picked:
        raise SystemExit(f"no {task} episodes matched the selection")
    (out / "episodes").mkdir(parents=True)
    rows = []
    import numpy as np
    for _, j, m in picked:
        for suffix in (".json", ".npz"):
            shutil.copy2(j.with_suffix(suffix), out / "episodes" / (m["id"] + suffix))
        with np.load(j.with_suffix(".npz")) as z:
            tags = z["tags"] if "tags" in z.files else np.array([])
        held = float(np.mean([t.startswith("GLOVE_") and t != "GLOVE_ARMED" for t in tags])) if len(tags) else 0.0
        rows.append({"id": m["id"], "source": m["source"], "controller": m.get("controller"), "length": m.get("length"),
                     "termination": m.get("termination"), "success": m.get("success"), "seed": m.get("seed"),
                     "created_unix": m.get("created_unix"), "non_active_tick_fraction": round(held, 4),
                     "sha256_npz": _sha(j.with_suffix(".npz")), "raw_recording": m.get("raw_recording")})
    counts: dict = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    manifest = {"name": name, "task": task, "created_unix": time.time(), "selection": {
        "from": sources, "ids": ids, "include_sources": include_sources, "success_only": success_only, "last": last},
        "episodes": rows, "counts_by_source": counts,
        "note": "immutable copy; GLOVE ticks tagged PAUSED/DISARMED are held commands, see non_active_tick_fraction"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


def start_training(snap: str, name: str, config: str | dict, init_from: str | None = None, epochs: int | None = None,
                   seed: int = 0, foreground: bool = False, root: Path | None = None) -> Path:
    d = (root or CANDIDATES) / name
    if d.exists():
        raise SystemExit(f"candidate {d} exists; pick a new name")
    d.mkdir(parents=True)
    snap_p = Path(snap)
    manifest = json.loads((snap_p / "manifest.json").read_text())
    from ..config import load_config
    cfg = dict(config) if isinstance(config, dict) else load_config(config)
    cfg.setdefault("task", manifest.get("task", "fragile_cup"))
    cfg.update({"name": name, "data": str(snap_p / "episodes"), "out": str(d / "checkpoint"), "seed": seed,
                "val_fraction": cfg.get("val_fraction", 0.2)})
    if init_from:
        cfg["init_from"] = init_from
    if epochs:
        cfg.setdefault("train", {})["epochs"] = epochs
    (d / "train_config.json").write_text(json.dumps(cfg, indent=1))
    info = {"name": name, "task": cfg["task"], "snapshot": str(snap_p), "snapshot_counts": manifest["counts_by_source"],
            "n_episodes": len(manifest["episodes"]), "init_from": init_from,
            "training": "fine-tune" if init_from else "from scratch",
            "config": config if isinstance(config, str) else "inline",
            "started_unix": time.time(), "state": "training"}
    cmd = [sys.executable, "-c", "import json,sys; from dermis.cup.train import train; "
           "train(json.load(open(sys.argv[1])))", str(d / "train_config.json")]
    if foreground:
        t0 = time.time()
        r = subprocess.run(cmd, stdout=open(d / "train.log", "w"), stderr=subprocess.STDOUT)
        info.update(state="done" if r.returncode == 0 else f"failed ({r.returncode})", wall_s=round(time.time() - t0, 2))
    else:
        proc = subprocess.Popen(cmd, stdout=open(d / "train.log", "w"), stderr=subprocess.STDOUT,
                                start_new_session=True)
        info["pid"] = proc.pid
    (d / "candidate.json").write_text(json.dumps(info, indent=1))
    return d


def status(name: str, root: Path | None = None, active: Path | None = None) -> dict:
    d = (root or CANDIDATES) / name
    info = json.loads((d / "candidate.json").read_text())
    ck = d / "checkpoint"
    if info["state"] == "training":
        alive = _alive(info.get("pid"))
        if (ck / "model.pt").exists() and not alive:
            info["state"] = "done"
        elif not alive:
            info["state"] = "failed (see train.log)"
    if (ck / "metadata.json").exists():
        meta = json.loads((ck / "metadata.json").read_text())
        info["train_seconds"] = meta["provenance"]["train_seconds"]
        info["pretrained_from"] = meta["provenance"].get("pretrained_from")
        info["best_val_mae_effort_n"] = round(meta["best_val_mae_effort_n"], 3)
    elif (ck / "status.json").exists():
        info["progress"] = json.loads((ck / "status.json").read_text())
    ev = d / "eval_dev.json"
    if ev.exists():
        e = json.loads(ev.read_text())
        info["dev_eval"] = {"success": f"{e['summary']['successes']}/{e['summary']['attempts']}",
                            "terminations": e["summary"]["terminations"], "wall_s": e["wall_seconds"]}
    ap = active or ACTIVE
    act = json.loads(ap.read_text()) if ap.exists() else {}
    info["active"] = act.get("checkpoint") == str(ck)
    (d / "candidate.json").write_text(json.dumps({k: v for k, v in info.items() if k not in ("active", "progress")}, indent=1))
    return info


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:                                              # reap if it is our finished child (zombie)
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


def evaluate_candidate(name: str, suite: str, workers: int) -> dict:
    from .evaluate import evaluate
    d = CANDIDATES / name
    if not (d / "checkpoint" / "model.pt").exists():
        raise SystemExit(f"{name} has no checkpoint yet (state: {status(name)['state']})")
    return evaluate({"kind": "bc", "checkpoint": str(d / "checkpoint")}, suite, str(d / "eval_dev.json"), workers)


def activate(name: str, force: bool = False, root: Path | None = None, active: Path | None = None) -> dict:
    d = (root or CANDIDATES) / name
    st = status(name, root, active)
    if not (d / "checkpoint" / "model.pt").exists():
        raise SystemExit(f"{name}: no checkpoint")
    if "dev_eval" not in st and not force:
        raise SystemExit(f"{name}: no development evaluation yet; run `cup evaluate-candidate --name {name}` first")
    rec = {"checkpoint": str(d / "checkpoint"), "candidate": name, "activated_unix": time.time(),
           "snapshot": st["snapshot"], "snapshot_counts": st["snapshot_counts"], "dev_eval": st.get("dev_eval"),
           "pretrained_from": st.get("pretrained_from"), "forced_without_eval": bool(force and "dev_eval" not in st)}
    ap = active or ACTIVE
    ap.parent.mkdir(parents=True, exist_ok=True)
    ap.write_text(json.dumps(rec, indent=1))
    return rec


def active_checkpoint(active: Path | None = None) -> str | None:
    ap = active or ACTIVE
    if ap.exists():
        c = json.loads(ap.read_text())["checkpoint"]
        if Path(c, "model.pt").exists():
            return c
    return None


def teaching_study(out: str, ns: list[int], pretrained: str | None, suite: str, workers: int,
                   demonstrator: str = "slip_reflex", log=print) -> dict:
    """Measures the mechanics and cost of live teaching with SCRIPTED stand-in demonstrations
    (seeds 400000+, never used elsewhere). For each N: snapshot of the N most recent demos ->
    train from scratch, and fine-tune from `pretrained` if given -> evaluate on the dev suite.
    Scripted demos are cleaner and more consistent than human keyboard demos, so these numbers are
    an optimistic bound for fresh human demonstrations, not a prediction."""
    global CANDIDATES
    from ..data.episodes import EpisodeStore
    from .env import CupConfig
    from .evaluate import evaluate
    from .rollout import collect
    from ..config import load_config
    root = Path(out)
    old = CANDIDATES
    CANDIDATES = root / "candidates"
    try:
        demos = root / "demos"
        t0 = time.time()
        collect(str(demos), list(range(400_000, 400_000 + max(ns))), demonstrator, 1.0,
                CupConfig.from_dict(load_config("configs/cup/task.yaml")).to_dict())
        collect_wall = time.time() - t0
        store = EpisodeStore(demos)
        lengths = [store.load_meta(i)["length"] for i in store.ids()]
        rows = []
        if pretrained:
            t0 = time.time()
            e = evaluate({"kind": "bc", "checkpoint": pretrained}, suite, str(root / "pretrained_only.json"), workers, log=log)
            meta = json.loads((Path(pretrained) / "metadata.json").read_text())
            rows.append({"n_demos": 0, "mode": "pretrained only (no new demos)", "train_s": None,
                         "eval_s": round(time.time() - t0, 1), "dev_success": f"{e['summary']['successes']}/{e['summary']['attempts']}",
                         "terminations": e["summary"]["terminations"],
                         "pretrained_sources": meta["provenance"]["sources"],
                         "pretrained_train_episodes": len(meta["provenance"]["train_episode_ids"])})
        for n in ns:
            snap = snapshot([str(demos)], f"n{n}", out_root=root / "datasets", last=n)
            for mode, init in (("from scratch", None), ("fine-tune", pretrained)):
                if mode == "fine-tune" and not pretrained:
                    continue
                name = f"n{n}_{'ft' if init else 'scratch'}"
                t0 = time.time()
                start_training(str(snap), name, "configs/cup/bc_contact.yaml", init_from=init, foreground=True)
                train_s = time.time() - t0
                if not (CANDIDATES / name / "checkpoint" / "model.pt").exists():
                    rows.append({"n_demos": n, "mode": mode, "train_s": round(train_s, 1), "dev_success": "training failed",
                                 "log": str(CANDIDATES / name / "train.log")})
                    continue
                t0 = time.time()
                e = evaluate({"kind": "bc", "checkpoint": str(CANDIDATES / name / "checkpoint")}, suite,
                             str(CANDIDATES / name / "eval_dev.json"), workers, log=log)
                st = status(name)
                rows.append({"n_demos": n, "mode": mode, "train_s": round(train_s, 1),
                             "train_s_inside_trainer": st.get("train_seconds"), "eval_s": round(time.time() - t0, 1),
                             "dev_success": f"{e['summary']['successes']}/{e['summary']['attempts']}",
                             "terminations": e["summary"]["terminations"]})
                log(f"[teaching-study] N={n} {mode}: dev {rows[-1]['dev_success']} train {train_s:.1f}s "
                    f"eval {rows[-1]['eval_s']}s")
        res = {"demonstrator": f"{demonstrator} (SCRIPTED stand-in for a human demonstrator)", "suite": suite,
               "demo_ticks": lengths, "demo_seconds_each_realtime": [round(l / 30, 1) for l in lengths],
               "collection_wall_s_headless": round(collect_wall, 1), "rows": rows,
               "note": "Recording a demonstration in TEACH runs in real time: about the episode length per demo plus "
                       "reset; scripted collection here is headless and faster."}
        (root / "teaching_study.json").write_text(json.dumps(res, indent=1))
        return res
    finally:
        CANDIDATES = old
