"""`python -m dermis cup <command>`: the fragile_cup experiment (separate from the original benchmark)."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from ..config import load_config


def _cfg(path):
    from .env import CupConfig
    return CupConfig.from_dict(load_config(path) if path else None)


def _workers(n):
    from ..evaluate import default_workers
    return default_workers() if n is None else n


def cmd_check(a):
    from .evaluate import check_task
    check_task(a.suite, [float(x) for x in a.efforts.split(",")], a.out, _workers(a.workers), a.damage_models.split(","))
    print(Path(a.out, "task_check.md").read_text())
    return 0


def cmd_collect(a):
    from .rollout import collect_parallel
    t0 = time.time()
    res = collect_parallel(a.out, a.episodes, a.seed_start, a.controller, a.noise, _cfg(a.task), _workers(a.workers))
    terms = {}
    for r in res:
        terms[r["termination"]] = terms.get(r["termination"], 0) + 1
    print(f"[SCRIPTED] {a.controller}: {len(res)} episodes -> {a.out} {terms} ({time.time() - t0:.0f} s)")
    return 0


def cmd_train(a):
    from .train import train
    cfg = load_config(a.config)
    for k in ("data", "out"):
        if getattr(a, k):
            cfg[k] = getattr(a, k)
    train(cfg)
    return 0


def cmd_evaluate(a):
    from .evaluate import evaluate
    spec = {"kind": "scripted", "name": a.scripted} if a.scripted else {"kind": "bc", "checkpoint": a.checkpoint}
    name = a.scripted or Path(a.checkpoint).name
    out = a.out or f"results/cup/{name}_{Path(a.suite).stem}.json"
    evaluate(spec, a.suite, out, 1 if a.gui else _workers(a.workers), gui=a.gui, damage_model=a.damage_model)
    return 0


def report(results: list[str], out_dir: str, title: str) -> str:
    rows = [json.loads(Path(r).read_text()) for r in results]
    lines = [f"# {title}", "", f"Suite `{rows[0]['suite']}`, damage model `{rows[0]['damage_model']}`. {rows[0]['note']}",
             "Separate from the grasp-lift-place benchmark; metrics are not merged.", "",
             "| Controller | Mode | Success | Terminations | Failure state distribution | Peak normal N (median / max) | Mean effort N (median) | Final slip mm (median) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s, c = r["summary"], r["controller"]
        lines.append(f"| {c['name']} | {c['label']} | {s['successes']}/{s['attempts']} | {s['terminations']} | "
                     f"{s['failure_state_distribution'] or '–'} | {s['peak_normal_n_median']:.1f} / {s['peak_normal_n_max']:.1f} | "
                     f"{s['mean_effort_n_median']:.1f} | {s['final_slip_mm_median']:.1f} |")
    hashes = [[e["scenario_hash"] for e in r["episodes"]] for r in rows]
    same = all(h == hashes[0] for h in hashes)
    lines += ["", f"Identical scenarios (initial conditions + load schedules) across all rows: **{same}** "
              f"({len(hashes[0])} scenario hashes compared)."]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    md = "\n".join(lines) + "\n"
    Path(out_dir, "report.md").write_text(md)
    return md


def seed_summary(results: list[str], names: list[str], seeds: list[int]) -> str:
    by = {json.loads(Path(r).read_text())["controller"]["name"]: json.loads(Path(r).read_text())["summary"] for r in results}
    lines = ["", "## Across training seeds (same episodes, split, budget and scenarios)", "",
             "| Policy | Success per seed | Mean | Terminations (all seeds) |", "|---|---|---|---|"]
    for n in names:
        cells = [by[n if s == 0 else f"{n}_s{s}"] for s in seeds]
        terms: dict = {}
        for c in cells:
            for k, v in c["terminations"].items():
                terms[k] = terms.get(k, 0) + v
        per_seed = ", ".join(f"{c['successes']}/{c['attempts']}" for c in cells)
        lines.append(f"| {n} | {per_seed} | "
                     f"{sum(c['success_rate'] for c in cells) / len(cells):.3f} | {terms} |")
    return "\n".join(lines) + "\n"


def cmd_benchmark(a):
    from .evaluate import evaluate
    from .rollout import collect_parallel
    from .train import train
    b = load_config(a.config)
    root = Path(a.out or b["out"])
    w = _workers(a.workers)
    data = root / "episodes"
    if a.recollect and data.exists():
        shutil.rmtree(data)
    if not data.exists():
        c = b["collect"]
        res = collect_parallel(str(data), c["episodes"], c.get("seed_start", 0), c["controller"], c.get("noise", 0.0),
                               _cfg(b.get("task")), w)
        print(f"[SCRIPTED] collected {len(res)} episodes with {c['controller']}")
    suite = a.suite or b["suite"]
    dm = load_config(b["task"]).get("damage_model") if b.get("task") else None
    results = []
    for name in b.get("reference_controllers", []):
        pth = root / "results" / f"{name}.json"
        if a.force or not pth.exists():
            evaluate({"kind": "scripted", "name": name}, suite, str(pth), w, damage_model=dm)
        results.append(str(pth))
    timings = {}
    seeds = b.get("train_seeds", [0])
    for seed in seeds:
        for pol in b["policies"]:
            name = pol["name"] if seed == 0 else f"{pol['name']}_s{seed}"       # seed 0 keeps the default checkpoint path
            pth = root / "results" / f"{name}.json"
            ck = root / "runs" / name
            if a.force or not pth.exists():
                cfg = load_config(pol["config"])
                cfg.update({"data": str(data), "out": str(ck), "name": name, "seed": seed})
                t0 = time.time()
                train(cfg)
                timings[name] = round(time.time() - t0, 1)
                evaluate({"kind": "bc", "checkpoint": str(ck)}, suite, str(pth), w, damage_model=dm)
            results.append(str(pth))
    md = report(results, str(root / "report"), b.get("title", "fragile_cup benchmark"))
    md += "\nTraining wall time (s) of runs trained in this invocation: " + json.dumps(timings) + "\n"
    if len(seeds) > 1:
        md += seed_summary(results, [p["name"] for p in b["policies"]], seeds)
    Path(root, "report", "report.md").write_text(md)
    print(md)
    return 0


def cmd_smoke(a):
    """Pipeline check only: tiny data and training; results are not a performance claim."""
    from .evaluate import evaluate
    from .rollout import collect
    from .train import CupPolicy, train
    root = Path(a.out)
    if root.exists():
        shutil.rmtree(root)
    t0 = time.time()
    cfg = _cfg(a.task)
    print("[cup smoke 1/4] scripted demonstrations")
    collect(str(root / "episodes"), list(range(a.episodes)), "slip_reflex", 1.0, cfg.to_dict())
    print("[cup smoke 2/4] training both variants briefly")
    times = {}
    for name, groups in (("no_contact", ["gripper", "object"]), ("contact", ["gripper", "object", "contact"])):
        t = time.time()
        train({"name": name, "data": str(root / "episodes"), "out": str(root / name), "feature_groups": groups,
               "model": {"hidden": [32, 32]}, "train": {"epochs": a.epochs}})
        times[name] = round(time.time() - t, 2)
        CupPolicy.load(root / name)
    print("[cup smoke 3/4] AUTONOMOUS evaluation on identical scenarios")
    suite = root / "suite.json"
    suite.write_text(json.dumps({"name": "cup_smoke", "task_config": cfg.to_dict(),
                                 "generate": {"seed_start": 1_500_000, "count": a.eval_episodes}}))
    res = [str(root / f"results_{n}.json") for n in times]
    for n, r in zip(times, res):
        evaluate({"kind": "bc", "checkpoint": str(root / n)}, str(suite), r, 1)
    print("[cup smoke 4/4] report")
    md = report(res, str(root), "fragile_cup smoke test (pipeline check, not performance)")
    summary = {"ok": True, "seconds": round(time.time() - t0, 1), "train_seconds": times}
    Path(root, "smoke_summary.json").write_text(json.dumps(summary, indent=1))
    print(md + json.dumps(summary, indent=1))
    return 0


def open_glove(a):
    """Live glove input for TEACH: source -> raw recording (separate file, host receive times) ->
    armed two-point-calibrated GripControl -> GloveGrip (same act() interface as the keyboard)."""
    import time as _t
    from ..glove.grip import GripControl, GripMappingConfig, TwoPointCalibration
    from ..glove.procsource import ProcessTelemetry
    from .controllers import GloveGrip
    if not a.experimental_glove_teleop or not a.grip_calibration:
        raise SystemExit("glove TEACH needs --experimental-glove-teleop and --grip-calibration "
                         "(from `python -m dermis glove inspect`: 0, 1, c)")
    mapping = GripMappingConfig.load(a.mapping)
    cal = TwoPointCalibration.load(a.grip_calibration)
    pods = list(dict.fromkeys(list(mapping.required_pods) + list(mapping.optional_pods) + list(cal.pods)))
    raw = Path(a.raw_dir) / f"glove-raw-{_t.strftime('%Y%m%dT%H%M%S')}.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    # ingestion runs in its own process: PyBullet holds the GIL and starves in-process threads
    runner = ProcessTelemetry({"kind": a.glove_input, "path": a.glove_path, "host": a.glove_host, "port": a.glove_port,
                               "serial_port": a.serial_port, "wire": a.wire, "loop": True, "profile": "step"},
                              pods, record_path=str(raw), timeout_s=mapping.watchdog_s)
    glove = GloveGrip(runner, GripControl(mapping, cal), enabled=True)
    glove.raw_recording = str(raw)
    print(f"[GLOVE] raw sensor recording -> {raw} (separate from simulator episodes; linked by host time)")
    print("[GLOVE] press A in the window to ARM (requires live, usable, matching input); X disarms")
    return glove


def cmd_teach(a):
    from .live import TEACH, run_gui
    glove = open_glove(a) if a.input == "glove" else None
    return run_gui(TEACH, a.seed, [], a.out, _cfg(a.task), glove=glove)


def cmd_judge(a):
    from .live import AUTONOMOUS, run_gui
    if not a.checkpoint:
        from .teaching import active_checkpoint
        act = active_checkpoint()
        a.checkpoint = [act] if act else DEFAULT_CKPTS
        print(f"[AUTONOMOUS] using {'ACTIVE checkpoint ' + act if act else 'benchmark checkpoints (no active candidate)'}")
    for c in a.checkpoint:
        if not Path(c, "model.pt").exists():
            print(f"missing checkpoint {c}: run `python -m dermis cup benchmark` first")
            return 2
    return run_gui(AUTONOMOUS, a.seed, a.checkpoint, "", _cfg(a.task))


def cmd_snapshot(a):
    from .teaching import snapshot
    out = snapshot(a.sources, a.name, ids=a.ids.split(",") if a.ids else None,
                   include_sources=a.source_filter.split(",") if a.source_filter else None,
                   success_only=a.success_only, last=a.last)
    m = json.loads((out / "manifest.json").read_text())
    print(f"snapshot {out}: {len(m['episodes'])} episodes {m['counts_by_source']}")
    return 0


def cmd_train_candidate(a):
    from .teaching import start_training, status
    d = start_training(a.snapshot, a.name, a.config, a.init_from, a.epochs, a.seed, a.foreground)
    st = status(a.name)
    print(f"candidate {a.name}: {st['state']} ({d}); log {d / 'train.log'}"
          + (f"; fine-tuning from {a.init_from}" if a.init_from else ""))
    return 0 if not st["state"].startswith("failed") else 1


def cmd_candidates(a):
    from .teaching import CANDIDATES, status
    for d in sorted(CANDIDATES.glob("*/candidate.json")):
        st = status(d.parent.name)
        print(json.dumps({k: st.get(k) for k in ("name", "state", "active", "n_episodes", "snapshot_counts", "train_seconds",
                                                 "best_val_mae_effort_n", "dev_eval", "pretrained_from", "progress")}))
    return 0


def cmd_evaluate_candidate(a):
    from .teaching import evaluate_candidate
    evaluate_candidate(a.name, a.suite, _workers(a.workers))
    return 0


def cmd_activate(a):
    from .teaching import activate
    r = activate(a.name, a.force)
    print(f"ACTIVE checkpoint: {r['checkpoint']} (snapshot {r['snapshot']} {r['snapshot_counts']}; dev {r['dev_eval']})"
          + ("  [FORCED without evaluation]" if r["forced_without_eval"] else ""))
    return 0


def cmd_teaching_study(a):
    from .teaching import teaching_study
    r = teaching_study(a.out, [int(x) for x in a.ns.split(",")], a.pretrained, a.suite, _workers(a.workers))
    print(json.dumps(r, indent=1))
    return 0


def cmd_worker(a):
    from .worker import plot_trace, run_trace
    spec = {"kind": "scripted", "name": a.scripted} if a.scripted else {"kind": "bc", "checkpoint": a.checkpoint}
    r = run_trace(spec, a.seed, a.out, a.pour_at)
    if a.plot:
        plot_trace(a.out, a.plot)
    print(json.dumps(r, indent=1))
    return 0


def cmd_inspect(a):
    from .inspect import inspect
    r = inspect(a.data, a.n)
    print(json.dumps(r["summary"], indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))
    return 0


def cmd_replay(a):
    """REPLAY a recorded episode: same scenario, recorded efforts played open loop. Not learned control."""
    import numpy as np
    from ..data.episodes import load_episode
    from .controllers import ReplayEffort
    from .env import CupConfig, CupHoldEnv, CupScenario
    from .rollout import run_episode
    ep = load_episode(Path(a.episode).with_suffix(""))
    env = CupHoldEnv(CupConfig.from_dict(ep.meta["task_config"]))
    view = None
    if a.gui:
        from .viewer import CupViewer
        view = CupViewer(env, controls=False)
        view.text("mode", f"REPLAY of {ep.meta['label']} episode {ep.meta['id']} (recorded commands, open loop)", 0,
                  (0.8, 0.5, 0.0))
    oc = ep.meta["outcome"]                              # older scripted episodes poured at pre_pour_s;
    pour_at = oc["pour_t_s"] if "pour_t_s" in oc else env.cfg.pre_pour_s   # None = never poured
    tick = (lambda e, *_: (view.sync(e), view.status(e))) if view else None
    out, _ = run_episode(env, CupScenario.from_dict(ep.meta["scenario"]), ReplayEffort(ep.actions, ep.id),
                         pour_at_s=pour_at, realtime=a.gui, on_tick=tick)
    print(f"[REPLAY] {ep.id}: recorded {ep.meta['termination']} -> replayed {out['termination']} "
          f"(open-loop playback of issued efforts, not autonomous)")
    env.close()
    if view:
        view.close()
    return 0


DEFAULT_CKPTS = ["runs/cup/benchmark/runs/cup_bc_no_contact", "runs/cup/benchmark/runs/cup_bc_contact"]


def add_parser(sub) -> None:
    cup = sub.add_parser("cup", help="fragile_cup experiment (separate benchmark)")
    cs = cup.add_subparsers(dest="cup_cmd", required=True)

    sp = cs.add_parser("check-task", help="is adaptation needed? fixed efforts vs feedback controllers")
    sp.add_argument("--suite", default="configs/cup/dev.yaml")
    sp.add_argument("--efforts", default="2,4,6,8,10,12,14,16,18,20,22,24")
    sp.add_argument("--damage-models", default="instant,instant+sustained")
    sp.add_argument("--out", default="runs/cup/task_check")
    sp.add_argument("--workers", type=int)
    sp.set_defaults(fn=cmd_check)

    sp = cs.add_parser("collect", help="SCRIPTED demonstrations")
    sp.add_argument("--episodes", type=int, default=200)
    sp.add_argument("--seed-start", type=int, default=0)
    sp.add_argument("--controller", default="slip_reflex")
    sp.add_argument("--noise", type=float, default=1.0, help="execution noise std on effort, N (labels stay clean)")
    sp.add_argument("--task", default="configs/cup/task.yaml")
    sp.add_argument("--out", default="data/episodes/cup_scripted")
    sp.add_argument("--workers", type=int)
    sp.set_defaults(fn=cmd_collect)

    sp = cs.add_parser("train", help="behavioral cloning")
    sp.add_argument("--config", required=True)
    sp.add_argument("--data")
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_train)

    sp = cs.add_parser("evaluate", help="evaluate a checkpoint (AUTONOMOUS) or scripted controller on a suite")
    sp.add_argument("--checkpoint")
    sp.add_argument("--scripted", help="fixed_<N>N | grip_load_expert | slip_reflex | oracle_privileged")
    sp.add_argument("--suite", default="configs/cup/heldout.yaml")
    sp.add_argument("--damage-model", help="override the suite's damage model")
    sp.add_argument("--out")
    sp.add_argument("--workers", type=int)
    sp.add_argument("--gui", action="store_true")
    sp.set_defaults(fn=cmd_evaluate)

    sp = cs.add_parser("benchmark", help="collect -> train both -> evaluate on held-out -> report")
    sp.add_argument("--config", default="configs/cup/benchmark.yaml")
    sp.add_argument("--suite")
    sp.add_argument("--out")
    sp.add_argument("--recollect", action="store_true")
    sp.add_argument("--force", action="store_true", help="re-run evaluations/training even if results exist")
    sp.add_argument("--workers", type=int)
    sp.set_defaults(fn=cmd_benchmark)

    sp = cs.add_parser("smoke-test", help="fast fragile_cup pipeline check")
    sp.add_argument("--out", default="runs/cup/smoke")
    sp.add_argument("--episodes", type=int, default=6)
    sp.add_argument("--epochs", type=int, default=3)
    sp.add_argument("--eval-episodes", type=int, default=3)
    sp.add_argument("--task", default="configs/cup/task.yaml")
    sp.set_defaults(fn=cmd_smoke)

    sp = cs.add_parser("teach", help="GUI TEACH mode: human grip control, recorded")
    sp.add_argument("--input", choices=["keyboard", "glove"], default="keyboard")
    sp.add_argument("--seed", type=int, default=500_000)
    sp.add_argument("--out", default="data/episodes/cup_teach")
    sp.add_argument("--task", default="configs/cup/task.yaml")
    sp.add_argument("--experimental-glove-teleop", action="store_true")
    sp.add_argument("--grip-calibration", help="grip_calibration.json from `glove inspect` (0, 1, c)")
    sp.add_argument("--mapping", default="configs/glove/grip.yaml")
    sp.add_argument("--glove-input", default="serial", choices=["serial", "tcp", "stdin", "replay", "synthetic", "wire-file"])
    sp.add_argument("--glove-path")
    sp.add_argument("--glove-host", default="127.0.0.1", help="tcp: listen address (0.0.0.0 to accept the glove over LAN)")
    sp.add_argument("--glove-port", type=int, default=8765)
    sp.add_argument("--serial-port")
    sp.add_argument("--wire", action="store_true")
    sp.add_argument("--raw-dir", default="data/glove_raw", help="raw sensor recordings (kept separate from episodes)")
    sp.set_defaults(fn=cmd_teach)

    sp = cs.add_parser("snapshot", help="copy selected episodes into an immutable dataset snapshot")
    sp.add_argument("--from", dest="sources", action="append", required=True, help="episode directory (repeatable)")
    sp.add_argument("--name", required=True)
    sp.add_argument("--ids", help="comma list of episode ids")
    sp.add_argument("--source-filter", help="comma list: keyboard,glove,scripted")
    sp.add_argument("--success-only", action="store_true")
    sp.add_argument("--last", type=int, help="only the N most recent episodes")
    sp.set_defaults(fn=cmd_snapshot)

    sp = cs.add_parser("train-candidate", help="train a candidate policy from a snapshot (background by default)")
    sp.add_argument("--snapshot", required=True)
    sp.add_argument("--name", required=True)
    sp.add_argument("--config", default="configs/cup/bc_contact.yaml")
    sp.add_argument("--init-from", help="fine-tune from this checkpoint dir (its sources are disclosed)")
    sp.add_argument("--epochs", type=int)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--foreground", action="store_true")
    sp.set_defaults(fn=cmd_train_candidate)

    sp = cs.add_parser("candidates", help="list candidates: training state, timings, dev results, ACTIVE")
    sp.set_defaults(fn=cmd_candidates)

    sp = cs.add_parser("evaluate-candidate", help="evaluate a candidate on the development suite")
    sp.add_argument("--name", required=True)
    sp.add_argument("--suite", default="configs/cup/dev.yaml")
    sp.add_argument("--workers", type=int)
    sp.set_defaults(fn=cmd_evaluate_candidate)

    sp = cs.add_parser("activate", help="explicitly make a candidate the AUTONOMOUS checkpoint")
    sp.add_argument("--name", required=True)
    sp.add_argument("--force", action="store_true", help="activate without a dev evaluation (recorded as forced)")
    sp.set_defaults(fn=cmd_activate)

    sp = cs.add_parser("teaching-study", help="measure teaching cost/outcome with SCRIPTED stand-in demos")
    sp.add_argument("--out", default="runs/cup/teaching_study")
    sp.add_argument("--ns", default="1,3,10")
    sp.add_argument("--pretrained", default="runs/cup/benchmark/runs/cup_bc_contact")
    sp.add_argument("--suite", default="configs/cup/dev.yaml")
    sp.add_argument("--workers", type=int)
    sp.set_defaults(fn=cmd_teaching_study)

    sp = cs.add_parser("worker", help="headless run writing a per-tick physics trace (joints, cup pose, contact, status)")
    sp.add_argument("--checkpoint")
    sp.add_argument("--scripted", help="fixed_<N>N | slip_reflex | hybrid_expert | grip_load_expert | oracle_privileged")
    sp.add_argument("--seed", type=int, default=2_000_000)
    sp.add_argument("--pour-at", type=float, default=1.0)
    sp.add_argument("--out", required=True)
    sp.add_argument("--plot")
    sp.set_defaults(fn=cmd_worker)

    sp = cs.add_parser("inspect-episodes", help="check observation/action alignment and contact plausibility")
    sp.add_argument("--data", required=True)
    sp.add_argument("--n", type=int, default=6)
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_inspect)

    sp = cs.add_parser("replay", help="REPLAY a recorded episode's issued efforts (open loop; not autonomous)")
    sp.add_argument("--episode", required=True, help="path to <id>.json")
    sp.add_argument("--gui", action="store_true")
    sp.set_defaults(fn=cmd_replay)

    sp = cs.add_parser("judge", help="GUI AUTONOMOUS mode: Pour / Reset / paired comparison")
    sp.add_argument("--checkpoint", action="append", default=None)
    sp.add_argument("--seed", type=int, default=3_000_000)
    sp.add_argument("--task", default="configs/cup/task.yaml")
    sp.set_defaults(fn=lambda a: cmd_judge(a if a.checkpoint else _with(a, checkpoint=DEFAULT_CKPTS)))


def _with(a, **kw):
    for k, v in kw.items():
        setattr(a, k, v)
    return a
