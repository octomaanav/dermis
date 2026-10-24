"""python -m dermis <command>  (run `python -m dermis -h` for the list)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from .config import load_config


def _workers(n):
    from .evaluate import default_workers
    return default_workers() if n is None else n


def cmd_collect(a) -> int:
    if a.source != "scripted":
        print("collect only automates the scripted expert; use `demo --source keyboard|glove` for human demos")
        return 2
    from .sim.env import TaskConfig
    from .sim.rollout import collect_scripted, collect_scripted_parallel
    cfg = TaskConfig.from_dict(load_config(a.task)) if a.task else TaskConfig()
    t0 = time.time()
    if a.gui:
        res = collect_scripted(a.out, list(range(a.seed_start, a.seed_start + a.episodes)), a.noise, cfg, gui=True)
    else:
        res = collect_scripted_parallel(a.out, a.episodes, a.seed_start, a.noise, cfg, workers=_workers(a.workers))
    ok = sum(r["success"] for r in res)
    print(f"[SCRIPTED] {ok}/{len(res)} episodes succeeded; saved to {a.out} ({time.time() - t0:.0f} s)")
    return 0


def cmd_train(a) -> int:
    from .bc.train import train
    cfg = load_config(a.config)
    for k in ("data", "out", "epochs"):
        v = getattr(a, k)
        if v is not None:
            if k == "epochs":
                cfg.setdefault("train", {})["epochs"] = v
            else:
                cfg[k] = v
    train(cfg)
    return 0


def cmd_evaluate(a) -> int:
    from .evaluate import evaluate
    out = a.out or (f"results/{Path(a.checkpoint).name if a.checkpoint else 'scripted'}_"
                    f"{Path(a.suite).stem}.json")
    evaluate(a.checkpoint, a.suite, out, workers=1 if a.gui else _workers(a.workers), gui=a.gui, scripted=a.scripted)
    return 0


def cmd_make_suite(a) -> int:
    from .evaluate import materialize_suite
    s = materialize_suite(a.spec, a.out)
    print(f"wrote {len(s['scenarios'])} scenarios to {a.out}")
    return 0


def cmd_compare(a) -> int:
    from .report import compare
    r = compare(a.results, a.out, a.title)
    print(r["markdown"])
    print(f"\nwrote {a.out}/comparison.md, comparison.json, comparison.png")
    return 0


def cmd_smoke(a) -> int:
    """Pipeline check only: tiny data, tiny training. Says nothing about policy quality."""
    from .bc.model import BCPolicy
    from .bc.train import train
    from .evaluate import evaluate
    from .sim.rollout import collect_scripted
    root = Path(a.out)
    if root.exists():
        shutil.rmtree(root)
    t0 = time.time()
    print("[smoke 1/5] collecting scripted demonstrations")
    res = collect_scripted(str(root / "episodes"), list(range(a.episodes)), noise_std=0.03, log=print)
    if not any(r["success"] for r in res):
        print("SMOKE FAIL: scripted expert completed no episode")
        return 1
    print("[smoke 2/5] training a tiny policy")
    ckpt = train({"name": "smoke", "data": str(root / "episodes"), "out": str(root / "run"), "history": 2,
                  "model": {"hidden": [32, 32]}, "train": {"epochs": 3, "batch_size": 128}})
    print("[smoke 3/5] reloading checkpoint in a fresh evaluation path")
    BCPolicy.load(ckpt)                                    # fails loudly on a broken checkpoint
    suite = root / "smoke_suite.json"
    suite.write_text(json.dumps({"name": "smoke", "generate": {"seed_start": 1_500_000, "count": a.eval_episodes}}))
    print("[smoke 4/5] headless AUTONOMOUS episodes")
    result = evaluate(str(ckpt), str(suite), str(root / "results.json"), workers=1)
    print("[smoke 5/5] results written")
    s = result["summary"]
    summary = {"ok": True, "seconds": round(time.time() - t0, 1), "demos_collected": len(res),
               "demos_succeeded": sum(r["success"] for r in res), "eval_attempts": s["attempts"],
               "eval_successes": s["successes"], "terminations": s["terminations"],
               "note": "pipeline execution check; success counts here are not a performance claim"}
    (root / "smoke_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


def cmd_benchmark(a) -> int:
    """Reproducible: collect -> train both BC variants -> expert + both policies on the same suite -> compare."""
    from .bc.train import train
    from .evaluate import evaluate
    from .report import compare
    from .sim.rollout import collect_scripted_parallel
    bench = load_config(a.config)
    root = Path(a.out or bench.get("out", "runs/benchmark"))
    data = root / "episodes"
    workers = _workers(a.workers)
    c = bench["collect"]
    if not data.exists() or a.recollect:
        if data.exists():
            shutil.rmtree(data)
        res = collect_scripted_parallel(str(data), c["episodes"], c.get("seed_start", 0), c.get("noise", 0.0),
                                        workers=workers, log=lambda *_: None)
        print(f"[SCRIPTED] collected {sum(r['success'] for r in res)}/{len(res)} successful demonstrations")
    suite = a.suite or bench["suite"]
    results = []
    if bench.get("include_expert", True):
        p = root / "results" / "scripted_expert.json"
        evaluate(None, suite, str(p), workers=workers, scripted=True)
        results.append(str(p))
    for pol in bench["policies"]:
        cfg = load_config(pol["config"])
        cfg.update({"data": str(data), "out": str(root / "runs" / pol["name"]), "name": pol["name"]})
        ckpt = train(cfg)
        p = root / "results" / f"{pol['name']}.json"
        evaluate(str(ckpt), suite, str(p), workers=workers)
        results.append(str(p))
    compare(results, str(root / "report"), bench.get("title", "Benchmark"))
    print((root / "report" / "comparison.md").read_text())
    return 0


def cmd_study(a) -> int:
    from .study import run_study
    run_study(a.config, _workers(a.workers), a.out)
    return 0


def cmd_demo(a) -> int:
    from .teleop.demo import run_demo
    return run_demo(a)


def cmd_replay(a) -> int:
    from .replay import replay
    return replay(a.episode, gui=a.gui, plot=a.plot)


def cmd_rollout(a) -> int:
    from .replay import autonomous_gui
    return autonomous_gui(a.checkpoint, a.seed, a.episodes, gui=not a.headless)


def cmd_stream(a) -> int:
    from .telemetry.cli import stream
    return stream(a)


def cmd_calibrate(a) -> int:
    from .telemetry.cli import calibrate
    return calibrate(a)


def cmd_dashboard(a) -> int:
    from .dashboard.server import serve
    return serve(a)


def add_source_args(sp: argparse.ArgumentParser) -> None:
    g = sp.add_argument_group("telemetry source")
    g.add_argument("--source", choices=["replay", "synthetic", "stdin", "tcp", "wire-file", "serial"], default="synthetic")
    g.add_argument("--path", help="replay: .json/.jsonl decoded frames; wire-file: binary capture")
    g.add_argument("--rate", type=float, default=None, help="replay/synthetic frame rate, Hz")
    g.add_argument("--loop", action="store_true", help="replay: loop forever")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8765, help="tcp: port to listen on for JSONL")
    g.add_argument("--serial-port", help="serial: device path, e.g. /dev/cu.usbmodem1101")
    g.add_argument("--wire", action="store_true", help="serial: Dermis Wire v1 binary instead of JSON lines")
    g.add_argument("--faults", default="", help="synthetic: comma list, e.g. drop,duplicate,reorder,stale,invalid,missing_pod,config_change,gap")
    g.add_argument("--pods", default="index_tip,thumb_tip,middle_tip,thenar_apb", help="expected pod order")
    g.add_argument("--calibration", help="calibration JSON to apply")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m dermis", description="Dermis hardware-free core")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("collect", help="automated SCRIPTED demonstrations")
    sp.add_argument("--source", default="scripted", choices=["scripted", "keyboard", "glove"])
    sp.add_argument("--episodes", type=int, default=50)
    sp.add_argument("--out", default="data/episodes/scripted")
    sp.add_argument("--seed-start", type=int, default=0)
    sp.add_argument("--noise", type=float, default=0.03, help="execution noise std on velocity, m/s (labels stay clean)")
    sp.add_argument("--task", help="TaskConfig overrides (YAML)")
    sp.add_argument("--workers", type=int)
    sp.add_argument("--gui", action="store_true")
    sp.set_defaults(fn=cmd_collect)

    sp = sub.add_parser("train", help="behavioral cloning")
    sp.add_argument("--config", required=True)
    sp.add_argument("--data")
    sp.add_argument("--out")
    sp.add_argument("--epochs", type=int)
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser("evaluate", help="AUTONOMOUS rollouts on a saved suite")
    sp.add_argument("--checkpoint")
    sp.add_argument("--suite", required=True)
    sp.add_argument("--out")
    sp.add_argument("--scripted", action="store_true", help="run the scripted expert instead (sanity check)")
    sp.add_argument("--workers", type=int)
    sp.add_argument("--gui", action="store_true")
    sp.set_defaults(fn=cmd_evaluate)

    sp = sub.add_parser("make-suite", help="materialize a suite spec into explicit scenarios")
    sp.add_argument("--spec", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_make_suite)

    sp = sub.add_parser("compare", help="comparison table + chart from result JSONs")
    sp.add_argument("results", nargs="+")
    sp.add_argument("--out", default="results/compare")
    sp.add_argument("--title", default="Evaluation comparison")
    sp.set_defaults(fn=cmd_compare)

    sp = sub.add_parser("smoke-test", help="fast end-to-end pipeline check")
    sp.add_argument("--out", default="runs/smoke")
    sp.add_argument("--episodes", type=int, default=4)
    sp.add_argument("--eval-episodes", type=int, default=2)
    sp.set_defaults(fn=cmd_smoke)

    sp = sub.add_parser("benchmark", help="reproducible expert vs BC (no contact) vs BC (contact)")
    sp.add_argument("--config", default="configs/benchmark.yaml")
    sp.add_argument("--suite", help="override the suite in the config")
    sp.add_argument("--out")
    sp.add_argument("--recollect", action="store_true")
    sp.add_argument("--workers", type=int)
    sp.set_defaults(fn=cmd_benchmark)

    sp = sub.add_parser("study", help="data-scaling x seed study: BC with vs without contact")
    sp.add_argument("--config", default="configs/study.yaml")
    sp.add_argument("--out")
    sp.add_argument("--workers", type=int)
    sp.set_defaults(fn=cmd_study)

    sp = sub.add_parser("demo", help="human demonstrations in the GUI (KEYBOARD or GLOVE)")
    sp.add_argument("--demo-source", "--source", dest="demo_source", choices=["keyboard", "glove"], default="keyboard")
    sp.add_argument("--out", default="data/episodes/human")
    sp.add_argument("--seed-start", type=int, default=None)
    sp.add_argument("--headless", action="store_true", help="no window (glove source only; for integration tests)")
    sp.add_argument("--max-episodes", type=int, default=None)
    sp.add_argument("--experimental-glove-teleop", action="store_true",
                    help="enable the provisional tilt->velocity / magnetic-change->grip mapping")
    sp.add_argument("--glove-input", default="synthetic", choices=["replay", "synthetic", "stdin", "tcp", "wire-file", "serial"])
    sp.add_argument("--glove-path")
    sp.add_argument("--glove-port", type=int, default=8765)
    sp.add_argument("--serial-port")
    sp.add_argument("--wire", action="store_true")
    sp.add_argument("--calibration")
    sp.set_defaults(fn=cmd_demo)

    sp = sub.add_parser("replay", help="REPLAY a recorded episode (open-loop actions, for inspection)")
    sp.add_argument("--episode", required=True, help="path to <id>.json or <id>.npz")
    sp.add_argument("--gui", action="store_true")
    sp.add_argument("--plot", help="write an inspection plot PNG here")
    sp.set_defaults(fn=cmd_replay)

    sp = sub.add_parser("rollout", help="watch AUTONOMOUS episodes from a checkpoint")
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--seed", type=int, default=3_000_000)
    sp.add_argument("--episodes", type=int, default=3)
    sp.add_argument("--headless", action="store_true")
    sp.set_defaults(fn=cmd_rollout)

    sp = sub.add_parser("stream", help="run a telemetry source through the ingestion pipeline")
    add_source_args(sp)
    sp.add_argument("--max-frames", type=int)
    sp.add_argument("--record", help="append raw decoded frames (JSONL) here")
    sp.add_argument("--print-every", type=int, default=30)
    sp.add_argument("--json", action="store_true", help="print one normalized JSON object per frame")
    sp.set_defaults(fn=cmd_stream)

    sp = sub.add_parser("calibrate", help="capture an unloaded magnetic baseline")
    add_source_args(sp)
    sp.add_argument("--frames", type=int, default=300)
    sp.add_argument("--out", required=True)
    sp.add_argument("--scale-ut", type=float, default=500.0, help="|delta B| that maps to normalized deformation 1.0")
    sp.set_defaults(fn=cmd_calibrate)

    # TCP teleop path (src/dermis/server/; see docs/tcp_teleop.md)
    from .server.teleop.session import add_args as _teleop_args, cmd_teleop
    from .server.dev_sender import add_args as _sender_args, cmd_send_test
    from .server.teleop.pipeline_check import add_args as _check_args, cmd_test_tcp_pipeline
    from .server.websocket_bridge import (add_bridge_args as _bridge_args, add_mock_args as _mock_ws_args,
                                          cmd_mock_ws_server, cmd_websocket_bridge)
    sp = sub.add_parser("teleop", help="TELEOP test: TCP glove telemetry -> simulated gripper (EXPERIMENTAL mapping)")
    _teleop_args(sp)
    sp.set_defaults(fn=cmd_teleop)
    sp = sub.add_parser("send-test", help="development TCP sender: fixture | replay | SYNTHETIC, with fault injection")
    _sender_args(sp)
    sp.set_defaults(fn=cmd_send_test)
    sp = sub.add_parser("test-tcp-pipeline", help="headless TCP -> simulator integration check")
    _check_args(sp)
    sp.set_defaults(fn=cmd_test_tcp_pipeline)
    sp = sub.add_parser("websocket-bridge", help="subscribe to WebSocket frames and forward to the local TCP receiver")
    _bridge_args(sp)
    sp.set_defaults(fn=cmd_websocket_bridge)
    sp = sub.add_parser("mock-ws-server", help="development WebSocket decoded-frame source")
    _mock_ws_args(sp)
    sp.set_defaults(fn=cmd_mock_ws_server)

    from .cup.cli import add_parser as _add_cup
    _add_cup(sub)
    from .wipe.cli import add_parser as _add_wipe
    _add_wipe(sub)
    from .glove.cli import add_parser as _add_glove
    _add_glove(sub)

    sp = sub.add_parser("dashboard", help="local web dashboard")
    add_source_args(sp)
    sp.add_argument("--http-port", type=int, default=8000)
    sp.add_argument("--data-root", default="data/episodes")
    sp.add_argument("--runs-root", default="runs")
    sp.add_argument("--results-root", default="results")
    sp.add_argument("--recordings", default="data/telemetry")
    sp.add_argument("--cup-controller", default="", help="fragile_cup live panel: comma list of SCRIPTED controllers, e.g. grip_load_expert,fixed_14N")
    sp.add_argument("--cup-checkpoint", action="append", default=[], help="fragile_cup live panel: trained checkpoint dir (repeatable)")
    sp.add_argument("--cup-seed", type=int, default=4_000_000)
    sp.set_defaults(fn=cmd_dashboard)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
