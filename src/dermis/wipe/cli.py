"""`python -m dermis wipe <command>`: wipe_v1 feasibility study (separate from other benchmarks)."""
from __future__ import annotations

from pathlib import Path


def _w(n):
    from ..evaluate import default_workers
    return default_workers() if n is None else n


def cmd_physics(a):
    from .physics_checks import run_all
    return 0 if run_all(a.out)["all_passed"] else 1


def cmd_tune(a):
    from .evaluate import tune
    tune(a.out, _w(a.workers))
    return 0


def cmd_freeze(a):
    from .evaluate import make_eval_suite
    if Path(a.out).exists() and not a.force:
        print(f"{a.out} already exists (frozen). Use a new version name instead of overwriting.")
        return 2
    s = make_eval_suite(a.out, a.count)
    print(f"froze {len(s['scenarios'])} scenarios -> {a.out} (sha256 {s['sha256'][:16]})")
    return 0


def cmd_evaluate(a):
    from .evaluate import evaluate
    evaluate(a.tuning, a.suite, a.out, _w(a.workers))
    print(Path(a.out, "report.md").read_text())
    return 0


def cmd_glove_record(a):
    from .glove_trials import record_session
    return record_session(a)


def cmd_glove_report(a):
    from .glove_trials import report_session
    return report_session(a.session, a.out)


def add_parser(sub) -> None:
    w = sub.add_parser("wipe", help="wipe_v1 contact-wiping feasibility study")
    ws = w.add_subparsers(dest="wipe_cmd", required=True)

    sp = ws.add_parser("physics-check", help="physics validation (run first)")
    sp.add_argument("--out", default="runs/wipe/physics_checks.json")
    sp.set_defaults(fn=cmd_physics)

    sp = ws.add_parser("tune", help="equal-budget tuning of all controllers on the dev suite")
    sp.add_argument("--out", default="runs/wipe")
    sp.add_argument("--workers", type=int)
    sp.set_defaults(fn=cmd_tune)

    sp = ws.add_parser("freeze-suite", help="write the frozen evaluation suite (once per task version)")
    sp.add_argument("--out", default="configs/wipe/eval_v1.json")
    sp.add_argument("--count", type=int, default=40)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(fn=cmd_freeze)

    sp = ws.add_parser("evaluate", help="tuned controllers on the frozen suite: nominal + stress tests")
    sp.add_argument("--tuning", default="runs/wipe/tuning.json")
    sp.add_argument("--suite", default="configs/wipe/eval_v1.json")
    sp.add_argument("--out", default="runs/wipe/eval")
    sp.add_argument("--workers", type=int)
    sp.set_defaults(fn=cmd_evaluate)

    sp = ws.add_parser("glove-record", help="record labelled glove pad trials (real hardware; PENDING)")
    sp.add_argument("--source", default="serial", choices=["serial", "tcp", "stdin", "replay", "synthetic", "wire-file"])
    sp.add_argument("--serial-port")
    sp.add_argument("--wire", action="store_true")
    sp.add_argument("--path")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--out", required=True, help="session directory")
    sp.add_argument("--pod", default="index_tip")
    sp.add_argument("--mount-note", default="", help="free text: pad material/thickness, sensor mount, orientation")
    sp.add_argument("--repeats", type=int, default=3)
    sp.add_argument("--trial-s", type=float, default=5.0)
    sp.add_argument("--baseline-frames", type=int, default=300)
    sp.add_argument("--scale-csv", help="optional reference scale log: host_unix_s,grams per line")
    sp.add_argument("--auto", action="store_true", help="no prompts (software testing with SYNTHETIC input only)")
    sp.set_defaults(fn=cmd_glove_record)

    sp = ws.add_parser("glove-report", help="plots + repeatability numbers for a recorded session")
    sp.add_argument("--session", required=True)
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_glove_report)
