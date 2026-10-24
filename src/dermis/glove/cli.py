"""`python -m dermis glove <command>`: glove bring-up (inspection, labelled trials, grip calibration)."""
from __future__ import annotations

import json


def cmd_inspect(a):
    from .session import run
    return run(a)


def cmd_report(a):
    from .report import report
    r = report(a.session, a.out)
    print(json.dumps({k: v for k, v in r.items() if k != "pipeline"}, indent=1, default=float)[:4000])
    return 0


def add_parser(sub) -> None:
    g = sub.add_parser("glove", help="glove bring-up: inspect signals, labelled trials, two-point grip calibration")
    gs = g.add_subparsers(dest="glove_cmd", required=True)
    sp = gs.add_parser("inspect", help="discover pods, live readout, labelled raw recording, grip calibration")
    sp.add_argument("--source", default="serial", choices=["serial", "tcp", "stdin", "replay", "synthetic", "wire-file"])
    sp.add_argument("--serial-port")
    sp.add_argument("--wire", action="store_true", help="serial: Dermis Wire v1 binary instead of JSON lines")
    sp.add_argument("--port", type=int, default=8765, help="tcp: listen port")
    sp.add_argument("--path", help="replay / wire-file path")
    sp.add_argument("--rate", type=float, help="synthetic/replay rate, Hz")
    sp.add_argument("--faults", default="", help="synthetic only (software testing)")
    sp.add_argument("--synthetic-profile", default="step", choices=["step", "periodic"],
                    help="synthetic only: step = 3 s unloaded / 3 s held press / 3 s released (default)")
    sp.add_argument("--out", required=True, help="session directory")
    sp.add_argument("--mapping", default="configs/glove/grip.yaml")
    sp.add_argument("--trial-s", type=float, default=4.0)
    sp.add_argument("--reference-s", type=float, default=1.5)
    sp.add_argument("--print-every", type=float, default=1.0)
    sp.add_argument("--confirm-hardware", action="store_true",
                    help="operator confirms a TCP/stdin sender is the physical glove (label REAL HARDWARE)")
    sp.add_argument("--script", help="comma list of commands for headless runs, e.g. 0,1,c,u,l,q")
    sp.set_defaults(fn=cmd_inspect)
    sp = gs.add_parser("report", help="plots + numbers for a recorded session")
    sp.add_argument("--session", required=True)
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_report)
