"""`python -m dermis glass <command>`: three-finger glass hold under a water pour (SIMULATED), with SYNTHETIC
glove frames in the dermis-4mlx-1imu-force-provisional-v1 schema."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CKPT = "runs/glass/glass_policy_v1"


def cmd_check_schema(a):
    from .schema import SchemaError, parse
    bad = 0
    for i, line in enumerate(Path(a.file).read_text().splitlines()):
        try:
            parse(json.loads(line))
        except (SchemaError, json.JSONDecodeError) as e:
            bad += 1
            print(f"line {i + 1}: {e}")
    print(f"{a.file}: {i + 1 - bad}/{i + 1} frames valid")
    return 1 if bad else 0


def cmd_collect(a):
    from .data import collect
    print(json.dumps(collect(a.out, a.n, a.seed0, a.noise), indent=1))


def cmd_train(a):
    from .train import train
    m = train(a.data, a.out, history=a.history, epochs=a.epochs, seed=a.seed)
    print(f"saved {a.out}: best val MAE {m['best_val_mae_nm']} N*m ({m['train_s']} s)")


def cmd_evaluate(a):
    from .evaluate import evaluate
    s = evaluate(a.controller, a.suite, a.workers, a.out)
    print(json.dumps({k: v for k, v in s.items() if k != "episodes"}, indent=1))


def cmd_demo(a):
    from .demo import gui_replay, plot_demo, render_video, run_demo
    demo = run_demo(a.checkpoint, a.seed, a.out)
    for l in demo["summary"]["lanes"]:
        print(f"{l['controller']:>22} [{l['label']}]: {l['termination']} at {l['termination_t']} s, "
              f"peak pod normal {max(l['peak_normal_n'].values()):.1f} N")
    plot_demo(demo, str(Path(a.out, "demo_plot.png")))
    if not a.no_video:
        print("video:", render_video(demo, str(Path(a.out, "glass_demo.mp4"))))
    if a.gui:
        gui_replay(demo, a.speed)


def _glove_export(a):
    from .glove_cal import load_export
    if a.source == "synthetic":                                # calibrations describe the real glove only
        return None
    exp = load_export(a.glove_cal)
    print(f"glove calibration: {a.glove_cal} (device {exp['_device_id']})" if exp else
          f"glove calibration: none at {a.glove_cal}; activation uses default noise levels")
    return exp


def cmd_pickup(a):
    from .pickup import PickupConfig
    from .pickup_live import GRIP_MEASURE, ReplaySource, SyntheticSource, WebSocketSource, run
    export = _glove_export(a)
    gain, ut, mode, activation = a.grip_gain, a.ut_per_n, "auto", None
    if (a.source != "synthetic" and a.calibration and Path(a.calibration).exists()   # venue calibration is for the real glove
            and a.grip_gain is None and a.ut_per_n is None):
        cal = json.loads(Path(a.calibration).read_text())
        if cal.get("grip_measure") != GRIP_MEASURE:
            print(f"{a.calibration} was made with an older grip measure; run `python -m dermis glass pickup-calibrate` again.")
            return 2
        gain, ut, mode = cal["grip_gain"], cal["ut_per_n"], ("proxy" if cal["mode"] == "proxy" else "auto")
        activation = cal.get("activation_ut") if "free_motion_max_ut" in cal else None
        if activation is None:
            print("  this calibration has no free-movement step: run `python -m dermis glass pickup-calibrate` again "
                  "so moving the glove is not taken for a grip")
        print(f"using venue calibration {a.calibration} ({cal['mode']}, {cal['created']}, pods {cal.get('pods_registered')})")
    src = {"ws": lambda: WebSocketSource(a.url), "replay": lambda: ReplaySource(a.file),
           "synthetic": lambda: SyntheticSource(a.profile)}[a.source]()
    cfg = PickupConfig(grip_gain=gain or PickupConfig.grip_gain, crack_force_n=None if a.no_crack else 16.0, grip_mode=a.grip_mode,
                       lift_mode=a.lift_mode, imu_acc_dev=a.imu_acc, imu_gyro=a.imu_gyro)
    r = run(src, cfg, gui=not a.headless, record=a.record, ut_per_n=ut or 160.0, duration_s=a.duration,
            grip_mode=mode, export=export, activation_ut=activation)
    print(json.dumps({k: v for k, v in r.items() if k != "events"}), "\n" + "\n".join(json.dumps(e) for e in r["events"]))


def cmd_pickup_calibrate(a):
    from .pickup_live import ReplaySource, SyntheticSource, WebSocketSource, calibrate
    export = _glove_export(a)
    src = {"ws": lambda: WebSocketSource(a.url), "replay": lambda: ReplaySource(a.file),
           "synthetic": lambda: SyntheticSource("normal")}[a.source]()
    try:
        cal = calibrate(src, a.out, export=export)
    finally:
        src.close()
    print(json.dumps(cal, indent=1), f"\nsaved {a.out}")


def cmd_glove_check(a):
    from .pickup_live import WebSocketSource, glove_check
    src = WebSocketSource(a.url)
    try:
        rep = glove_check(src, a.seconds)
    finally:
        src.close()
    print(json.dumps(rep, indent=1))
    return 0 if rep["frames"] else 1


def add_parser(sub) -> None:
    g = sub.add_parser("glass", help="glass_hold: 3-finger hand + thenar pod, water pour (SIMULATED, SYNTHETIC glove frames)")
    gs = g.add_subparsers(dest="glass_cmd", required=True)
    sp = gs.add_parser("check-schema", help="validate a JSONL file of glove frames against the provisional-force schema")
    sp.add_argument("file")
    sp.set_defaults(fn=cmd_check_schema)
    sp = gs.add_parser("collect", help="SCRIPTED load_expert demonstrations -> SYNTHETIC glove frames + features")
    sp.add_argument("--out", default="runs/glass/demos")
    sp.add_argument("--n", type=int, default=200)
    sp.add_argument("--seed0", type=int, default=10_000)
    sp.add_argument("--noise", type=float, default=0.008, help="executed effort noise, N*m (labels stay clean)")
    sp.set_defaults(fn=cmd_collect)
    sp = gs.add_parser("train", help="behavioral cloning on the glove-frame features")
    sp.add_argument("--data", default="runs/glass/demos")
    sp.add_argument("--out", default=DEFAULT_CKPT)
    sp.add_argument("--history", type=int, default=4)
    sp.add_argument("--epochs", type=int, default=60)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(fn=cmd_train)
    sp = gs.add_parser("evaluate", help="dev / held-out suite; --controller scripted:<name> or a checkpoint dir")
    sp.add_argument("--controller", default=DEFAULT_CKPT)
    sp.add_argument("--suite", choices=["dev", "heldout"], default="heldout")
    sp.add_argument("--workers", type=int, default=6)
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_evaluate)
    sp = gs.add_parser("demo", help="three lanes on one scenario: too weak / too strong / network; video + plot (+ --gui)")
    sp.add_argument("--checkpoint", default=DEFAULT_CKPT)
    sp.add_argument("--seed", type=int, default=30_000)
    sp.add_argument("--out", default="runs/glass/demo")
    sp.add_argument("--gui", action="store_true", help="replay the three lanes in a PyBullet window")
    sp.add_argument("--speed", type=float, default=1.0)
    sp.add_argument("--no-video", action="store_true")
    sp.set_defaults(fn=cmd_demo)
    sp = gs.add_parser("pickup", help="REAL-TIME: judge grasps a real cup, the simulated hand picks up a simulated cup")
    sp.add_argument("--source", choices=["ws", "replay", "synthetic"], default="ws")
    sp.add_argument("--url", default="ws://192.168.4.1:81/ws", help="glove Wi-Fi WebSocket")
    sp.add_argument("--file", help="raw glove frames JSONL for --source replay")
    sp.add_argument("--profile", choices=["cycle", "normal", "light", "crush"], default="cycle", help="--source synthetic")
    sp.add_argument("--grip-gain", type=float, help="sim digit effort (N*m) per N of glove grip (default: calibration, else 0.027)")
    sp.add_argument("--ut-per-n", type=float, help="proxy scale when frames carry no force_n (default: calibration, else 160)")
    sp.add_argument("--calibration", default="runs/glass/pickup_calibration.json", help="from `glass pickup-calibrate`")
    sp.add_argument("--glove-cal", default="runs/glass/glove_calibration.json",
                    help="glove calibration export (dermis-calibration v1): pod map, rest field, noise, load axis")
    sp.add_argument("--no-crack", action="store_true", help="disable the simulated crack threshold")
    sp.add_argument("--grip-mode", choices=["combined", "per_finger"], default="combined",
                    help="combined: mean of the two strongest pods drives all fingers (robust to a loose glove)")
    sp.add_argument("--lift-mode", choices=["imu", "auto"], default="imu", help="imu: hand motion starts the lift")
    sp.add_argument("--imu-acc", type=float, default=0.8, help="m/s^2 change of |acc| that counts as motion")
    sp.add_argument("--imu-gyro", type=float, default=0.5, help="rad/s change of gyro that counts as motion")
    sp.add_argument("--record", help="write received frames + every tick to this JSONL")
    sp.add_argument("--headless", action="store_true")
    sp.add_argument("--duration", type=float, help="stop after this many seconds")
    sp.set_defaults(fn=cmd_pickup)
    sp = gs.add_parser("glove-check", help="what the glove Wi-Fi stream is sending: rate, decode errors, pod/force state")
    sp.add_argument("--url", default="ws://192.168.4.1:81/ws")
    sp.add_argument("--seconds", type=float, default=5.0)
    sp.set_defaults(fn=cmd_glove_check)
    sp = gs.add_parser("pickup-calibrate", help="venue calibration: open hand, then a normal hold of the real cup")
    sp.add_argument("--source", choices=["ws", "replay", "synthetic"], default="ws")
    sp.add_argument("--url", default="ws://192.168.4.1:81/ws")
    sp.add_argument("--file")
    sp.add_argument("--out", default="runs/glass/pickup_calibration.json")
    sp.add_argument("--glove-cal", default="runs/glass/glove_calibration.json")
    sp.set_defaults(fn=cmd_pickup_calibrate)
