"""Three-lane glass demo on ONE identical scenario (same glass, same water pour):
  1  too little force  (SCRIPTED too_weak)      -> the glass slides out and falls as water is added
  2  too much force    (SCRIPTED over_squeeze)  -> a pod passes the SIMULATED crack threshold
  3  neural network    (AUTONOMOUS checkpoint)  -> reads only the SYNTHETIC glove frames; holds the glass

Each lane is simulated headless first (separate GlassEnv, identical physics), recording per-tick state and
the synthetic glove frames; after a lane ends its last effort is held and physics keeps running so a
dropped glass is seen falling. The video and the GUI then REPLAY those recorded states; nothing is
re-simulated or altered for display.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .controllers import scripted
from .env import GlassEnv, GlassScenario
from .hand3 import PODS
from .schema import FORCE_MAX_N

LANES = (("too_weak", "1 · TOO LITTLE FORCE", "SCRIPTED fixed 0.02 N·m"),
         ("over_squeeze", "2 · TOO MUCH FORCE", "SCRIPTED: squeezes harder per gram poured"),
         (None, "3 · NEURAL NETWORK", "AUTONOMOUS: input = SYNTHETIC glove frames only"))
AFTER_S = 1.5
POD_SHORT = {"thumb_tip": "thumb", "index_tip": "index", "middle_tip": "middle", "thenar_apb": "thenar_apb"}


def simulate_lane(ctrl, sc: GlassScenario, total_s: float, out_dir: Path, tag: str) -> dict:
    env = GlassEnv()
    try:
        env.reset(sc)
        ctrl.reset(env)
        rows, frames = [], [env.baseline_frame, env.frame]
        end_effort = None
        while env.t < total_s - 1e-9:
            e = ctrl.act(env) if env.termination is None else end_effort
            env.step(e)
            if env.termination is not None and end_effort is None:
                end_effort = env.effort
            st = env.state()
            st["pod_force_n"] = {k: env.frame["pods"][k]["mag"]["force_n"] for k in PODS}
            st["pod_flags"] = {k: env.frame["pods"][k]["mag"]["flags"] for k in PODS}
            st["after_end"] = env.termination is not None and env.events[-1]["t"] < env.t - 1e-9
            rows.append(st)
            frames.append(env.frame)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{tag}.frames.jsonl", "w") as fh:
            for f in frames:
                fh.write(json.dumps(f, separators=(",", ":")) + "\n")
        res = {"controller": ctrl.name, "label": ctrl.label, "termination": env.termination,
               "termination_t": env.events[-1]["t"] if env.events else None,
               "peak_normal_n": {k: round(v, 2) for k, v in env.peak_normal.items()}, "end_t": env.end_t}
        with open(out_dir / f"{tag}.trace.jsonl", "w") as fh:
            fh.write(json.dumps({"header": True, "scenario": sc.to_dict(), "task_config": env.cfg.to_dict(), **res}) + "\n")
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return {"rows": rows, **res}
    finally:
        env.close()


def run_demo(checkpoint: str, seed: int, out_dir: str) -> dict:
    from .controllers import PolicyController
    from .train import GlassPolicy
    sc = GlassScenario.sample(seed)
    probe = GlassEnv()
    probe.sc = sc
    total = probe.end_t + AFTER_S
    probe.close()
    out = Path(out_dir)
    lanes = []
    for name, title, sub in LANES:
        ctrl = scripted(name) if name else PolicyController(GlassPolicy.load(checkpoint))
        tag = name or "network"
        lanes.append({"title": title, "sub": sub, "tag": tag, **simulate_lane(ctrl, sc, total, out, tag)})
    summary = {"seed": seed, "scenario": sc.to_dict(), "checkpoint": checkpoint,
               "lanes": [{k: v for k, v in l.items() if k != "rows"} for l in lanes]}
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=1))
    return {"summary": summary, "lanes": lanes, "scenario": sc}


# -- rendering -------------------------------------------------------------------------------------------
def _status(row: dict) -> tuple[str, tuple]:
    t = row["termination"]
    if t == "drop":
        return "DROPPED", (200, 40, 40)
    if t == "crack":
        return "CRACKED (simulated threshold)", (200, 40, 40)
    if t == "success":
        return "HELD - pour finished", (20, 140, 60)
    return ("HOLDING" if row["drop_cm"] < 0.5 else "SLIPPING"), ((20, 140, 60) if row["drop_cm"] < 0.5 else (210, 130, 0))


def _panel(draw, x0, y0, w, lane, row, crack_n, fonts):
    from .env import GlassConfig
    big, mid, small = fonts
    draw.text((x0 + 12, y0 + 8), lane["title"], fill=(20, 20, 20), font=big)
    draw.text((x0 + 12, y0 + 38), lane["sub"], fill=(90, 90, 90), font=small)
    s, col = _status(row)
    draw.text((x0 + 12, y0 + 60), s, fill=col, font=mid)
    draw.text((x0 + 12, y0 + 88), f"water added {row['water_kg'] * 1000:4.0f} g    glass+water {row['weight_n']:.2f} N    "
              f"grip {row['effort_nm']:.3f} N·m", fill=(40, 40, 40), font=small)
    by, bw, scale = y0 + 114, w - 150, (w - 150) / 20.0
    for i, k in enumerate(PODS):
        y = by + i * 24
        true_n = row["contact"][k]["normal"]
        draw.text((x0 + 12, y + 2), POD_SHORT[k], fill=(40, 40, 40), font=small)
        bx = x0 + 100
        draw.rectangle([bx, y + 4, bx + bw, y + 18], outline=(200, 200, 200))
        fill = (200, 40, 40) if true_n > crack_n else (59, 130, 246)
        draw.rectangle([bx, y + 4, bx + min(bw, true_n * scale), y + 18], fill=fill)
        cx = bx + crack_n * scale
        draw.line([cx, y + 1, cx, y + 21], fill=(200, 40, 40), width=2)
        clamp = "CLAMPED_HIGH" in " ".join(row["pod_flags"][k])
        draw.text((bx + bw + 6, y + 2), f"{row['pod_force_n'][k]:4.1f}{'*' if clamp else ' '}", fill=(40, 40, 40), font=small)
    draw.text((x0 + 12, by + 4 * 24 + 2), f"bar: simulated contact N   red line: {crack_n:.0f} N crack threshold   "
              f"number: glove force_n", fill=(110, 110, 110), font=small)


def render_video(demo: dict, out_mp4: str, w: int = 480, h: int = 380) -> str:
    import pybullet as p
    import pybullet_data
    from PIL import Image, ImageDraw, ImageFont
    from .env import GlassConfig
    from .scene import Lane
    lanes, sc = demo["lanes"], demo["scenario"]
    cfg = GlassConfig()
    d = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=d)
    p.loadURDF("plane.urdf", physicsClientId=d)
    gy = lanes[0]["rows"][0]["glass_pos"][1]
    offs = [-0.6, 0.0, 0.6]
    views = [Lane(d, o, sc.radius, cfg.glass_height, gy) for o in offs]
    proj = p.computeProjectionMatrixFOV(42, w / h, 0.02, 3.0)
    cams = [p.computeViewMatrixFromYawPitchRoll((o, 0.05, 0.15), 0.40, yaw=160, pitch=-16, roll=0, upAxisIndex=2) for o in offs]
    try:
        font = lambda s: ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", s)
        fonts = (font(22), font(19), font(13))
    except OSError:
        fonts = (ImageFont.load_default(),) * 3
    panel_h, foot_h = 250, 46
    W, H = w * 3, h + panel_h + foot_h
    n = min(len(l["rows"]) for l in lanes)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", "30", "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "22", out_mp4],
                            stdin=subprocess.PIPE)
    try:
        for i in range(n):
            canvas = Image.new("RGB", (W, H), (250, 250, 250))
            draw = ImageDraw.Draw(canvas)
            for k, (lane, view, cam) in enumerate(zip(lanes, views, cams)):
                row = lane["rows"][i]
                view.sync(row)
                img = p.getCameraImage(w, h, cam, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=d)[2]
                canvas.paste(Image.fromarray(np.asarray(img, np.uint8).reshape(h, w, 4)[:, :, :3]), (k * w, 0))
                _panel(draw, k * w, h, w, lane, row, cfg.crack_force_n, fonts)
                draw.line([k * w, 0, k * w, h + panel_h], fill=(210, 210, 210), width=2)
            t = lanes[0]["rows"][i]["t"]
            draw.text((12, h + panel_h + 4), f"t = {t:4.1f} s   SIMULATION (PyBullet). Water = mass added to the glass (no fluid). "
                      f"Crack = simulated per-pod force threshold ({cfg.crack_force_n:.0f} N); the glass is rigid and is not shown breaking.",
                      fill=(60, 60, 60), font=fonts[2])
            draw.text((12, h + panel_h + 24), "Glove readings are SYNTHETIC frames (schema dermis-4mlx-1imu-force-provisional-v1) "
                      "made from simulator contact; force_n is provisional and clamps at 9.81 N (*). Same glass and pour in all three lanes.",
                      fill=(60, 60, 60), font=fonts[2])
            proc.stdin.write(canvas.tobytes())
            if i in (int(n * 0.15), int(n * 0.45), n - 1):
                canvas.save(str(Path(out_mp4).with_name(f"{Path(out_mp4).stem}_t{t:04.1f}.png")))
    finally:
        proc.stdin.close()
        proc.wait()
        p.disconnect(d)
    return out_mp4


def plot_demo(demo: dict, png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .env import GlassConfig
    cfg = GlassConfig()
    lanes = demo["lanes"]
    fig, ax = plt.subplots(4, 3, figsize=(14, 9), sharex=True, sharey="row")
    for c, lane in enumerate(lanes):
        rows = [r for r in lane["rows"]]
        t = [r["t"] for r in rows]
        ax[0, c].set_title(f"{lane['title']}\n{lane['sub']}\noutcome: {lane['termination']} at {lane['termination_t']} s", fontsize=9)
        ax[0, c].plot(t, [r["water_kg"] * 1000 for r in rows], color="#3b82f6")
        ax[1, c].plot(t, [r["effort_nm"] for r in rows], color="k")
        for k in PODS:
            ax[2, c].plot(t, [r["contact"][k]["normal"] for r in rows], label=f"{k} (simulated)")
        ax[2, c].axhline(cfg.crack_force_n, color="r", lw=0.8, label="crack threshold (simulated)")
        ax[2, c].axhline(FORCE_MAX_N, color="gray", lw=0.8, ls="--", label="glove force_n clamp")
        ax[3, c].plot(t, [r["drop_cm"] for r in rows], color="#f59e0b")
        ax[3, c].axhline(cfg.drop_m * 100, color="r", lw=0.8)
        ax[3, c].set_xlabel("time (s)")
        if lane["termination_t"]:
            for a in ax[:, c]:
                a.axvline(lane["termination_t"], color="r", ls=":", lw=0.8)
    ax[0, 0].set_ylabel("water added (g)")
    ax[1, 0].set_ylabel("grip effort (N·m)")
    ax[2, 0].set_ylabel("pod contact normal (N)")
    ax[2, 2].legend(fontsize=7, loc="upper left")
    ax[3, 0].set_ylabel("glass drop (cm)")
    ax[3, 0].set_ylim(-0.5, 6)
    fig.suptitle(f"glass_hold SIMULATION, seed {demo['summary']['seed']}: identical glass and water pour in each column", fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=105)
    plt.close(fig)


def gui_replay(demo: dict, speed: float = 1.0, loop: bool = True) -> None:
    """Replay the recorded lanes side by side in a PyBullet GUI window (display only)."""
    import time

    import pybullet as p
    import pybullet_data
    from .env import GlassConfig
    from .scene import Lane
    lanes, sc, cfg = demo["lanes"], demo["scenario"], GlassConfig()
    d = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=d)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=d)
    p.loadURDF("plane.urdf", physicsClientId=d)
    offs = [-0.3, 0.0, 0.3]
    gy = lanes[0]["rows"][0]["glass_pos"][1]
    views = [Lane(d, o, sc.radius, cfg.glass_height, gy) for o in offs]
    p.resetDebugVisualizerCamera(0.75, 180, -15, (0, 0.05, 0.16), physicsClientId=d)
    for o, lane in zip(offs, lanes):
        p.addUserDebugText(lane["title"], (o - 0.1, 0.05, 0.33), (0.1, 0.1, 0.1), 1.2, physicsClientId=d)
        p.addUserDebugText(lane["sub"], (o - 0.12, 0.05, 0.305), (0.35, 0.35, 0.35), 0.8, physicsClientId=d)
    p.addUserDebugText("SIMULATION REPLAY - water = added mass, crack = simulated threshold, glove data SYNTHETIC",
                       (-0.42, 0.05, 0.02), (0.3, 0.3, 0.3), 0.9, physicsClientId=d)
    n = min(len(l["rows"]) for l in lanes)
    status_ids = [None] * 3
    shown = [None] * 3
    try:
        while p.isConnected(d):
            t0 = time.time()
            for i in range(n):
                for k, (lane, view) in enumerate(zip(lanes, views)):
                    row = lane["rows"][i]
                    view.sync(row)
                    s, col = _status(row)
                    if s != shown[k]:                           # GUI text redraws are slow on macOS: only on change
                        kw = {"replaceItemUniqueId": status_ids[k]} if status_ids[k] is not None else {}
                        status_ids[k] = p.addUserDebugText(s, (offs[k] - 0.08, 0.05, 0.28), [c / 255 for c in col], 1.3,
                                                           physicsClientId=d, **kw)
                        shown[k] = s
                wait = t0 + (i + 1) / (30 * speed) - time.time()
                if wait > 0:
                    time.sleep(wait)
            time.sleep(1.5)
            if not loop:
                break
    finally:
        if p.isConnected(d):
            p.disconnect(d)
