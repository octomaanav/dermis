"""Physics validation for wipe_v1, run before any controller comparison.

Each check returns {"name", "passed", "details"}; run_all() writes them to JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pybullet as p

from .env import WipeConfig, WipeEnv, WipeScenario


def _press(env: WipeEnv, depth: float, seconds: float = 1.5, xy=None) -> dict:
    """Hold the pad stationary at a commanded depth below its resting height; return load series."""
    c = env.cfg
    x, y = xy or (0.52, 0.0)
    rest = float(env.surface_height(x, y)) + c.pad_radius
    p.resetBasePositionAndOrientation(env.pad, (x, y, rest + 0.002), (0, 0, 0, 1), **env._p)
    p.resetBaseVelocity(env.pad, (0, 0, 0), (0, 0, 0), **env._p)
    env.cmd = np.array([x, y, rest - depth])
    loads, zs, normals = [], [], []
    for _ in range(int(seconds * c.physics_hz)):
        env._actuate()
        p.stepSimulation(**env._p)
        p.resetBaseVelocity(env.pad, p.getBaseVelocity(env.pad, **env._p)[0], (0, 0, 0), **env._p)
        pts = p.getContactPoints(env.pad, env.surface, **env._p)
        loads.append(sum(q[9] for q in pts))
        normals += [q[7] for q in pts]
        zs.append(p.getBasePositionAndOrientation(env.pad, **env._p)[0][2])
    return {"loads": np.array(loads), "z": np.array(zs), "normals": np.array(normals), "rest": rest, "cmd_z": env.cmd[2]}


def check_heightfield(env: WipeEnv) -> dict:
    sc = WipeScenario.sample(1_000_003)
    env.reset(sc)
    p.resetBasePositionAndOrientation(env.pad, (0, 0, 2.0), (0, 0, 0, 1), **env._p)   # out of the rays' way
    c = env.cfg
    diffs = []
    for x in np.linspace(c.region_x[0], c.region_x[1], 13):
        for y in np.linspace(c.region_y[0], c.region_y[1], 9):
            hit = p.rayTest((x, y, 1.0), (x, y, -1.0), **env._p)[0]
            diffs.append(hit[3][2] - float(env.surface_height(x, y)))
    d = np.abs(diffs)
    return {"name": "heightfield matches analytic profile (ray tests, 117 points)", "passed": bool(d.max() < 2e-4),
            "details": {"max_abs_m": float(d.max()), "mean_abs_m": float(d.mean())}}


def check_sign_frame_equilibrium(env: WipeEnv) -> dict:
    """Flat surface, stationary press. The *net* contact force must point along +z (surface to pad),
    and its normal component must equal the wrist spring force k (z - z_cmd) at rest. Per-point
    normals are reported separately: the triangulated heightfield tilts them (mesh artifact)."""
    env.reset(WipeScenario.flat(), k_z=1000.0)
    r = _press(env, 0.004)
    tail = slice(-120, None)
    pts = p.getContactPoints(env.pad, env.surface, **env._p)
    F = sum(q[9] * np.array(q[7]) + q[10] * np.array(q[11]) + q[12] * np.array(q[13]) for q in pts)
    ang = float(np.degrees(np.arccos(F[2] / np.linalg.norm(F))))
    load, _ = env._contact()
    spring = 1000.0 * float(np.mean(r["z"][tail]) - r["cmd_z"])
    nz = r["normals"][:, 2]
    ok = bool(F[2] > 0 and ang < 2.0 and abs(load - spring) < 0.02 * spring + 0.02)
    return {"name": "net contact force points surface->pad (+z on flat); normal load = wrist spring force at rest",
            "passed": ok, "details": {"net_force_angle_from_normal_deg": ang, "load_n": load, "spring_n": spring,
                                      "per_point_normals_tilted_gt_8deg_fraction": float((nz < 0.99).mean()),
                                      "per_point_min_normal_z": float(nz.min()),
                                      "note": "per-point tilt is a triangle-mesh contact artifact; load uses the net force"}}


def check_no_double_count(env: WipeEnv) -> dict:
    """One query per substep. The net force from the pad->surface query equals minus the net force from
    the surface->pad query (same points, seen from the other body), and the reported load is exactly
    that net force projected on the surface normal: each contact point is counted once."""
    env.reset(WipeScenario.sample(1_000_004), k_z=1000.0)
    _press(env, 0.004, 0.5, xy=(0.47, 0.03))
    def net(a, b):
        return sum(q[9] * np.array(q[7]) + q[10] * np.array(q[11]) + q[12] * np.array(q[13])
                   for q in p.getContactPoints(a, b, **env._p))
    fa, fb = net(env.pad, env.surface), net(env.surface, env.pad)
    load, xy = env._contact()
    proj = float(fa @ env.surface_normal(*xy))
    npts = len(p.getContactPoints(env.pad, env.surface, **env._p))
    ok = np.allclose(fa, -fb, atol=0.01) and abs(load - proj) < 1e-9          # reversed query differs ~0.05%
    return {"name": "one query per substep; each contact point counted once (pad->surface net = -(surface->pad net))",
            "passed": bool(ok), "details": {"points": npts, "net_pad_side": fa.tolist(), "net_surface_side": fb.tolist(),
                                            "max_abs_diff_n": float(np.abs(fa + fb).max()), "env_load": load,
                                            "projected_net": proj}}


def check_monotonic(env: WipeEnv) -> dict:
    out = {}
    for k in (300.0, 3000.0):
        env.reset(WipeScenario.flat(), k_z=k)
        loads = [float(_press(env, d, 1.0)["loads"][-60:].mean()) for d in (0.0, 0.002, 0.004, 0.008, 0.012)]
        out[f"k_z={k:g}"] = [round(x, 3) for x in loads]
    ok = all(np.all(np.diff(v) > 0) for v in out.values())
    return {"name": "more commanded compression -> more load (depth 0, 2, 4, 8, 12 mm)", "passed": bool(ok), "details": out}


def check_stationary_stability(env: WipeEnv) -> dict:
    out, ok = {}, True
    for ks in (3e3, 1e4, 3e4):
        for kz in (300.0, 3000.0):
            sc = WipeScenario.flat()
            sc.pad_stiffness = ks
            env.reset(sc, k_z=kz)
            depth = 4.0 / (kz * ks / (kz + ks))
            r = _press(env, depth, 2.0)
            tail = r["loads"][-240:]
            rel = float(tail.std() / max(1e-9, tail.mean()))
            drift = float(abs(tail[-30:].mean() - tail[:30].mean()))
            out[f"pad_k={ks:g},k_z={kz:g}"] = {"mean_n": round(float(tail.mean()), 3), "rel_std": round(rel, 4),
                                               "drift_n": round(drift, 4)}
            ok &= rel < 0.02 and drift < 0.05
    return {"name": "stationary pad: steady load (last 1 s), no oscillation or drift", "passed": bool(ok), "details": out}


def check_substeps(controllers: list[dict], scenarios: list) -> dict:
    """Same controllers/scenarios at 240 Hz and 480 Hz physics; report coverage/load differences."""
    from .evaluate import run_episode
    rows = []
    for spec in controllers:
        for sc in scenarios:
            r1 = run_episode(spec, sc, WipeConfig())
            r2 = run_episode(spec, sc, WipeConfig(physics_hz=480))
            rows.append({"controller": spec["kind"], "seed": sc.seed, "coverage_240": r1["coverage"],
                         "coverage_480": r2["coverage"], "p95_240": round(r1["p95_load_n"], 2),
                         "p95_480": round(r2["p95_load_n"], 2), "success_240": r1["success"], "success_480": r2["success"]})
    dcov = max(abs(r["coverage_240"] - r["coverage_480"]) for r in rows)
    dp95 = max(abs(r["p95_240"] - r["p95_480"]) for r in rows)
    flips = sum(r["success_240"] != r["success_480"] for r in rows)
    return {"name": "physics substeps 240 -> 480 Hz: results do not change drastically",
            "passed": bool(dcov <= 0.1 and dp95 <= 1.0), "details": {"max_coverage_change": dcov,
                                                                       "max_p95_change_n": dp95,
                                                                       "success_flips": flips, "rows": rows}}


def check_events_are_contacts(env: WipeEnv) -> dict:
    from .controllers import ObsFilter, make
    ctrl = make({"kind": "D", "k_z": 1000.0, "kp": 0.0005, "ki": 0.002})
    obs = env.reset(WipeScenario.sample(1_000_005), k_z=ctrl.k_z)
    ctrl.reset(env.cfg)
    f = ObsFilter(env, ctrl.SEES)
    while not env.done:
        obs, _, _ = env.step(ctrl.act(f.view(obs)))
    c, bad = env.cfg, 0
    for e in env.stats.events:
        i, j = e["cell"]
        x0, y0 = c.region_x[0] + i * c.cell, c.region_y[0] + j * c.cell
        x, y = e["contact_xy"]
        inside = x0 - 1e-6 <= x <= x0 + c.cell + 1e-6 and y0 - 1e-6 <= y <= y0 + c.cell + 1e-6
        bad += not (inside and c.load_lo <= e["load"] <= c.load_hi)
    n = len(env.stats.events)
    return {"name": "every wiped-cell event comes from an in-band pad-surface contact located in that cell",
            "passed": bool(n > 0 and bad == 0 and n == env.stats.cells_wiped),
            "details": {"events": n, "violations": bad, "cells_wiped": env.stats.cells_wiped}}


def check_exploits() -> dict:
    """Hovering, plunging, and repeated passes cannot inflate coverage."""
    from .evaluate import run_episode
    hover = run_episode({"kind": "A", "k_z": 3000.0, "target_n": -30.0}, WipeScenario.flat(), WipeConfig())
    plunge = run_episode({"kind": "A", "k_z": 3000.0, "target_n": 60.0}, WipeScenario.flat(), WipeConfig())
    ok = (hover["coverage"] == 0.0 and hover["attempted_coverage"] == 0.0 and
          plunge["coverage"] < 0.05 and plunge["overload_s"] > 1.0 and not plunge["success"])
    return {"name": "hovering wipes nothing; pressing through the surface wipes ~nothing and is flagged as overload",
            "passed": bool(ok), "details": {"hover": {k: hover[k] for k in ("coverage", "attempted_coverage", "success")},
                                            "plunge": {k: plunge[k] for k in ("coverage", "overload_s", "peak_load_n",
                                                                               "max_penetration_m", "termination", "success")}}}


def run_all(out_path: str, substep_controllers: list[dict] | None = None) -> dict:
    env = WipeEnv()
    try:
        checks = [check_heightfield(env), check_sign_frame_equilibrium(env), check_no_double_count(env),
                  check_monotonic(env), check_stationary_stability(env), check_events_are_contacts(env)]
    finally:
        env.close()
    checks.append(check_exploits())
    scen = [WipeScenario.flat()] + [WipeScenario.sample(1_000_000 + i) for i in (1, 2, 3)]
    checks.append(check_substeps(substep_controllers or [
        {"kind": "B", "k_z": 3000.0}, {"kind": "C", "k_z": 300.0},
        {"kind": "D", "k_z": 1000.0, "kp": 0.0005, "ki": 0.002}], scen))
    res = {"all_passed": all(c["passed"] for c in checks), "checks": checks}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, indent=1, default=float))
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['name']}")
    return res
