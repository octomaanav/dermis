"""Wiping controllers. All share the raster x/y planner, velocity limits, 30 Hz rate and starting
conditions; they differ only in how they choose the vertical command and the wrist stiffness k_z.

Observation access is enforced: each controller declares SEES, and ObsFilter passes only those
keys (with the configured stress-test noise). Available keys:
    planner_xy    shared path target at the next tick (time-based, identical for everyone)
    cmd_pos       the controller's own commanded wrist position (always known to a controller)
    meas_pos      measured pad position (proprioception)
    geometry      observed surface height function h_obs(x, y) = true profile + calibration
                  offset (scenario) [+ per-query noise in the obs-noise stress test]
    normal_load   measured pad-surface normal load, N (simulated tactile sensor)
Never available: true geometry offset, pad stiffness, scenario parameters, cell/dwell/coverage
state, success, future path changes beyond the shared planner.

Objective: every controller targets the same load (TARGET_N = centre of the documented band);
non-tactile controllers convert it to a press depth using the *nominal* pad stiffness.
"""
from __future__ import annotations

import math

import numpy as np

from .env import V_XY_MAX, V_Z_MAX

TARGET_N = 4.0
NOMINAL_PAD_K = 1.0e4      # N/m, geometric mean of the pad stiffness range: what a designer would assume


def series(k1, k2):
    return k1 * k2 / (k1 + k2)


class ObsFilter:
    def __init__(self, env, sees: tuple, noise: dict | None = None, seed: int = 0):
        self.env, self.sees, self.noise = env, set(sees), noise or {}
        self.rng = np.random.default_rng([seed, 31])

    def view(self, full: dict) -> dict:
        e, n, r = self.env, self.noise, self.rng
        out = {}
        if "planner_xy" in self.sees:
            out["planner_xy"] = e.planner.xy(full["t"] + e.cfg.dt)
            out["phase"] = e.planner.phase(full["t"])
        if "cmd_pos" in self.sees:
            out["cmd_pos"] = full["cmd_pos"]
        if "meas_pos" in self.sees:
            out["meas_pos"] = full["meas_pos"] + r.normal(0, n.get("pose_m", 0.0), 3)
        if "geometry" in self.sees:
            off, gn = e.scenario.geom_offset, n.get("geometry_m", 0.0)
            out["geometry"] = lambda x, y: float(e.surface_height(x, y)) + off + float(r.normal(0, gn))
        if "normal_load" in self.sees:
            out["normal_load"] = max(0.0, full["normal_load"] + float(r.normal(0, n.get("load_n", 0.0))))
        out["t"] = full["t"]
        return out


class Base:
    name = "base"
    SEES: tuple = ("planner_xy", "cmd_pos")

    def __init__(self, k_z: float):
        self.k_z = float(k_z)

    def reset(self, cfg):
        self.cfg = cfg

    def z_target(self, o: dict) -> float:
        raise NotImplementedError

    def act(self, o: dict) -> np.ndarray:
        c, dt = self.cfg, self.cfg.dt
        cmd = o["cmd_pos"]
        vxy = np.clip((o["planner_xy"] - cmd[:2]) / dt, -V_XY_MAX, V_XY_MAX)
        if o["phase"] in ("retract", "done"):
            vz = V_Z_MAX
        else:
            vz = float(np.clip((self.z_target(o) - cmd[2]) / dt, -V_Z_MAX, V_Z_MAX))
        return np.array([vxy[0], vxy[1], vz])

    def press_depth(self, target=TARGET_N) -> float:
        return target / series(self.k_z, NOMINAL_PAD_K)

    def params(self) -> dict:
        return {"k_z": self.k_z}


class ConstantHeight(Base):
    """A: wipe at a fixed height = nominal (flat, level) surface height + radius - press depth."""
    name = "A_constant_height"
    SEES = ("planner_xy", "cmd_pos")

    def __init__(self, k_z: float, target_n: float = TARGET_N):
        super().__init__(k_z)
        self.target = target_n

    def z_target(self, o):
        return self.cfg.surface_base_z + self.cfg.pad_radius - self.press_depth(self.target)

    def params(self):
        return {"k_z": self.k_z, "target_n": self.target}


class GeometryFollow(Base):
    """B (stiff k) / C (compliant k): follow the observed surface geometry at the measured pad x/y.
    Same code; C differs by being tuned over low (impedance-style) wrist stiffness."""
    SEES = ("planner_xy", "cmd_pos", "meas_pos", "geometry")

    def __init__(self, k_z: float, target_n: float = TARGET_N, name: str = "B_geometry_stiff"):
        super().__init__(k_z)
        self.target, self.name = target_n, name

    def z_target(self, o):
        x, y = o["planner_xy"]
        return o["geometry"](x, y) + self.cfg.pad_radius - self.press_depth(self.target)

    def params(self):
        return {"k_z": self.k_z, "target_n": self.target}


class _LoadRegulator(Base):
    """Geometry feed-forward + PI on a load signal. Integrator only acts while the pad should be
    wiping (or once in contact), clamped against wind-up."""
    def __init__(self, k_z: float, kp: float, ki: float):
        super().__init__(k_z)
        self.kp, self.ki = kp, ki

    def reset(self, cfg):
        super().reset(cfg)
        self.u = 0.0

    def load(self, o) -> float:
        raise NotImplementedError

    def z_target(self, o):
        x, y = o["planner_xy"]
        z_ff = o["geometry"](x, y) + self.cfg.pad_radius - self.press_depth()
        f = self.load(o)
        err = TARGET_N - f
        if o["phase"] == "wipe" or f > 0.2:
            self.u = float(np.clip(self.u + self.ki * err * self.cfg.dt, -0.006, 0.006))
        return z_ff - self.u - self.kp * err

    def params(self):
        return {"k_z": self.k_z, "kp_m_per_n": self.kp, "ki_m_per_ns": self.ki}


class ProprioLoad(_LoadRegulator):
    """C+ (no tactile sensor): estimates load from the wrist spring deflection,
    F_est = k_z * (meas_z - cmd_z). Shows whether proprioception alone can infer contact."""
    name = "Cp_proprio_force"
    SEES = ("planner_xy", "cmd_pos", "meas_pos", "geometry")

    def load(self, o):
        return max(0.0, self.k_z * (o["meas_pos"][2] - o["cmd_pos"][2]))


class ContactPI(_LoadRegulator):
    """D: regulates the measured normal load (simulated tactile) with geometry feed-forward."""
    name = "D_contact_feedback"
    SEES = ("planner_xy", "cmd_pos", "meas_pos", "geometry", "normal_load")

    def load(self, o):
        return o["normal_load"]


def make(spec: dict):
    kind = spec["kind"]
    if kind == "A":
        return ConstantHeight(spec["k_z"], spec.get("target_n", TARGET_N))
    if kind == "B":
        return GeometryFollow(spec["k_z"], spec.get("target_n", TARGET_N), "B_geometry_stiff")
    if kind == "C":
        return GeometryFollow(spec["k_z"], spec.get("target_n", TARGET_N), "C_geometry_compliant")
    if kind == "Cp":
        return ProprioLoad(spec["k_z"], spec["kp"], spec["ki"])
    if kind == "D":
        return ContactPI(spec["k_z"], spec["kp"], spec["ki"])
    raise ValueError(kind)


ACCESS_TABLE = [
    ("A constant height", ConstantHeight.SEES),
    ("B geometry-following, stiff wrist", GeometryFollow.SEES),
    ("C geometry-following, compliant (impedance) wrist", GeometryFollow.SEES),
    ("Cp proprioceptive load estimate (deflection)", ProprioLoad.SEES),
    ("D contact feedback (tactile normal load)", ContactPI.SEES),
]
