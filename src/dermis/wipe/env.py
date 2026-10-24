"""wipe_v1: contact-sensitive wiping feasibility benchmark (headless PyBullet).

SYNTHETIC MANIPULATION BENCHMARK. Not validated cleaning, tissue or material physics. The load
band, overload threshold and dwell rule are benchmark definitions, not measured requirements.

Units: m, s, N, kg. World z up. 30 Hz control, 240 Hz physics (8 substeps per tick) unless
configured otherwise; contact quantities are evaluated every substep.

Bodies
    surface   static PyBullet heightfield over a 0.30 x 0.22 m patch (5 mm grid). Profile =
              base height + tilt + sinusoidal bumps (scenario). PyBullet centres heightfields
              between their min and max, which surface_height() accounts for (verified by ray
              tests in physics_checks.py).
    pad       a 0.1 kg sphere, radius 2 cm, with *soft contact* (PyBullet contactStiffness /
              contactDamping; scenario-varied: the pad material's compliance). Rotation is held
              at zero, so it slides (wipes) rather than rolls; friction 0.3 against the surface.

Actuation (the "wrist"): every substep a spring-damper force pulls the pad toward the commanded
position: stiff in x/y (4000 N/m), vertical stiffness k_z chosen by the controller within
[K_MIN, K_MAX] (impedance), plus gravity feed-forward. The commanded position integrates the
commanded velocity, bounded per axis (|vx|,|vy| <= 0.12 m/s, |vz| <= 0.05 m/s).
Commanded pose != measured pose: the pad lags/deflects; both are reported separately.

Contamination grid: 12 x 8 cells of 2 cm over the wiping region. A cell becomes WIPED when the
pad-surface contact point lies in it while the total normal load is within [load_lo, load_hi]
during the wiping phase, accumulated for >= dwell_s. Each cell counts once.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pybullet as p

K_MIN, K_MAX = 100.0, 4000.0
V_XY_MAX, V_Z_MAX = 0.12, 0.05


@dataclass
class WipeConfig:
    control_hz: int = 30
    physics_hz: int = 240
    pad_radius: float = 0.02
    pad_mass: float = 0.1
    pad_friction: float = 0.3
    k_xy: float = 4000.0
    surface_base_z: float = 0.05
    surface_center: tuple = (0.52, 0.0)
    surface_size: tuple = (0.30, 0.22)
    surface_res: float = 0.005
    region_x: tuple = (0.40, 0.64)            # wiping region = grid extent
    region_y: tuple = (-0.08, 0.08)
    cell: float = 0.02
    load_lo: float = 2.0                      # N, in-range band (benchmark definition)
    load_hi: float = 6.0
    overload: float = 10.0                    # N, excessive load (benchmark definition)
    contact_min: float = 0.2                  # N, below this the pad counts as not in contact
    dwell_s: float = 0.15                     # in-range contact time needed inside one cell
    max_penetration: float = 0.008            # m beyond the pad's resting depth -> flagged
    wipe_speed: float = 0.08                  # m/s along lanes (shared path planner)
    approach_s: float = 2.0
    retract_s: float = 0.6
    success_coverage: float = 0.90
    success_overload_s: float = 0.25

    @property
    def substeps(self) -> int:
        return self.physics_hz // self.control_hz

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "WipeConfig":
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in (d or {}).items()})


@dataclass
class WipeScenario:
    """Physically motivated variation. Ranges and reasons in docs/wipe_feasibility.md."""
    seed: int
    tilt_x_deg: float = 0.0          # surface not level (furniture, fixturing): +-3 deg
    tilt_y_deg: float = 0.0
    bump_amp: float = 0.0            # m, smooth undulation: 0..5 mm
    bump_wavelength: float = 0.10    # m: 6..15 cm
    bump_dir_deg: float = 0.0
    bump_phase: float = 0.0
    pad_stiffness: float = 1.0e4     # N/m contact stiffness (pad material): 3e3..3e4
    geom_offset: float = 0.0         # m, constant error of the observed geometry (calibration): +-2 mm
    start_height: float = 0.04       # m, initial pad clearance above the surface: 3..6 cm

    @classmethod
    def sample(cls, seed: int) -> "WipeScenario":
        r = np.random.default_rng([seed, 777])
        return cls(int(seed), float(r.uniform(-3, 3)), float(r.uniform(-3, 3)), float(r.uniform(0, 0.005)),
                   float(r.uniform(0.06, 0.15)), float(r.uniform(0, 180)), float(r.uniform(0, 2 * math.pi)),
                   float(math.exp(r.uniform(math.log(3e3), math.log(3e4)))), float(r.uniform(-0.002, 0.002)),
                   float(r.uniform(0.03, 0.06)))

    @classmethod
    def flat(cls, seed: int = 0) -> "WipeScenario":
        return cls(int(seed))                                  # sanity case: level, no bumps, nominal pad

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WipeScenario":
        return cls(**d)

    def profile(self, x, y, cfg: WipeConfig):
        """Surface height (analytic) relative to base, at world x, y."""
        cx, cy = cfg.surface_center
        dx, dy = np.asarray(x) - cx, np.asarray(y) - cy
        tilt = math.tan(math.radians(self.tilt_x_deg)) * dx + math.tan(math.radians(self.tilt_y_deg)) * dy
        a = math.radians(self.bump_dir_deg)
        s = dx * math.cos(a) + dy * math.sin(a)
        return tilt + self.bump_amp * np.sin(2 * math.pi * s / self.bump_wavelength + self.bump_phase)


@dataclass
class WipeStats:
    ticks: int = 0
    termination: str | None = None             # completed | workspace_escape | excess_penetration
    wipe_time_s: float = 0.0
    approach_time_s: float = 0.0
    retract_time_s: float = 0.0
    contact_loss_s: float = 0.0                 # wiping phase, normal < contact_min
    overload_s: float = 0.0                     # any phase, normal > overload
    overload_events: int = 0
    peak_load_n: float = 0.0
    p95_load_n: float = 0.0                     # over wiping-phase substeps in contact
    penetration_s: float = 0.0
    max_penetration_m: float = 0.0
    cells_total: int = 0
    cells_wiped: int = 0                        # in range, dwell met
    cells_touched: int = 0                      # any contact during wiping ("attempted")
    coverage: float = 0.0
    attempted_coverage: float = 0.0
    success: bool = False
    events: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("events")
        return d


class WipeEnv:
    def __init__(self, cfg: WipeConfig | None = None):
        self.cfg = c = cfg or WipeConfig()
        self.client = p.connect(p.DIRECT)
        self._p = dict(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, **self._p)
        p.setTimeStep(1.0 / c.physics_hz, **self._p)
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=c.pad_radius, **self._p)
        self.pad = p.createMultiBody(c.pad_mass, col, -1, (0.5, 0, 0.2), **self._p)
        self.surface = None
        nx = int(round((c.region_x[1] - c.region_x[0]) / c.cell))
        ny = int(round((c.region_y[1] - c.region_y[0]) / c.cell))
        self.grid_shape = (nx, ny)
        self.scenario: WipeScenario | None = None
        self.done = True

    # -- geometry ---------------------------------------------------------------------

    def _build_surface(self, sc: WipeScenario) -> None:
        c = self.cfg
        if self.surface is not None:
            p.removeBody(self.surface, **self._p)
        nx = int(round(c.surface_size[0] / c.surface_res)) + 1
        ny = int(round(c.surface_size[1] / c.surface_res)) + 1
        xs = c.surface_center[0] + (np.arange(nx) - (nx - 1) / 2) * c.surface_res
        ys = c.surface_center[1] + (np.arange(ny) - (ny - 1) / 2) * c.surface_res
        X, Y = np.meshgrid(xs, ys)
        H = sc.profile(X, Y, c)
        self._h_center = (H.max() + H.min()) / 2           # PyBullet centres the heightfield
        shape = p.createCollisionShape(p.GEOM_HEIGHTFIELD, meshScale=[c.surface_res, c.surface_res, 1],
                                       heightfieldData=H.flatten().tolist(), numHeightfieldRows=nx,
                                       numHeightfieldColumns=ny, **self._p)
        self.surface = p.createMultiBody(0, shape, -1, (*c.surface_center, c.surface_base_z + self._h_center), **self._p)
        p.changeDynamics(self.surface, -1, lateralFriction=1.0, **self._p)

    def surface_height(self, x, y) -> np.ndarray:
        """True surface z at (x, y) (analytic; matches the heightfield to < 0.1 mm, see physics checks)."""
        return self.cfg.surface_base_z + self.scenario.profile(x, y, self.cfg)

    # -- episode -------------------------------------------------------------------------

    def reset(self, scenario: WipeScenario | int, k_z: float = 1000.0, delay_ticks: int = 0) -> dict:
        sc = scenario if isinstance(scenario, WipeScenario) else WipeScenario.sample(int(scenario))
        self.scenario = sc
        c = self.cfg
        self._build_surface(sc)
        p.changeDynamics(self.pad, -1, lateralFriction=c.pad_friction / 1.0, contactStiffness=sc.pad_stiffness,
                         contactDamping=2 * math.sqrt(sc.pad_stiffness * c.pad_mass), linearDamping=0.0,
                         angularDamping=0.0, **self._p)
        x0, y0 = c.region_x[0] + c.cell / 2, c.region_y[0] + c.cell / 2
        z0 = float(self.surface_height(x0, y0)) + c.pad_radius + sc.start_height
        p.resetBasePositionAndOrientation(self.pad, (x0, y0, z0), (0, 0, 0, 1), **self._p)
        p.resetBaseVelocity(self.pad, (0, 0, 0), (0, 0, 0), **self._p)
        self.cmd = np.array([x0, y0, z0])
        self.k_z = float(np.clip(k_z, K_MIN, K_MAX))
        self.delay = int(delay_ticks)
        self._queue = []
        nx, ny = self.grid_shape
        self.dwell = np.zeros((nx, ny))
        self.touched = np.zeros((nx, ny), bool)
        self.wiped = np.zeros((nx, ny), bool)
        self.stats = WipeStats(cells_total=nx * ny)
        self._wipe_loads: list = []
        self._prev_over = False
        self.t = 0.0
        self.load = 0.0
        self.contact_xy = None
        self.done = False
        self.planner = RasterPlanner(c, x0, y0)
        return self.observe()

    def step(self, action) -> tuple[dict, bool, dict]:
        """action = (vx, vy, vz) commanded velocity, m/s (clipped). k_z fixed per episode."""
        if self.done:
            raise RuntimeError("episode finished")
        c = self.cfg
        a = np.asarray(action, float).reshape(3)
        a = np.nan_to_num(a)
        a = np.clip(a, [-V_XY_MAX, -V_XY_MAX, -V_Z_MAX], [V_XY_MAX, V_XY_MAX, V_Z_MAX])
        self._queue.append(a)
        a = self._queue.pop(0) if len(self._queue) > self.delay else np.zeros(3)   # control-delay stress test
        self.cmd = self.cmd + a * c.dt
        phase = self.planner.phase(self.t)
        h = 1.0 / c.physics_hz
        loads = []
        for _ in range(c.substeps):
            self._actuate()
            p.stepSimulation(**self._p)
            p.resetBaseVelocity(self.pad, p.getBaseVelocity(self.pad, **self._p)[0], (0, 0, 0), **self._p)
            p.resetBasePositionAndOrientation(self.pad, p.getBasePositionAndOrientation(self.pad, **self._p)[0],
                                              (0, 0, 0, 1), **self._p)
            load, cxy = self._contact()
            loads.append(load)
            self._account(load, cxy, phase, h)
        self.load = float(np.mean(loads))
        self.t += c.dt
        self.stats.ticks += 1
        self._check_termination(phase)
        return self.observe(), self.done, {"phase": phase, "load": self.load, "loads": loads}

    def _actuate(self) -> None:
        c = self.cfg
        pos, _ = p.getBasePositionAndOrientation(self.pad, **self._p)
        vel, _ = p.getBaseVelocity(self.pad, **self._p)
        pos, vel = np.array(pos), np.array(vel)
        k = np.array([c.k_xy, c.k_xy, self.k_z])
        d = 2 * np.sqrt(k * c.pad_mass) * np.array([1.0, 1.0, 0.7])
        f = k * (self.cmd - pos) - d * vel + np.array([0, 0, c.pad_mass * 9.81])
        p.applyExternalForce(self.pad, -1, tuple(f), tuple(pos), p.WORLD_FRAME, **self._p)

    def surface_normal(self, x, y) -> np.ndarray:
        """Analytic unit surface normal (up) at (x, y)."""
        sc, c = self.scenario, self.cfg
        cx, cy = c.surface_center
        a = math.radians(sc.bump_dir_deg)
        s_ = (x - cx) * math.cos(a) + (y - cy) * math.sin(a)
        db = sc.bump_amp * 2 * math.pi / sc.bump_wavelength * math.cos(2 * math.pi * s_ / sc.bump_wavelength + sc.bump_phase)
        gx = math.tan(math.radians(sc.tilt_x_deg)) + db * math.cos(a)
        gy = math.tan(math.radians(sc.tilt_y_deg)) + db * math.sin(a)
        n = np.array([-gx, -gy, 1.0])
        return n / np.linalg.norm(n)

    def _contact(self) -> tuple[float, tuple | None]:
        """Normal load = net pad-surface contact force projected on the analytic surface normal.

        All contact points come from ONE query per substep (no double counting). Each point's force
        vector (normal + both friction directions) is summed first; per-point normals from the
        triangulated heightfield tilt by up to ~20 deg (a mesh artifact, see physics checks), so
        summing per-point normal magnitudes would depend on triangulation. Returns (load, contact xy)."""
        pts = p.getContactPoints(self.pad, self.surface, **self._p)
        if not pts:
            self.shear = 0.0
            return 0.0, None
        F = np.zeros(3)
        w = np.array([max(0.0, q[9]) for q in pts])
        for q in pts:   # force on the pad from the surface: normal on B points from surface toward pad
            F += q[9] * np.array(q[7]) + q[10] * np.array(q[11]) + q[12] * np.array(q[13])
        if w.sum() <= 0:
            self.shear = 0.0
            return 0.0, None
        loc = np.array([q[6] for q in pts])
        xy = (w[:, None] * loc[:, :2]).sum(0) / w.sum()
        n = self.surface_normal(xy[0], xy[1])
        load = float(F @ n)
        self.shear = float(np.linalg.norm(F - load * n))
        return max(0.0, load), (float(xy[0]), float(xy[1]))

    def _cell(self, xy):
        c = self.cfg
        i = int(math.floor((xy[0] - c.region_x[0]) / c.cell))
        j = int(math.floor((xy[1] - c.region_y[0]) / c.cell))
        nx, ny = self.grid_shape
        return (i, j) if 0 <= i < nx and 0 <= j < ny else None

    def _account(self, load: float, cxy, phase: str, h: float) -> None:
        c, st = self.cfg, self.stats
        st.peak_load_n = max(st.peak_load_n, load)
        over = load > c.overload
        if over:
            st.overload_s += h
            if not self._prev_over:
                st.overload_events += 1
        self._prev_over = over
        pos = np.array(p.getBasePositionAndOrientation(self.pad, **self._p)[0])
        rest = float(self.surface_height(pos[0], pos[1])) + c.pad_radius
        pen = rest - pos[2]
        if pen > c.max_penetration:
            st.penetration_s += h
        st.max_penetration_m = max(st.max_penetration_m, pen)
        if phase == "approach":
            st.approach_time_s += h
        elif phase == "retract":
            st.retract_time_s += h
        elif phase == "wipe":
            st.wipe_time_s += h
            if load < c.contact_min:
                st.contact_loss_s += h
            else:
                self._wipe_loads.append(load)
            cell = self._cell(cxy) if (cxy is not None and load >= c.contact_min) else None
            if cell:
                self.touched[cell] = True
                if c.load_lo <= load <= c.load_hi:
                    self.dwell[cell] += h
                    if not self.wiped[cell] and self.dwell[cell] >= c.dwell_s:
                        self.wiped[cell] = True
                        st.events.append({"t": round(self.t, 3), "cell": cell, "load": round(load, 3),
                                          "contact_xy": [round(v, 4) for v in cxy]})

    def _check_termination(self, phase: str) -> None:
        c, st = self.cfg, self.stats
        pos = np.array(p.getBasePositionAndOrientation(self.pad, **self._p)[0])
        hx, hy = c.surface_size[0] / 2, c.surface_size[1] / 2
        cx, cy = c.surface_center
        term = None
        if not (cx - hx - 0.02 < pos[0] < cx + hx + 0.02 and cy - hy - 0.02 < pos[1] < cy + hy + 0.02) or pos[2] > 0.4:
            term = "workspace_escape"
        elif st.max_penetration_m > 1.5 * c.max_penetration + 0.004:
            term = "excess_penetration"                     # pad driven deep into the surface
        elif phase == "done":
            term = "completed"
        if term:
            st.termination = term
            st.cells_wiped = int(self.wiped.sum())
            st.cells_touched = int(self.touched.sum())
            st.coverage = st.cells_wiped / st.cells_total
            st.attempted_coverage = st.cells_touched / st.cells_total
            st.p95_load_n = float(np.percentile(self._wipe_loads, 95)) if self._wipe_loads else 0.0
            st.success = (term == "completed" and st.coverage >= c.success_coverage
                          and st.overload_s <= c.success_overload_s and st.penetration_s == 0.0)
            self.done = True

    # -- observation (full; controllers receive filtered views) -----------------------------

    def observe(self) -> dict:
        pos = np.array(p.getBasePositionAndOrientation(self.pad, **self._p)[0])
        vel = np.array(p.getBaseVelocity(self.pad, **self._p)[0])
        return {"t": self.t, "cmd_pos": self.cmd.copy(), "meas_pos": pos, "meas_vel": vel,
                "normal_load": self.load, "k_z": self.k_z, "path": self.planner}

    def close(self) -> None:
        if p.isConnected(self.client):
            p.disconnect(self.client)


class RasterPlanner:
    """Shared x/y path for every controller: hold at the start cell (approach), then lanes along x,
    one per grid row, at wipe_speed, then hold while retracting. Purely time-based (open loop),
    so all controllers follow identical x/y schedules."""

    def __init__(self, cfg: WipeConfig, x0: float, y0: float):
        c = cfg
        self.c = c
        nx = int(round((c.region_x[1] - c.region_x[0]) / c.cell))
        ny = int(round((c.region_y[1] - c.region_y[0]) / c.cell))
        xa, xb = c.region_x[0] + c.cell / 2 - 0.006, c.region_x[1] - c.cell / 2 + 0.006
        pts = []
        for j in range(ny):
            y = c.region_y[0] + c.cell / 2 + j * c.cell
            pts += [(xa, y), (xb, y)] if j % 2 == 0 else [(xb, y), (xa, y)]
        pts[0] = (x0 - 0.006, y0)
        self.pts = np.array(pts)
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.cum = np.concatenate([[0], np.cumsum(seg)])
        self.wipe_s = self.cum[-1] / c.wipe_speed
        self.nx, self.ny = nx, ny

    def phase(self, t: float) -> str:
        c = self.c
        if t < c.approach_s:
            return "approach"
        if t < c.approach_s + self.wipe_s:
            return "wipe"
        if t < c.approach_s + self.wipe_s + c.retract_s:
            return "retract"
        return "done"

    def xy(self, t: float) -> np.ndarray:
        s = np.clip((t - self.c.approach_s) * self.c.wipe_speed, 0, self.cum[-1])
        i = min(int(np.searchsorted(self.cum, s, side="right")) - 1, len(self.pts) - 2)
        f = (s - self.cum[i]) / max(1e-9, self.cum[i + 1] - self.cum[i])
        return self.pts[i] + f * (self.pts[i + 1] - self.pts[i])
