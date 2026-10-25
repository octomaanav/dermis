"""Hold status for fragile_cup, computed from MEASURED simulator state every control tick.

Movement status (one of):
    HOLDING   cup within the grasp region (slip < drop threshold) and not SLIPPING
    SLIPPING  downward speed of the cup relative to the gripper > ENTER_MPS for ENTER_TICKS consecutive
              ticks; cleared only after EXIT_TICKS consecutive ticks below EXIT_MPS (hysteresis, so noise
              does not make the label flicker)
    DROPPED   the benchmark drop rule: slip > drop_slip_m (3 cm) or no finger contact for 3 ticks
              (latched; identical to the env's 'drop' termination)
Load status (separate: a stationary cup can still be over-squeezed):
    OK / SUSTAINED-SQUEEZE WARNING (sustained model: more than half the N*s budget used) /
    EXCESSIVE LOAD (per-finger normal force > break_force_n, or the sustained budget exceeded; these are
    the benchmark's SIMULATED limits). Never shown as cup bending: the cup is rigid.
Outcome: the env's own termination (success / drop / damage_break / damage_sustained / excess_slip / timeout).

Known, explained discrepancy: 'excess_slip' is decided only at the end (total slip > success_slip_m =
1.5 cm but < 3 cm). A slow creep under ENTER_MPS shows as HOLDING while it happens; the final
outcome then reads EXCESS SLIP. The overlay shows the slip distance in cm so this is visible.
"""
from __future__ import annotations

ENTER_MPS, ENTER_TICKS = 0.003, 5
EXIT_MPS, EXIT_TICKS = 0.001, 10


class HoldStatus:
    def __init__(self):
        self.reset()

    def reset(self):
        self.movement = "HOLDING"
        self._fast = self._slow = 0
        self.slip_events = 0

    def update(self, env) -> dict:
        c, st = env.cfg, env.stats
        rel_vz = float(env.observe()[env_obs_index("obj_vz")])          # m/s, negative = cup moving down vs gripper
        down = -rel_vz
        slip = st.final_slip_m
        normals = env._contact["normal"]
        if st.termination == "drop" or slip > c.drop_slip_m:
            self.movement = "DROPPED"
        elif self.movement != "DROPPED":
            if down > ENTER_MPS:
                self._fast, self._slow = self._fast + 1, 0
            elif down < EXIT_MPS:
                self._slow, self._fast = self._slow + 1, 0
            else:
                self._fast = self._slow = 0
            if self.movement == "HOLDING" and self._fast >= ENTER_TICKS:
                self.movement = "SLIPPING"
                self.slip_events += 1
            elif self.movement == "SLIPPING" and self._slow >= EXIT_TICKS:
                self.movement = "HOLDING"
        sustained = "sustained" in c.damage_model
        if st.termination in ("damage_break", "damage_sustained") or float(normals.max()) > c.break_force_n:
            load = "EXCESSIVE LOAD"
        elif sustained and st.damage_ns > 0.5 * c.damage_budget_ns:
            load = "SUSTAINED-SQUEEZE WARNING"
        else:
            load = "OK"
        end = env.end_t
        return {"movement": self.movement, "load_status": load, "outcome": st.termination,
                "success": st.success, "slip_cm": round(slip * 100, 2), "rel_down_speed_mm_s": round(down * 1000, 2),
                "hold_s": round(env.t, 2), "required_s": None if end is None else round(end, 2),
                "damage_ns": round(st.damage_ns, 2) if sustained else None,
                "budget_ns": c.damage_budget_ns if sustained else None, "slip_events": self.slip_events}


_IDX = {}


def env_obs_index(name: str) -> int:
    if not _IDX:
        from .env import OBS_NAMES
        _IDX.update({n: i for i, n in enumerate(OBS_NAMES)})
    return _IDX[name]


def outcome_text(term: str | None) -> str:
    return {None: "", "success": "SUCCESS: held for the full schedule", "drop": "DROPPED",
            "damage_break": "EXCESSIVE LOAD (instant simulated limit)",
            "damage_sustained": "EXCESSIVE LOAD (sustained simulated squeeze budget)",
            "excess_slip": "EXCESS SLIP (> 1.5 cm total; benchmark failure)", "timeout": "TIMED OUT (no load applied)",
            "setup_failed": "SETUP FAILED", "aborted": "ABORTED"}.get(term, term)
