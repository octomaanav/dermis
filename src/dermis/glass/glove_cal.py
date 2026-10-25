"""Grip activation from a glove calibration export ("kind": "dermis-calibration", version 1).

The export maps firmware channels to pods (assignments: ch -> palm / middle / thumb / index), and gives
per channel the resting field `b`, its noise `noiseUt`, and the firmware's span direction `unit` (the
direction the field moves when the pad is loaded). GripSensor turns raw b_ut into a per-pod grip that is
ZERO unless the pod is really pressed:

  signal   |(b - rest) . unit| (uT): only movement along the pad's load axis counts, so drift across it
           (temperature, glove shifting on the hand) does not. The absolute value is used because the
           first real test showed the firmware's force going NEGATIVE during a grip (FORCE_CLAMPED_LOW), so
           the sign of `unit` may be reversed relative to a real press. Without an export: |b - rest|.
  active   signal > max(on_sigma * noise, min_on_ut, the venue calibration's per-pod activation level, which
           is 1.5x the largest signal free hand movement produced); stays active until it falls below half that.
           Inactive pods report 0 N.
  rest     starts from the session's first frames with the hand open (falling back to the export's `b`),
           then follows slow drift (time constant adapt_tau_s) of each pod that is not active while the
           caller says the hand is not gripping (a slow creep up to ~on/tau uT/s, ~8 uT/s, is absorbed; a
           real press is far faster). A pod that stays active for stuck_s while the hand is not gripping (an
           offset: a rest taken with the pad loaded, or a step change of the glove on the hand) is re-zeroed
           to its current reading (one pad must press firmly, past solo_n, to start a grasp on its own).
  grip     signal / ut_per_n (N, scaled by the venue calibration), active pods only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROLE_POD = {"palm": "thenar_apb", "middle": "middle_tip", "thumb": "thumb_tip", "index": "index_tip"}
POD_NAMES = {"thenar_apb": ("thenar_apb", "palm_center"), "middle_tip": ("middle_tip",),
             "thumb_tip": ("thumb_tip",), "index_tip": ("index_tip",)}
PODS = ("thumb_tip", "index_tip", "middle_tip", "thenar_apb")
DEFAULT_NOISE_UT = 6.0
GLOVE_CAL = "runs/glass/glove_calibration.json"


def load_export(path: str | None) -> dict | None:
    """Per-pod {'rest', 'noise', 'unit', 'ch'} from a calibration export, or None if absent."""
    if not path or not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text())
    if d.get("kind") != "dermis-calibration" or d.get("version") != 1:
        raise ValueError(f"{path}: not a dermis-calibration v1 export")
    spans = {int(s["ch"]): s for s in d.get("firmware_spans", []) if s.get("ready")}
    out = {}
    for ch, role in d["assignments"].items():
        pod = ROLE_POD.get(role)
        if pod is None:
            continue
        base = d["baseline"].get(str(ch), {})
        unit = spans.get(int(ch), {}).get("unit")
        out[pod] = {"ch": int(ch), "rest": np.array(base["b"], float) if "b" in base else None,
                    "noise": float(base.get("noiseUt", DEFAULT_NOISE_UT)),
                    "unit": (np.array(unit, float) / np.linalg.norm(unit)) if unit else None}
    out["_device_id"] = d.get("device_id")
    return out


def pod_b(frame: dict, pod: str, ch: int | None = None) -> np.ndarray | None:
    """Raw b_ut of a pod: by firmware channel when known (the export's mapping), else by name."""
    pods = frame.get("pods", {})
    mags = [((v or {}).get("mag") or {}) for v in pods.values()]
    mag = next((m for m in mags if ch is not None and m.get("ch") == ch), None)
    if mag is None:
        mag = next((((pods[n] or {}).get("mag") or {}) for n in POD_NAMES[pod] if n in pods), {})
    b = mag.get("b_ut")
    return np.array(b, float) if mag.get("valid") and isinstance(b, list) and len(b) == 3 else None


class GripSensor:
    def __init__(self, export: dict | None = None, ut_per_n: float = 160.0, on_sigma: float = 15.0,
                 off_sigma: float = 7.0, min_on_ut: float = 80.0, adapt_tau_s: float = 5.0, frame_hz: float = 10.0,
                 stuck_s: float = 6.0, activation_ut: dict | None = None):
        self.cal = export or {}
        self.activation_ut = activation_ut or {}         # per pod, measured by the venue calibration
        self.ut_per_n = ut_per_n
        self.on_sigma, self.off_sigma, self.min_on_ut = on_sigma, off_sigma, min_on_ut
        self.alpha = 1.0 / max(1.0, adapt_tau_s * frame_hz)
        self.stuck_frames = int(stuck_s * frame_hz)
        self.active_run = {k: 0 for k in PODS}
        self.rezeroed = []                               # pods re-zeroed as steady offsets, for the caller
        self.rest = {k: None for k in PODS}
        self.active = {k: False for k in PODS}
        self.signal = {k: 0.0 for k in PODS}

    def info(self, pod: str) -> dict:
        return self.cal.get(pod) or {"ch": None, "rest": None, "noise": DEFAULT_NOISE_UT, "unit": None}

    def thresholds(self, pod: str) -> tuple[float, float]:
        n = self.info(pod)["noise"]
        on = max(self.min_on_ut, self.on_sigma * n, self.activation_ut.get(pod) or 0.0)
        return on, max(0.5 * on, self.off_sigma * n)

    def set_rest(self, frames: list):
        """Rest from frames taken with the hand open; pods without readings keep the export's value."""
        for k in PODS:
            bs = [b for b in (pod_b(f, k, self.info(k)["ch"]) for f in frames) if b is not None]
            self.rest[k] = np.mean(bs, 0) if bs else self.info(k)["rest"]

    def raw_signal(self, b: np.ndarray, pod: str) -> float:
        d = b - self.rest[pod]
        u = self.info(pod)["unit"]
        return float(abs(d @ u)) if u is not None else float(np.linalg.norm(d))

    def update(self, frame: dict, gripping: bool = False) -> dict:
        """Per-pod grip in N (0 for inactive pods, None for invalid pods)."""
        out, bs = {}, {}
        for k in PODS:
            b = pod_b(frame, k, self.info(k)["ch"])
            bs[k] = b
            if b is None:
                out[k], self.active[k], self.signal[k] = None, False, 0.0
                continue
            if self.rest[k] is None:
                self.rest[k] = self.info(k)["rest"] if self.info(k)["rest"] is not None else b.copy()
            s = self.raw_signal(b, k)
            on, off = self.thresholds(k)
            self.active[k] = s > (off if self.active[k] else on)
            self.signal[k] = s
            out[k] = s / self.ut_per_n if self.active[k] else 0.0
        for k, b in bs.items():
            self.active_run[k] = self.active_run[k] + 1 if (self.active[k] and not gripping) else 0
            if b is not None and self.active_run[k] >= self.stuck_frames:      # steady offset, not a grip
                self.rest[k], self.active[k], self.active_run[k], out[k] = b.copy(), False, 0, 0.0
                self.rezeroed.append(k)
        if not gripping:                                                 # follow slow drift of quiet pods
            for k, b in bs.items():
                if b is not None and not self.active[k]:
                    self.rest[k] = self.rest[k] + self.alpha * (b - self.rest[k])
        return out
