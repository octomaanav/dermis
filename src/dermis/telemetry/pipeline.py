"""The single ingestion path for every telemetry source (replay, synthetic, stdin, TCP, wire, serial).

    source -> RawMessage -> adapter -> TelemetryPipeline.process() -> Snapshot

Frame ordering (frame counter is u32 and wraps; differences are taken mod 2**32):
    delta == 0                    duplicate        -> dropped, counted
    0 < delta < 2**31             forward          -> accepted; delta - 1 frames counted as lost
    backward by <= reorder_window late/out of order -> dropped, counted (never re-applied)
    backward by more              stream restart   -> state reset, event (e.g. hub reboot)
    different device_id           device change    -> full reset, event
    config_id change / CONFIG_CHANGED              -> event; held values are discarded because
                                                      scale factors may have changed
    t_device_us goes backwards on a forward frame  device clock reset -> held values discarded, event
    reset_stream() (e.g. a new TCP connection)     -> ordering and holds start over; device_id kept

Per channel the Snapshot keeps three things apart, so a real zero is never confused with "missing":
    *_raw      this frame's value as delivered (NaN where absent or invalid); STALE values included
    *_fresh    mask: usable and not STALE in this frame
    *_age_us   device-clock time since the last fresh sample (inf if none yet)
    *_held     last fresh value while age <= max_hold_us, else NaN; *_held_mask says which,
               *_imputed marks channels whose held value is not from this frame.
That hold is the only imputation, and it is bounded. Wall-clock silence is reported by
source_stale(); teleoperation must pause on it.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

from .adapter import AdapterError, adapt_decoded_frame
from .model import DEFAULT_POD_ORDER, TelemetryFrame

U32 = 1 << 32


@dataclass
class RawMessage:
    obj: dict | None
    t_recv_ns: int
    error: str | None = None


@dataclass
class Snapshot:
    frame: TelemetryFrame
    pods: tuple                      # fixed order, = pipeline.pod_order
    present: np.ndarray              # (P,) pod present in this frame
    mag_raw: np.ndarray              # (P, 3) uT
    mag_valid: np.ndarray            # (P,) usable (maybe stale)
    mag_fresh: np.ndarray
    mag_stale: np.ndarray
    mag_saturated: np.ndarray
    mag_age_us: np.ndarray
    mag_held: np.ndarray
    mag_filtered: np.ndarray             # robust median + low-pass value, held for max_hold_us
    mag_held_mask: np.ndarray
    mag_imputed: np.ndarray
    mag_temp_c: np.ndarray           # NaN if missing; temp freshness is separate from field freshness
    mag_temp_fresh: np.ndarray
    acc_raw: np.ndarray
    gyr_raw: np.ndarray
    imu_valid: np.ndarray
    imu_fresh: np.ndarray
    imu_stale: np.ndarray
    imu_saturated: np.ndarray
    imu_age_us: np.ndarray
    acc_held: np.ndarray
    gyr_held: np.ndarray
    acc_filtered: np.ndarray             # robust median + low-pass value, held for max_hold_us
    gyr_filtered: np.ndarray
    imu_held_mask: np.ndarray
    imu_imputed: np.ndarray
    lost_before: int                 # frames missing between the previous accepted frame and this one
    config_changed: bool
    events: list = field(default_factory=list)

    def summary(self) -> dict:
        f = self.frame
        return {
            "source": f.source, "device_id": f.device_id, "frame": f.frame, "t_device_us": f.t_device_us,
            "t_host_ns": f.t_host_ns, "t_recv_ns": f.t_recv_ns, "config_id": f.config_id,
            "flags": sorted(f.flags), "lost_before": self.lost_before, "events": self.events,
            "pods": {p: {
                "present": bool(self.present[i]),
                "b_ut": _list(self.mag_raw[i]), "mag_fresh": bool(self.mag_fresh[i]),
                "mag_stale": bool(self.mag_stale[i]), "mag_valid": bool(self.mag_valid[i]),
                "mag_age_ms": _ms(self.mag_age_us[i]), "mag_imputed": bool(self.mag_imputed[i]),
                "b_filtered_ut": _list(self.mag_filtered[i]),
                "mag_temp_fresh": bool(self.mag_temp_fresh[i]),
                "acc_mps2": _list(self.acc_raw[i]), "gyr_rads": _list(self.gyr_raw[i]),
                "imu_fresh": bool(self.imu_fresh[i]), "imu_stale": bool(self.imu_stale[i]),
                "imu_valid": bool(self.imu_valid[i]), "imu_age_ms": _ms(self.imu_age_us[i]),
                "acc_filtered_mps2": _list(self.acc_filtered[i]),
                "gyr_filtered_rads": _list(self.gyr_filtered[i]),
                "imu_imputed": bool(self.imu_imputed[i]),
            } for i, p in enumerate(self.pods)},
        }


def _list(v):
    return [None if not np.isfinite(x) else round(float(x), 4) for x in v]


def _ms(us):
    return None if not np.isfinite(us) else round(float(us) / 1000.0, 3)


class _RobustVectorFilter:
    """Short median window rejects isolated spikes; the EMA removes remaining broadband noise.

    Device timestamps set the low-pass coefficient, so the response is stable across source rates.
    A real step survives once it occupies a majority of the short window; a one-frame excursion does not.
    """

    def __init__(self, window: int, tau_s: float):
        self.history = deque(maxlen=max(1, int(window)))
        self.tau_s = max(0.0, float(tau_s))
        self.value: np.ndarray | None = None
        self.last_us: int | None = None

    def update(self, value: np.ndarray, t_us: int) -> np.ndarray:
        x = np.asarray(value, dtype=float)
        self.history.append(x.copy())
        center = np.median(np.asarray(self.history), axis=0)
        if self.value is None:
            self.value = center
        else:
            dt = 0.0 if self.last_us is None else max(0.0, min(0.25, (t_us - self.last_us) / 1e6))
            alpha = 1.0 if self.tau_s == 0.0 else 1.0 - math.exp(-dt / self.tau_s)
            self.value = self.value + alpha * (center - self.value)
        self.last_us = t_us
        return self.value.copy()

    def get(self) -> np.ndarray:
        return np.full(3, np.nan) if self.value is None else self.value.copy()


class TelemetryPipeline:
    def __init__(self, pod_order=DEFAULT_POD_ORDER, max_hold_us: int = 50_000, reorder_window: int = 64,
                 source_timeout_s: float = 0.25, adapter=adapt_decoded_frame, event_log: int = 200,
                 mag_filter_window: int = 5, mag_filter_tau_s: float = 0.08,
                 imu_filter_window: int = 3, acc_filter_tau_s: float = 0.025,
                 gyr_filter_tau_s: float = 0.02):
        self.pod_order = tuple(pod_order)
        self.max_hold_us = max_hold_us
        self.reorder_window = reorder_window
        self.source_timeout_s = source_timeout_s
        self.adapter = adapter
        self.mag_filter_window, self.mag_filter_tau_s = mag_filter_window, mag_filter_tau_s
        self.imu_filter_window = imu_filter_window
        self.acc_filter_tau_s, self.gyr_filter_tau_s = acc_filter_tau_s, gyr_filter_tau_s
        self.events: deque = deque(maxlen=event_log)
        self.counters = {k: 0 for k in ("messages", "accepted", "duplicates", "out_of_order", "lost_frames",
                                        "restarts", "device_changes", "config_changes", "adapter_errors",
                                        "unknown_pods", "pod_set_changes", "clock_resets", "stream_resets")}
        self._reset_stream()
        self.device_id = None
        self.last_recv_ns: int | None = None
        self.last_snapshot: Snapshot | None = None
        # Discovery: every pod/channel that has appeared in an accepted frame, including pods outside
        # pod_order (those are counted in unknown_pods and kept out of the fixed-order arrays).
        self.seen_pods: dict = {}

    def _reset_stream(self) -> None:
        P = len(self.pod_order)
        self.last_frame: int | None = None
        self.last_t_device_us: int | None = None
        self.config_id: int | None = None
        self._pod_set: frozenset | None = None
        self._mag_last = np.full((P, 3), np.nan)
        self._mag_last_t = np.full(P, -np.inf)
        self._acc_last = np.full((P, 3), np.nan)
        self._gyr_last = np.full((P, 3), np.nan)
        self._imu_last_t = np.full(P, -np.inf)
        self._reset_filters()

    def _reset_filters(self) -> None:
        P = len(self.pod_order)
        self._mag_filters = [_RobustVectorFilter(self.mag_filter_window, self.mag_filter_tau_s) for _ in range(P)]
        self._acc_filters = [_RobustVectorFilter(self.imu_filter_window, self.acc_filter_tau_s) for _ in range(P)]
        self._gyr_filters = [_RobustVectorFilter(self.imu_filter_window, self.gyr_filter_tau_s) for _ in range(P)]

    def reset_stream(self, reason: str) -> None:
        """Forget ordering and held values (new connection, operator request). Counters are kept."""
        self.counters["stream_resets"] += 1
        self._event("stream_reset", reason=reason)
        self._reset_stream()

    def _event(self, kind: str, **kw) -> dict:
        e = {"event": kind, **kw}
        self.events.append(e)
        return e

    def ingest(self, msg: RawMessage, source: str) -> Snapshot | None:
        self.counters["messages"] += 1
        if msg.error or msg.obj is None:
            self.counters["adapter_errors"] += 1
            self._event("bad_message", error=msg.error)
            return None
        try:
            frame = self.adapter(msg.obj, msg.t_recv_ns, source)
        except (AdapterError, TypeError, ValueError, KeyError, OverflowError) as e:
            self.counters["adapter_errors"] += 1
            self._event("adapter_error", error=str(e))
            return None
        return self.process(frame)

    def process(self, f: TelemetryFrame) -> Snapshot | None:
        ev = []
        if self.device_id is not None and f.device_id != self.device_id:
            self.counters["device_changes"] += 1
            ev.append(self._event("device_change", old=self.device_id, new=f.device_id))
            self._reset_stream()
        self.device_id = f.device_id

        lost = 0
        if self.last_frame is not None:
            delta = (f.frame - self.last_frame) % U32
            if delta == 0:
                self.counters["duplicates"] += 1
                return None
            if delta >= U32 // 2:
                back = U32 - delta
                if back <= self.reorder_window:
                    self.counters["out_of_order"] += 1
                    return None
                self.counters["restarts"] += 1
                ev.append(self._event("stream_restart", last_frame=self.last_frame, frame=f.frame))
                self._reset_stream()
            else:
                lost = delta - 1
                if lost:
                    self.counters["lost_frames"] += lost
                    ev.append(self._event("frame_gap", after=self.last_frame, frame=f.frame, lost=lost))
                if self.last_t_device_us is not None and f.t_device_us < self.last_t_device_us:
                    self.counters["clock_resets"] += 1
                    ev.append(self._event("clock_reset", last_t_us=self.last_t_device_us, t_us=f.t_device_us))
                    self._mag_last_t[:] = -np.inf      # device-time ages are meaningless across the reset
                    self._imu_last_t[:] = -np.inf

        cfg_changed = "CONFIG_CHANGED" in f.flags or (self.config_id is not None and f.config_id != self.config_id)
        if cfg_changed and self.config_id is not None:
            self.counters["config_changes"] += 1
            ev.append(self._event("config_change", old=self.config_id, new=f.config_id))
            self._mag_last_t[:] = -np.inf          # values under the old config are not comparable
            self._imu_last_t[:] = -np.inf
            self._reset_filters()
        self.config_id = f.config_id
        self.last_frame = f.frame
        self.last_t_device_us = f.t_device_us
        self.last_recv_ns = f.t_recv_ns
        self.counters["accepted"] += 1

        for loc, pod in f.pods.items():
            e = self.seen_pods.setdefault(loc, {"pod_id": pod.pod_id, "mag": False, "imu": False, "frames": 0,
                                                "in_pod_order": loc in self.pod_order})
            e["pod_id"] = pod.pod_id
            e["mag"] |= pod.mag is not None
            e["imu"] |= pod.imu is not None
            e["frames"] += 1
            e["last_frame"] = f.frame
        pod_set = frozenset(f.pods)
        if self._pod_set is not None and pod_set != self._pod_set:
            self.counters["pod_set_changes"] += 1
            ev.append(self._event("pod_set_change", added=sorted(pod_set - self._pod_set),
                                  removed=sorted(self._pod_set - pod_set)))
        self._pod_set = pod_set
        unknown = sorted(pod_set - set(self.pod_order))
        if unknown:
            self.counters["unknown_pods"] += 1

        snap = self._snapshot(f, lost, cfg_changed, ev)
        self.last_snapshot = snap
        return snap

    def _snapshot(self, f: TelemetryFrame, lost: int, cfg_changed: bool, ev: list) -> Snapshot:
        P = len(self.pod_order)
        nan3 = lambda: np.full((P, 3), np.nan)
        z = lambda: np.zeros(P, bool)
        present, mag_raw, mag_valid, mag_fresh, mag_stale, mag_sat = z(), nan3(), z(), z(), z(), z()
        mag_temp, mag_temp_fresh = np.full(P, np.nan), z()
        acc_raw, gyr_raw, imu_valid, imu_fresh, imu_stale, imu_sat = nan3(), nan3(), z(), z(), z(), z()
        now = f.t_device_us
        for i, loc in enumerate(self.pod_order):
            pod = f.pods.get(loc)
            if pod is None:
                continue
            present[i] = True
            m = pod.mag
            if m is not None:
                if m.usable:
                    mag_raw[i] = m.b_ut
                mag_valid[i], mag_fresh[i] = m.usable, m.fresh
                mag_stale[i] = m.usable and "STALE" in m.flags
                mag_sat[i] = m.saturated
                if m.temp_c is not None:
                    mag_temp[i] = m.temp_c
                mag_temp_fresh[i] = m.temp_fresh
                if m.fresh:
                    self._mag_last[i], self._mag_last_t[i] = m.b_ut, m.t_us
                    self._mag_filters[i].update(m.b_ut, m.t_us)
            r = pod.imu
            if r is not None:
                if r.usable:
                    acc_raw[i], gyr_raw[i] = r.acc_mps2, r.gyr_rads
                imu_valid[i], imu_fresh[i] = r.usable, r.fresh
                imu_stale[i] = r.usable and "STALE" in r.flags
                imu_sat[i] = r.saturated
                if r.fresh:
                    self._acc_last[i], self._gyr_last[i], self._imu_last_t[i] = r.acc_mps2, r.gyr_rads, r.t_us
                    self._acc_filters[i].update(r.acc_mps2, r.t_us)
                    self._gyr_filters[i].update(r.gyr_rads, r.t_us)
        mag_age = np.maximum(0.0, now - self._mag_last_t)
        imu_age = np.maximum(0.0, now - self._imu_last_t)
        mag_hold = mag_age <= self.max_hold_us
        imu_hold = imu_age <= self.max_hold_us
        mag_held = np.where(mag_hold[:, None], self._mag_last, np.nan)
        acc_held = np.where(imu_hold[:, None], self._acc_last, np.nan)
        gyr_held = np.where(imu_hold[:, None], self._gyr_last, np.nan)
        mag_filtered = np.array([flt.get() for flt in self._mag_filters])
        acc_filtered = np.array([flt.get() for flt in self._acc_filters])
        gyr_filtered = np.array([flt.get() for flt in self._gyr_filters])
        mag_filtered = np.where(mag_hold[:, None], mag_filtered, np.nan)
        acc_filtered = np.where(imu_hold[:, None], acc_filtered, np.nan)
        gyr_filtered = np.where(imu_hold[:, None], gyr_filtered, np.nan)
        return Snapshot(frame=f, pods=self.pod_order, present=present, mag_raw=mag_raw,
                        mag_valid=mag_valid, mag_fresh=mag_fresh, mag_stale=mag_stale,
                        mag_saturated=mag_sat, mag_age_us=mag_age, mag_held=mag_held,
                        mag_filtered=mag_filtered, mag_held_mask=mag_hold,
                        mag_imputed=mag_hold & ~mag_fresh, mag_temp_c=mag_temp,
                        mag_temp_fresh=mag_temp_fresh, acc_raw=acc_raw, gyr_raw=gyr_raw,
                        imu_valid=imu_valid, imu_fresh=imu_fresh, imu_stale=imu_stale,
                        imu_saturated=imu_sat, imu_age_us=imu_age, acc_held=acc_held,
                        gyr_held=gyr_held, acc_filtered=acc_filtered, gyr_filtered=gyr_filtered,
                        imu_held_mask=imu_hold, imu_imputed=imu_hold & ~imu_fresh,
                        lost_before=lost, config_changed=cfg_changed, events=ev)

    def source_stale(self, now_ns: int | None = None) -> bool:
        """True when nothing has been accepted for source_timeout_s of host time."""
        if self.last_recv_ns is None:
            return True
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        return (now_ns - self.last_recv_ns) > self.source_timeout_s * 1e9

    def stats(self) -> dict:
        return {**self.counters, "device_id": self.device_id, "config_id": self.config_id,
                "last_frame": self.last_frame}
