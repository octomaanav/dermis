"""Glove inspection session: discovery, live readout, labelled raw recording, two-point grip calibration.

    python -m dermis glove inspect --source serial --serial-port /dev/cu.usbmodemXXXX --out data/glove/s1
    python -m dermis glove inspect --source tcp --port 8765 --out data/glove/s1

Typed commands (letter + Enter):
    u / l / s / r / w   record a labelled trial: unloaded, light_press, stronger_press, release, sideways_load
    0                   capture the UNLOADED grip reference (hand still, pads untouched)
    1                   capture the COMFORTABLE PRESS grip reference (hold a steady, comfortable press)
    c                   build the two-point grip calibration from 0 and 1 (accepts or rejects with reasons)
    p                   print the full status now
    q                   finish and write the session
Sideways loading is recorded as sideways loading: nothing here detects or labels slip. Magnetic
values are microtesla and unitless closure, never newtons.

Session directory (raw sensor data, kept separate from simulator demonstrations):
    frames.jsonl        every decoded frame as received + `_t_recv_ns` (host monotonic receive time)
    markers.jsonl       trials and reference captures: label, host start/end (monotonic ns + unix s)
    layout.json         discovered pods/channels, DEVICE_INFO if the transport provided it
    meta.json           source label, transport, device/config ids, mapping config, clock anchors, counters
    grip_calibration.json   only if accepted
Device time (`t_device_us` inside each frame) and host receive time are both kept; they are not
synchronised (no clock-sync fit exists), and nothing here claims they are.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from ..telemetry.pipeline import TelemetryPipeline
from ..telemetry.runner import TelemetryRunner
from .discovery import discover
from .grip import CalibrationRejected, GripMappingConfig, ReferenceCapture, build_calibration

TRIAL_KEYS = {"u": "unloaded", "l": "light_press", "s": "stronger_press", "r": "release", "w": "sideways_load"}


class Stamped:
    """Pass-through source wrapper: adds `_t_recv_ns` to each frame (the adapter ignores extra keys)."""

    def __init__(self, src, prefix=()):
        self.src, self.prefix = src, list(prefix)
        self.label = getattr(src, "label", "UNKNOWN")

    @property
    def device_info(self):
        return getattr(self.src, "device_info", None)

    def __iter__(self):
        for m in self.prefix:
            if m.obj is not None:
                m.obj["_t_recv_ns"] = m.t_recv_ns
            yield m
        self.prefix = []
        for m in self.src:
            if m.obj is not None:
                m.obj["_t_recv_ns"] = m.t_recv_ns
            yield m


class _Rest:
    """Iterate the rest of an already-started source iterator (after discovery)."""

    def __init__(self, it, label, device_info):
        self.it, self.label, self.device_info = it, label, device_info

    def __iter__(self):
        return self.it


class GloveSession:
    def __init__(self, source, transport: str, out: str, mapping: GripMappingConfig, trial_s: float = 4.0,
                 reference_s: float = 1.5, operator_confirmed: bool = False, log=print):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.mapping, self.trial_s, self.reference_s, self.log = mapping, trial_s, reference_s, log
        layout, buf, rest, label = discover(source, transport=transport, operator_confirmed=operator_confirmed)
        self.layout, self.label, self.transport = layout, label, transport
        info = getattr(source, "device_info", None)
        stamped = Stamped(_Rest(rest, getattr(source, "label", ""), info), prefix=buf)
        self.runner = TelemetryRunner(stamped, TelemetryPipeline(pod_order=layout.pods), history=20000).start()
        self.runner.start_recording(str(self.out / "frames.jsonl"))
        (self.out / "layout.json").write_text(json.dumps({**layout.to_dict(), "device_info": info}, indent=1, default=str))
        self.anchor = {"host_unix_s": time.time(), "host_monotonic_ns": time.monotonic_ns()}
        self.refs: dict = {}
        self.calibration = None
        self.trials = 0
        self._ema: dict = {}
        self._marker_fh = open(self.out / "markers.jsonl", "a")
        log(f"[{self.label}] device {layout.device_id} config {layout.config_id}; discovered pods:\n" + layout.describe())
        for n in layout.notes:
            log("  note: " + n)
        missing = [p for p in mapping.required_pods if p not in layout.pods or not layout.channels[p]["mag"]]
        if missing:
            log(f"  WARNING: grip mapping requires {missing}, not present in this stream. Adjust "
                f"configs/glove/grip.yaml required_pods, or check wiring/DEVICE_INFO.")

    # -- recording ---------------------------------------------------------------------
    def _collect(self, seconds: float, cap: ReferenceCapture | None = None) -> tuple[int, int]:
        start = self.runner.since(-1)
        last = start[-1][0] if start else 0
        t_end = time.monotonic() + seconds
        n = 0
        while time.monotonic() < t_end:
            for seq, snap in self.runner.since(last):
                last = seq
                n += 1
                if cap:
                    cap.add(snap)
            time.sleep(0.02)
        return n, last

    def _marker(self, label: str, kind: str, start_ns: int, start_unix: float, extra: dict | None = None) -> None:
        self._marker_fh.write(json.dumps({"label": label, "kind": kind, "start_ns": start_ns, "end_ns": time.monotonic_ns(),
                                          "start_unix_s": start_unix, "end_unix_s": time.time(), **(extra or {})}) + "\n")
        self._marker_fh.flush()

    def trial(self, label: str) -> None:
        s_ns, s_unix = time.monotonic_ns(), time.time()
        self.log(f"  recording trial '{label}' for {self.trial_s:.0f} s ...")
        n, _ = self._collect(self.trial_s)
        self._marker(label, "trial", s_ns, s_unix, {"frames": n})
        self.trials += 1
        self.log(f"  trial '{label}' saved ({n} accepted frames)")

    def reference(self, state: str) -> None:
        pods = self.mapping.required_pods + self.mapping.optional_pods
        cap = ReferenceCapture(state, pods)
        s_ns, s_unix = time.monotonic_ns(), time.time()
        self.log(f"  capturing {state.upper()} reference for {self.reference_s:.1f} s: hold still ...")
        self._collect(self.reference_s, cap)
        self.refs[state] = cap
        st = cap.stats()
        self._marker(f"reference_{state}", "reference", s_ns, s_unix, {"stats": st})
        for p, v in st.items():
            sd = "-" if v["std"] is None else f"{max(v['std']):.1f}"
            self.log(f"    {p}: {v['n']} fresh samples, max stationary std {sd} uT")

    def calibrate(self) -> bool:
        if "unloaded" not in self.refs or "press" not in self.refs:
            self.log("  need both references first: 0 (unloaded) then 1 (press)")
            return False
        try:
            cal = build_calibration(self.refs["unloaded"], self.refs["press"], self.mapping)
        except CalibrationRejected as e:
            self.log("  CALIBRATION REJECTED:\n    " + "\n    ".join(e.reasons))
            (self.out / "grip_calibration_rejected.json").write_text(json.dumps(e.report, indent=1))
            return False
        cal.save(self.out / "grip_calibration.json")
        self.calibration = cal
        for p, c in cal.pods.items():
            self.log(f"  {p}: separation {c['separation']:.1f} (>= {self.mapping.min_separation}) accepted")
        self.log(f"  calibration {cal.calibration_id} saved")
        return True

    # -- display -------------------------------------------------------------------------
    def status_lines(self) -> list[str]:
        snap = self.runner.latest()
        st = self.runner.pipeline.stats()
        rate = _rate(self.runner)
        head = (f"[{self.label}] device {st['device_id']} cfg {st['config_id']} | accepted {st['accepted']} "
                f"lost {st['lost_frames']} dup {st['duplicates']} ooo {st['out_of_order']} | ~{rate:.0f} frames/s | "
                f"{'NO DATA' if self.runner.source_stale() else 'receiving'} | trials {self.trials}"
                f" | grip cal {self.calibration.calibration_id if self.calibration else 'none'}")
        if snap is None:
            return [head, "  (no accepted frames yet)"]
        lines = [head]
        unl = self.refs.get("unloaded")
        u_means = unl.stats() if unl else {}
        now = time.monotonic_ns()
        for i, p in enumerate(snap.pods):
            if not snap.present[i]:
                lines.append(f"  {p:12s} MISSING in latest frame")
                continue
            m = "fresh" if snap.mag_fresh[i] else ("STALE" if snap.mag_stale[i] else "INVALID")
            b = snap.mag_raw[i]
            base = (u_means.get(p) or {}).get("mean")
            d = (b - np.array(base)) if base is not None and np.isfinite(b).all() else None
            mag = float(np.linalg.norm(d)) if d is not None else None
            if mag is not None:
                self._ema[p] = mag if p not in self._ema else self._ema[p] + 0.2 * (mag - self._ema[p])
            imu = "fresh" if snap.imu_fresh[i] else ("STALE" if snap.imu_stale[i] else "INVALID/none")
            lines.append(
                f"  {p:12s} mag {m:7s} B=[{_v(b)}] uT  dB=[{_v(d) if d is not None else 'no unloaded ref':>26s}] "
                f"|dB| raw {_f(mag)} filt {_f(self._ema.get(p))}  age {snap.mag_age_us[i] / 1000:5.1f} ms")
            lines.append(f"  {'':12s} imu {imu:7s} acc=[{_v(snap.acc_raw[i])}] m/s2 gyr=[{_v(snap.gyr_raw[i], 3)}] rad/s"
                         f"  host age {(now - snap.frame.t_recv_ns) / 1e6:6.1f} ms")
        return lines

    def close(self) -> dict:
        self.runner.stop_recording()
        self._marker_fh.close()
        meta = {"source_label": self.label, "transport": self.transport, "layout": self.layout.to_dict(),
                "mapping": self.mapping.to_dict(), "clock_anchor": self.anchor,
                "clock_note": "t_device_us (device clock) and _t_recv_ns (host monotonic) are both raw; no sync fit",
                "pipeline": self.runner.pipeline.stats(), "trials": self.trials,
                "grip_calibration_id": self.calibration.calibration_id if self.calibration else None,
                "units": "magnetic field and change in microtesla; closure unitless; nothing is force"}
        (self.out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        self.log(f"session written to {self.out}")
        return meta


def _v(v, nd=1):
    if v is None:
        return "-"
    return ", ".join("  nan" if not np.isfinite(x) else f"{x:8.{nd}f}" for x in v)


def _f(x):
    return "   -   " if x is None else f"{x:7.1f}"


def _rate(runner) -> float:
    h = list(runner.history)[-200:]
    if len(h) < 2:
        return 0.0
    t = [s.frame.t_recv_ns for _, s in h]
    return (len(t) - 1) / max(1e-9, (t[-1] - t[0]) / 1e9)


def run(a) -> int:
    from ..telemetry.sources import open_source
    mapping = GripMappingConfig.load(a.mapping) if a.mapping else GripMappingConfig()
    src = open_source(a.source, path=a.path, port=a.port, serial_port=a.serial_port, wire=a.wire, loop=True,
                      faults=a.faults, rate=a.rate, profile=a.synthetic_profile)
    sess = GloveSession(src, a.source, a.out, mapping, a.trial_s, a.reference_s, a.confirm_hardware)
    cmds: queue.Queue = queue.Queue()
    if a.script:                                   # headless/testing: "0,~3,1,c,u,q" (~N waits N s)
        for c in a.script.split(","):
            cmds.put(c.strip())
    else:
        threading.Thread(target=lambda: [cmds.put(l.strip()) for l in sys.stdin], daemon=True).start()
        print(__doc__.split("Typed commands")[1].split("Session directory")[0])
    last_print = 0.0
    try:
        while True:
            try:
                c = cmds.get(timeout=0.1)
            except queue.Empty:
                c = None
            if c and c.startswith("~"):                     # script only: wait N seconds
                time.sleep(float(c[1:]))
            elif c in TRIAL_KEYS:
                sess.trial(TRIAL_KEYS[c])
            elif c == "0":
                sess.reference("unloaded")
            elif c == "1":
                sess.reference("press")
            elif c == "c":
                sess.calibrate()
            elif c == "q":
                break
            if c == "p" or time.monotonic() - last_print > a.print_every:
                print("\n".join(sess.status_lines()), flush=True)
                last_print = time.monotonic()
            if a.script and cmds.empty():
                break
            if sess.runner.error:
                raise sess.runner.error
    except KeyboardInterrupt:
        pass
    sess.close()
    return 0
