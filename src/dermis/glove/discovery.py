"""Pod discovery and source labelling. Nothing about the glove layout is hardcoded here: the layout
is the union of DEVICE_INFO pods (when the transport provides it) and pods seen in frames.

Source label (what a human should believe about the data), aligned with the console's rule
(docs/dashboard.md, "Source labels"):
    SYNTHETIC       device_id starts with "synthetic" (in-repo generators, `send-test`), or a synthetic source
    REPLAY          recorded data played back from a file (it may originally come from hardware)
    REAL HARDWARE   a USB serial device, or an operator explicitly confirming the sender is the glove
    TCP-UNVERIFIED  any other network / stdin stream: it could be the glove, a bridge, or a fixture
                    replay with a real-looking device_id (e.g. `send-test --mode fixture`)
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


def source_label(runner_label: str, device_id: str | None, transport: str = "",
                 operator_confirmed: bool = False) -> str:
    if runner_label == "SYNTHETIC" or (device_id or "").lower().startswith("synthetic"):
        return "SYNTHETIC"
    if runner_label == "REPLAY":
        return "REPLAY"
    if transport == "serial" or operator_confirmed:
        return "REAL HARDWARE"
    return "TCP-UNVERIFIED"


@dataclass
class Layout:
    pods: list                                  # discovery order: DEVICE_INFO order first, then first-seen order
    channels: dict                              # loc -> {"mag": bool, "imu": bool, "pod_id": str, "from": [...]}
    device_id: str | None = None
    config_id: int | None = None
    from_device_info: bool = False
    frames_examined: int = 0
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        rows = [f"{p}: mag={'yes' if c['mag'] else 'no'} imu={'yes' if c['imu'] else 'no'} pod_id={c['pod_id'] or '-'}"
                f" (seen in: {', '.join(c['from'])})" for p, c in self.channels.items()]
        return "\n".join(rows) if rows else "(no pods discovered)"


def from_device_info(info: dict) -> dict:
    out = {}
    for pod in info.get("pods", []):
        out[pod["location"]] = {"mag": pod.get("mag_ch") is not None, "imu": pod.get("imu_ch") is not None,
                                "pod_id": pod.get("id", ""), "from": ["device_info"]}
    return out


def merge(device_info: dict | None, seen_pods: dict, device_id=None, config_id=None, frames=0) -> Layout:
    channels = from_device_info(device_info) if device_info else {}
    for loc, s in seen_pods.items():
        c = channels.setdefault(loc, {"mag": False, "imu": False, "pod_id": s.get("pod_id", ""), "from": []})
        c["mag"] |= bool(s.get("mag"))
        c["imu"] |= bool(s.get("imu"))
        c["pod_id"] = c["pod_id"] or s.get("pod_id", "")
        c["from"].append("frames")
    notes = []
    for loc, c in channels.items():
        if c["from"] == ["device_info"]:
            notes.append(f"{loc} is declared in DEVICE_INFO but has not appeared in frames yet")
    return Layout(list(channels), channels, device_id, config_id, bool(device_info), frames, notes)


def discover(source, seconds: float = 1.5, max_frames: int = 300, transport: str = "",
             operator_confirmed: bool = False):
    """Read the first frames of a source to learn its layout. Returns (layout, buffered RawMessages, label);
    replay the buffer into the real pipeline so no data is lost."""
    from ..telemetry.pipeline import TelemetryPipeline
    probe = TelemetryPipeline(pod_order=())                  # no fixed order: everything lands in seen_pods
    buf, t0 = [], time.monotonic()
    it = iter(source)
    for msg in it:
        buf.append(msg)
        probe.ingest(msg, getattr(source, "label", "UNKNOWN"))
        if len(buf) >= max_frames or (time.monotonic() - t0 > seconds and probe.seen_pods):
            break
    info = getattr(source, "device_info", None)
    layout = merge(info, probe.seen_pods, probe.device_id, probe.config_id, len(buf))
    return layout, buf, it, source_label(getattr(source, "label", ""), probe.device_id, transport, operator_confirmed)
