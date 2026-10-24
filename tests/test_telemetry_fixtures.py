"""Supplied protocol fixtures (firmware/v1) against their schemas, the adapter and the wire decoder."""
import copy
import json

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from dermis.telemetry.adapter import AdapterError, adapt_decoded_frame
from dermis.telemetry.pipeline import RawMessage, TelemetryPipeline
from dermis.telemetry.sources import WireFileSource


@pytest.mark.parametrize("example,schema", [
    ("decoded_frame.json", "decoded_frame.schema.json"),
    ("device_info.json", "device_info.schema.json"),
    ("set_config.json", "set_config.schema.json"),
    ("stats.json", "stats.schema.json"),
])
def test_examples_match_schemas(fixture_path, example, schema):
    s = json.loads(fixture_path(schema).read_text())
    Draft202012Validator.check_schema(s)
    errors = list(Draft202012Validator(s).iter_errors(json.loads(fixture_path(example).read_text())))
    assert errors == []


def test_adapter_reads_supplied_frame(decoded_frame):
    f = adapt_decoded_frame(decoded_frame, 123, "REPLAY")
    assert f.device_id == "dermis-3c84279f2a10" and f.frame == 1234 and f.config_id == 3
    assert f.t_host_ns is None and f.t_recv_ns == 123 and "MARKER" in f.flags
    assert set(f.pods) == {"index_tip", "thumb_tip", "middle_tip", "thenar_apb"}
    assert f.raw is decoded_frame                                   # raw preserved for recording
    idx = f.pods["index_tip"]
    np.testing.assert_allclose(idx.mag.b_ut, [124.956, -283.554, 26999.456])
    assert idx.mag.fresh and idx.imu.fresh


def test_supplied_frame_invalid_and_stale_semantics(decoded_frame):
    snap = TelemetryPipeline().ingest(RawMessage(decoded_frame, 0), "REPLAY")
    i = {p: k for k, p in enumerate(snap.pods)}
    # thenar mag: BAD_LENGTH, b_ut null, valid false -> missing, not zero
    t = i["thenar_apb"]
    assert not snap.mag_valid[t] and not snap.mag_fresh[t] and np.isnan(snap.mag_raw[t]).all()
    assert np.isnan(snap.mag_held[t]).all() and not snap.mag_held_mask[t]
    # thenar IMU still fine
    assert snap.imu_fresh[t]
    # middle IMU: STALE with valid=true -> usable value kept, but not fresh
    m = i["middle_tip"]
    assert snap.imu_valid[m] and snap.imu_stale[m] and not snap.imu_fresh[m]
    assert np.isfinite(snap.acc_raw[m]).all()
    # middle mag: TEMP_STALE only -> field still fresh, temperature not fresh
    assert snap.mag_fresh[m] and not snap.mag_stale[m] and not snap.mag_temp_fresh[m]


def test_genuine_zero_is_not_missing(decoded_frame):
    d = copy.deepcopy(decoded_frame)
    d["pods"]["index_tip"]["mag"]["b_ut"] = [0.0, 0.0, 0.0]
    snap = TelemetryPipeline().ingest(RawMessage(d, 0), "SYNTHETIC")
    assert snap.mag_fresh[0] and (snap.mag_raw[0] == 0).all()


def test_valid_true_but_null_value_is_unusable(decoded_frame):
    d = copy.deepcopy(decoded_frame)
    d["pods"]["index_tip"]["mag"]["b_ut"] = None                    # malformed: valid stays true
    snap = TelemetryPipeline().ingest(RawMessage(d, 0), "SYNTHETIC")
    assert not snap.mag_valid[0] and not snap.mag_fresh[0] and np.isnan(snap.mag_raw[0]).all()


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("pods"),
    lambda d: d["pods"]["index_tip"]["mag"].__setitem__("b_ut", [1, 2]),
    lambda d: d["pods"]["index_tip"].pop("imu"),
    lambda d: d.__setitem__("pods", []),
])
def test_adapter_rejects_malformed(decoded_frame, mutate):
    d = copy.deepcopy(decoded_frame)
    mutate(d)
    with pytest.raises(AdapterError):
        adapt_decoded_frame(d, 0, "SYNTHETIC")
    pipe = TelemetryPipeline()
    assert pipe.ingest(RawMessage(d, 0), "SYNTHETIC") is None and pipe.counters["adapter_errors"] == 1


def test_wire_capture_decodes_through_same_pipeline(fixture_path):
    schema = Draft202012Validator(json.loads(fixture_path("decoded_frame.schema.json").read_text()))
    src = WireFileSource(str(fixture_path("capture_2s.bin")))
    pipe = TelemetryPipeline()
    n = invalid_mag = 0
    for msg in src:
        assert not list(schema.iter_errors(msg.obj))
        snap = pipe.ingest(msg, src.label)
        n += 1
        invalid_mag += int((~snap.mag_valid & snap.present).sum())
    # SPEC section 13: ~700 frames, one dropped frame, one BAD_LENGTH record
    assert 650 < n < 750
    assert pipe.counters["lost_frames"] == 1
    assert invalid_mag == 1
    assert sum(src.stream_errors.values()) == 0


def _close(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) < 1e-3
    return a == b


def test_frame_bin_decodes_to_vector_and_example(fixture_path, decoded_frame):
    import sys
    from dermis.telemetry.sources import WIRE_DIR
    sys.path.insert(0, str(WIRE_DIR))
    import dermis_wire as dw
    vec = next(p for p in json.loads(fixture_path("test_vectors.json").read_text())["packets"] if p["name"] == "frame")
    info = dw.DeviceInfo(json.loads(fixture_path("device_info.json").read_text()))
    pkts = dw.StreamDecoder().feed(fixture_path("frame.bin").read_bytes())
    assert len(pkts) == 1 and pkts[0].type == dw.Msg.FRAME
    decoded = dw.decode_frame(dw.parse_frame(pkts[0]), info)
    assert _close(decoded, vec["decoded"])
    assert _close(decoded, decoded_frame)              # the example JSON is this exact frame
    f = adapt_decoded_frame(decoded, 0, "REPLAY")
    assert "BAD_LENGTH" in f.pods["thenar_apb"].mag.flags and not f.pods["thenar_apb"].mag.usable
