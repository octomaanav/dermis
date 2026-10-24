import json
import math

import pytest

from dermis.server.ndjson import Line, MessageError, NDJSONFramer, Oversized, parse_line
from dermis.server.validation import schema_errors


def lines(items):
    return [i.data for i in items if isinstance(i, Line)]


def test_fragmented_message_reassembles():
    f = NDJSONFramer(1024)
    msg = b'{"a": 1, "b": [1, 2, 3]}\n'
    out = []
    for i in range(len(msg)):
        out += f.feed(msg[i:i + 1])
    assert lines(out) == [msg[:-1]]
    assert f.buffered == 0


def test_combined_messages_in_one_read():
    f = NDJSONFramer(1024)
    out = f.feed(b'{"n":1}\n{"n":2}\r\n{"n":3}\n{"n":')
    assert [json.loads(x)["n"] for x in lines(out)] == [1, 2, 3]
    assert f.pending_partial()
    assert [json.loads(x)["n"] for x in lines(f.feed(b'4}\n'))] == [4]


def test_utf8_split_across_chunks():
    f = NDJSONFramer(1024)
    msg = json.dumps({"device_id": "glöve-µ-✓"}, ensure_ascii=False).encode() + b"\n"
    cut = msg.index("✓".encode()) + 1               # inside the 3-byte character
    out = f.feed(msg[:cut]) + f.feed(msg[cut:])
    obj, nf = parse_line(lines(out)[0])
    assert obj["device_id"] == "glöve-µ-✓" and not nf


def test_oversized_line_rejected_and_stream_recovers():
    f = NDJSONFramer(100)
    out = f.feed(b"x" * 250 + b"\n" + b'{"ok":true}\n')
    assert isinstance(out[0], Oversized) and out[0].size == 250
    assert lines(out) == [b'{"ok":true}']


def test_oversized_without_newline_keeps_buffer_bounded():
    f = NDJSONFramer(100)
    out = []
    for _ in range(1000):
        out += f.feed(b"y" * 64)
        assert f.buffered <= 100
    out += f.feed(b"\n" + b'{"n":1}\n')
    assert sum(isinstance(i, Oversized) for i in out) == 1
    assert lines(out) == [b'{"n":1}']
    assert f.peak_buffered <= 100


@pytest.mark.parametrize("raw,kind", [(b'{"a": ', "bad_json"), (b"\xff\xfe", "bad_utf8"), (b"[1,2]", "not_object"),
                                      (b'"str"', "not_object")])
def test_parse_line_errors(raw, kind):
    with pytest.raises(MessageError) as e:
        parse_line(raw)
    assert e.value.kind == kind


def test_non_finite_values_flagged():
    obj, nf = parse_line(b'{"v": [NaN, 1e999, 1.5]}')
    assert nf and math.isnan(obj["v"][0]) and math.isinf(obj["v"][1])
    assert parse_line(b'{"v": 1.5}')[1] is False


def test_fixture_validates_and_bad_frames_do_not(decoded_frame):
    assert schema_errors(decoded_frame) == []
    bad = json.loads(json.dumps(decoded_frame))
    del bad["pods"]["index_tip"]["mag"]["valid"]
    assert schema_errors(bad)
    bad2 = json.loads(json.dumps(decoded_frame))
    bad2["flags"] = ["NOT_A_FLAG"]
    assert schema_errors(bad2)
    assert schema_errors({"pad": "x"})
