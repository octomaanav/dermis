"""Newline-delimited JSON framing over a byte stream (TCP, serial bridge, pipe).

TCP delivers bytes, not messages: one read may hold half a frame, several frames, or a frame
split in the middle of a multi-byte UTF-8 character. The framer therefore splits on the byte
0x0A *before* decoding. In UTF-8 the byte 0x0A only ever encodes "\\n" (continuation bytes are
0x80-0xBF), so a complete line is always a complete UTF-8 sequence and decoding per line is exact.

Size limit: a line longer than max_bytes (excluding the newline) is reported as Oversized and its
bytes are discarded as they arrive, up to and including the next newline. The internal buffer
therefore never exceeds max_bytes, whatever the peer sends.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Line:
    data: bytes                 # without the trailing newline (and without a trailing \r)


@dataclass(frozen=True)
class Oversized:
    size: int                   # bytes discarded (a lower bound if the line had not ended yet)


class NDJSONFramer:
    def __init__(self, max_bytes: int = 16384):
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self._buf = bytearray()
        self._discarding = 0            # > 0 while skipping an oversized line
        self.peak_buffered = 0

    def feed(self, data: bytes) -> list:
        out = []
        i = 0
        n = len(data)
        while i < n:
            j = data.find(b"\n", i)
            if self._discarding:
                if j < 0:
                    self._discarding += n - i
                    return out
                out.append(Oversized(self._discarding + (j - i)))
                self._discarding = 0
                i = j + 1
                continue
            end = n if j < 0 else j
            if len(self._buf) + (end - i) > self.max_bytes:
                size = len(self._buf) + (end - i)
                self._buf.clear()
                if j < 0:
                    self._discarding = size
                    return out
                out.append(Oversized(size))
                i = j + 1
                continue
            self._buf += data[i:end]
            self.peak_buffered = max(self.peak_buffered, len(self._buf))
            if j < 0:
                return out
            line = bytes(self._buf)
            self._buf.clear()
            if line.endswith(b"\r"):
                line = line[:-1]
            out.append(Line(line))
            i = j + 1
        return out

    @property
    def buffered(self) -> int:
        return len(self._buf)

    def pending_partial(self) -> bool:
        """True if the stream ended (or is ending) in the middle of a line."""
        return bool(self._buf) or bool(self._discarding)


class MessageError(ValueError):
    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


def parse_line(line: bytes) -> tuple[dict, bool]:
    """UTF-8 decode + JSON parse one line. Returns (object, had_non_finite).

    Strict JSON has no NaN/Infinity, but some encoders emit them, and 1e999 parses to inf.
    They are kept (as float nan/inf) so the adapter can mark just that channel unusable instead of
    dropping the whole frame; had_non_finite lets the caller count them.
    """
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as e:
        raise MessageError("bad_utf8", str(e)) from None
    flagged = [False]

    def const(name):
        flagged[0] = True
        return {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}[name]

    def num(s):
        v = float(s)
        if not math.isfinite(v):
            flagged[0] = True
        return v

    try:
        obj = json.loads(text, parse_constant=const, parse_float=num)
    except (json.JSONDecodeError, RecursionError) as e:
        raise MessageError("bad_json", str(e)) from None
    if not isinstance(obj, dict):
        raise MessageError("not_object", f"top-level JSON is {type(obj).__name__}, expected object")
    return obj, flagged[0]
