"""WebSocket decoded-frame subscriber and a small replaceable mock WebSocket server.

The bridge deliberately does no frame batching: each received WebSocket text message is
immediately written as one NDJSON record to the existing TCP telemetry receiver.

Examples:
    python -m dermis mock-ws-server --port 8766
    python -m dermis websocket-bridge --url ws://127.0.0.1:8766/frames

Replace the URL (and, if required, --subscribe / --header) with the real service.
Only standard-library networking is used, so this does not add a client dependency.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import time
from contextlib import suppress
from urllib.parse import urlsplit

from .dev_sender import PROFILES, SyntheticGlove

MAX_WS_MESSAGE_BYTES = 1_048_576


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_url(url: str):
    p = urlsplit(url)
    if p.scheme not in {"ws", "wss"} or not p.hostname or p.username or p.password:
        raise ValueError("--url must be a ws:// or wss:// URL without embedded credentials")
    port = p.port or (443 if p.scheme == "wss" else 80)
    path = (p.path or "/") + (("?" + p.query) if p.query else "")
    host_header = p.hostname if p.port is None else f"{p.hostname}:{p.port}"
    return p.scheme, p.hostname, port, path, host_header


def _headers(values: list[str]) -> list[tuple[str, str]]:
    result = []
    for raw in values:
        name, sep, value = raw.partition(":")
        if not sep or not name.strip() or "\r" in raw or "\n" in raw:
            raise ValueError("each --header must be in the form 'Name: value'")
        result.append((name.strip(), value.strip()))
    return result


async def _open_websocket(url: str, extra_headers: list[str]):
    scheme, host, port, path, host_header = _parse_url(url)
    reader, writer = await asyncio.open_connection(host, port, ssl=(ssl.create_default_context() if scheme == "wss" else None),
                                                   server_hostname=(host if scheme == "wss" else None))
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    lines = [
        f"GET {path} HTTP/1.1", f"Host: {host_header}", "Upgrade: websocket", "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}", "Sec-WebSocket-Version: 13",
    ]
    lines.extend(f"{name}: {value}" for name, value in _headers(extra_headers))
    writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("ascii"))
    await writer.drain()
    response = await reader.readuntil(b"\r\n\r\n")
    if len(response) > 16_384:
        raise ConnectionError("WebSocket handshake response is too large")
    status, *header_lines = response.decode("iso-8859-1").split("\r\n")
    response_headers = {}
    for line in header_lines:
        if ":" in line:
            name, value = line.split(":", 1)
            response_headers[name.lower()] = value.strip()
    expected = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
    if not status.startswith("HTTP/") or " 101 " not in status or response_headers.get("sec-websocket-accept") != expected:
        raise ConnectionError(f"WebSocket upgrade refused: {status}")
    return reader, writer


async def _read_frame(reader: asyncio.StreamReader) -> tuple[bool, int, bytes]:
    first, second = await reader.readexactly(2)
    fin, opcode, masked, size = first >> 7, first & 0x0F, second >> 7, second & 0x7F
    if not fin:
        raise ConnectionError("fragmented WebSocket messages are not supported")
    if size == 126:
        size = struct.unpack("!H", await reader.readexactly(2))[0]
    elif size == 127:
        size = struct.unpack("!Q", await reader.readexactly(8))[0]
    if size > MAX_WS_MESSAGE_BYTES:
        raise ConnectionError(f"WebSocket message exceeds {MAX_WS_MESSAGE_BYTES} bytes")
    mask = await reader.readexactly(4) if masked else None
    payload = await reader.readexactly(size)
    if mask:
        payload = bytes(v ^ mask[i % 4] for i, v in enumerate(payload))
    return bool(fin), opcode, payload


async def _read_message(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """Read one data/control message, joining RFC 6455 continuation frames."""
    final, opcode, payload = await _read_frame(reader)
    if opcode >= 0x8:  # control frames are always final and at most 125 bytes
        if not final or len(payload) > 125:
            raise ConnectionError("invalid fragmented or oversized WebSocket control frame")
        return opcode, payload
    if opcode not in {0x1, 0x2}:
        raise ConnectionError(f"unexpected WebSocket opcode {opcode}")
    parts = [payload]
    size = len(payload)
    while not final:
        final, continuation, payload = await _read_frame(reader)
        if continuation != 0:
            raise ConnectionError("WebSocket continuation frame expected")
        size += len(payload)
        if size > MAX_WS_MESSAGE_BYTES:
            raise ConnectionError(f"WebSocket message exceeds {MAX_WS_MESSAGE_BYTES} bytes")
        parts.append(payload)
    return opcode, b"".join(parts)


def _frame(opcode: int, payload: bytes, mask: bool = False) -> bytes:
    """Build a complete single-frame WebSocket message (RFC 6455)."""
    if len(payload) < 126:
        header = bytes([0x80 | opcode, (0x80 if mask else 0) | len(payload)])
    elif len(payload) <= 0xFFFF:
        header = bytes([0x80 | opcode, (0x80 if mask else 0) | 126]) + struct.pack("!H", len(payload))
    else:
        header = bytes([0x80 | opcode, (0x80 if mask else 0) | 127]) + struct.pack("!Q", len(payload))
    if not mask:
        return header + payload
    key = os.urandom(4)
    return header + key + bytes(v ^ key[i % 4] for i, v in enumerate(payload))


async def _forward(url: str, tcp_host: str, tcp_port: int, subscribe: str | None, headers: list[str], reconnect_s: float) -> None:
    forwarded = 0
    while True:
        tcp_writer = ws_writer = None
        try:
            tcp_reader, tcp_writer = await asyncio.open_connection(tcp_host, tcp_port)
            transport = tcp_writer.transport
            sock = transport.get_extra_info("socket")
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            ws_reader, ws_writer = await _open_websocket(url, headers)
            _log(f"[ws-bridge] subscribed to {url}; forwarding to {tcp_host}:{tcp_port}")
            if subscribe is not None:
                ws_writer.write(_frame(0x1, subscribe.encode("utf-8"), mask=True))
                await ws_writer.drain()
            while True:
                opcode, payload = await _read_message(ws_reader)
                if opcode == 0x1:  # text: one decoded-frame JSON object
                    tcp_writer.write(payload.rstrip(b"\r\n") + b"\n")
                    await tcp_writer.drain()  # no batching: hand off every frame now
                    forwarded += 1
                    if forwarded % 100 == 0:
                        _log(f"[ws-bridge] forwarded {forwarded} frames")
                elif opcode == 0x8:
                    raise ConnectionError("WebSocket closed by server")
                elif opcode == 0x9:
                    ws_writer.write(_frame(0xA, payload, mask=True))
                    await ws_writer.drain()
                elif opcode not in {0xA}:  # binary and unsupported control messages
                    _log(f"[ws-bridge] ignored WebSocket opcode {opcode}")
        except (ConnectionError, OSError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError) as exc:
            _log(f"[ws-bridge] disconnected: {exc}; retrying in {reconnect_s:g}s")
        finally:
            for writer in (ws_writer, tcp_writer):
                if writer is not None:
                    writer.close()
                    with suppress(Exception):
                        await writer.wait_closed()
        await asyncio.sleep(reconnect_s)


def add_bridge_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--url", default="ws://127.0.0.1:8766/frames", help="WebSocket frame-stream URL (ws:// or wss://)")
    sp.add_argument("--tcp-host", default="127.0.0.1", help="existing Dermis TCP receiver host")
    sp.add_argument("--tcp-port", type=int, default=8765, help="existing Dermis TCP receiver port")
    sp.add_argument("--subscribe", help="optional UTF-8 text message sent once after WebSocket connection")
    sp.add_argument("--header", action="append", default=[], help="extra handshake header, repeatable: 'Authorization: Bearer TOKEN'")
    sp.add_argument("--reconnect-s", type=float, default=1.0, help="delay before reconnecting after a disconnect")


def cmd_websocket_bridge(a) -> int:
    if a.reconnect_s < 0:
        raise ValueError("--reconnect-s must be non-negative")
    try:
        asyncio.run(_forward(a.url, a.tcp_host, a.tcp_port, a.subscribe, a.header, a.reconnect_s))
    except KeyboardInterrupt:
        _log("[ws-bridge] stopped")
    return 0


async def _mock_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, path: str, clients: set) -> None:
    try:
        request = await reader.readuntil(b"\r\n\r\n")
        lines = request.decode("iso-8859-1").split("\r\n")
        method, got_path, _ = lines[0].split(" ", 2)
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                name, value = line.split(":", 1)
                headers[name.lower()] = value.strip()
        key = headers.get("sec-websocket-key")
        if method != "GET" or got_path != path or not key:
            writer.write(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
            await writer.drain()
            return
        accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        writer.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode("ascii"))
        await writer.drain()
        clients.add(writer)
        while not reader.at_eof():
            opcode, payload = await _read_message(reader)
            if opcode == 0x8:
                return
            if opcode == 0x9:
                writer.write(_frame(0xA, payload))
                await writer.drain()
    except (ConnectionError, OSError, asyncio.IncompleteReadError, UnicodeDecodeError):
        pass
    finally:
        clients.discard(writer)
        writer.close()
        with suppress(Exception):
            await writer.wait_closed()


async def _mock_server(host: str, port: int, path: str, rate: float, profile: str) -> None:
    clients: set[asyncio.StreamWriter] = set()
    server = await asyncio.start_server(lambda r, w: _mock_client(r, w, path, clients), host, port)
    _log(f"[mock-ws] serving ws://{host}:{port}{path} at {rate:g} Hz")
    glove, started, index = SyntheticGlove(profile=profile), time.monotonic(), 0
    try:
        while True:
            deadline = started + index / rate
            await asyncio.sleep(max(0, deadline - time.monotonic()))
            frame = glove.frame_at(index / rate, index)
            payload = json.dumps(frame, separators=(",", ":"), allow_nan=True).encode("utf-8")
            data = _frame(0x1, payload)
            for writer in tuple(clients):
                try:
                    writer.write(data)
                    await writer.drain()
                except (ConnectionError, OSError):
                    clients.discard(writer)
            index += 1
    finally:
        server.close()
        await server.wait_closed()


def add_mock_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--host", default="127.0.0.1", help="listen address")
    sp.add_argument("--port", type=int, default=8766)
    sp.add_argument("--path", default="/frames")
    sp.add_argument("--rate", type=float, default=30.0)
    sp.add_argument("--profile", choices=PROFILES, default="pinch")


def cmd_mock_ws_server(a) -> int:
    if a.rate <= 0 or not a.path.startswith("/"):
        raise ValueError("--rate must be positive and --path must start with '/'")
    try:
        asyncio.run(_mock_server(a.host, a.port, a.path, a.rate, a.profile))
    except KeyboardInterrupt:
        _log("[mock-ws] stopped")
    return 0
