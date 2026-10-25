#!/usr/bin/env python3
"""Serve dermis-hand-practice-local.html and proxy review requests to Claude.

The Anthropic key is read from .env (ANTHROPIC_API_KEY=...) and stays on this machine;
the page only ever talks to /api/claude on this server.

    python3 serve_practice.py            # then open http://127.0.0.1:8090
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "dermis-hand-practice-local.html"
UPSTREAM = "https://api.anthropic.com/v1/messages"
MAX_BODY = 8 * 1024 * 1024


def load_env(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines, optional quotes, # comments. Real env vars win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        os.environ.setdefault(key.strip().removeprefix("export ").strip(), value)


def api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return "" if not key or "REPLACE" in key else key


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # close-delimited body, so streamed replies need no chunking

    def log_message(self, fmt, *args):
        print("[practice] " + fmt % args, flush=True)

    def host_ok(self) -> bool:
        # Refuse requests whose Host is not loopback (DNS-rebinding guard).
        return self.headers.get("Host", "").rsplit(":", 1)[0] in ("127.0.0.1", "localhost", "[::1]")

    def reply(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.host_ok():
            return self.reply(403, b"forbidden", "text/plain")
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/status":
            return self.reply(200, json.dumps({"key": bool(api_key())}).encode(), "application/json")
        self.reply(404, b"not found", "text/plain")

    def do_POST(self):
        if not self.host_ok() or self.path != "/api/claude":
            return self.reply(404, b"not found", "text/plain")
        key = api_key()
        if not key:  # 503 tells the page the key is missing, distinct from Anthropic's own 401
            return self.reply(503, json.dumps({"error": {"message": "ANTHROPIC_API_KEY not set in .env"}}).encode(), "application/json")
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY:
            return self.reply(413, b"bad body size", "text/plain")
        req = urllib.request.Request(
            UPSTREAM, data=self.rfile.read(length), method="POST",
            headers={"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"})
        try:
            resp = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            return self.reply(e.code, e.read(), e.headers.get("Content-Type", "application/json"))
        except OSError as e:
            return self.reply(502, json.dumps({"error": {"message": f"cannot reach Anthropic: {e}"}}).encode(), "application/json")
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.headers.get("Content-Type", "text/event-stream"))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with resp:
            while chunk := resp.read1(4096):
                self.wfile.write(chunk)
                self.wfile.flush()


def main() -> None:
    load_env(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Dermis hand practice server")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[practice] http://127.0.0.1:{args.port}  (API key {'loaded from .env' if api_key() else 'MISSING: set ANTHROPIC_API_KEY in .env'})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
