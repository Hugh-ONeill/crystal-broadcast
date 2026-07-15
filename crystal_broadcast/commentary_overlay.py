#!/usr/bin/env python3
"""Broadcast-overlay feed server for the Prism commentator.

Replaces the in-room chat relay (prism_relay.py) with a clean lower-third
caption: no Showdown login, no server patch, no impersonation. It simply
subscribes to AIRI's output feed (the same stream airi_bridge.py --watch
reads) and re-publishes each finalized commentary line to a browser overlay.

Two local endpoints:
  * http://127.0.0.1:8129/          -> serves overlay.html (the caption page)
  * ws://127.0.0.1:8130/            -> pushes {"turn", "text"} per new line

Two ways to display it, both fed from here:
  * overlay_kitty.sh  — a tiled kitty panel (recommended on this Hyprland
    setup; WebKitGTK crashes here). Reads the WS feed directly.
  * broadcast.html    — a single browser page that iframes the local battle
    with the caption composited on top (local-server only).

Run:  python showdown/commentary_overlay.py
"""
from __future__ import annotations

import asyncio
import functools
import json
import re
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

sys.path.insert(0, str(Path(__file__).parent.parent))

import websockets

from showdown.airi_bridge import (
    DEFAULT_URL as AIRI_URL, _load_token, _sanitize, _unwrap)

HTTP_PORT = 8129
WS_PORT = 8130
HERE = Path(__file__).parent

_clients: set = set()
# last line sent, so a browser that connects mid-match shows the current
# caption immediately instead of a blank bar
_latest: dict = {"turn": None, "text": ""}


def _turn_of(beat: str):
    m = re.search(r"\bT(\d+)\b", beat)
    return int(m.group(1)) if m else None


async def _broadcast(payload: str):
    for c in list(_clients):
        try:
            await c.send(payload)
        except Exception:
            _clients.discard(c)


async def _ws_handler(conn):
    _clients.add(conn)
    try:
        if _latest["text"]:
            await conn.send(json.dumps(_latest))
        async for _ in conn:  # overlay never sends; just hold the socket open
            pass
    except Exception:
        pass
    finally:
        _clients.discard(conn)


async def _airi_listener():
    """Subscribe to AIRI, sanitize each finalized reply, publish to overlays.
    Mirrors airi_bridge.watch()'s connect/keepalive/reconnect behavior."""
    while True:
        try:
            async with websockets.connect(AIRI_URL) as ws:
                await ws.send(json.dumps({
                    "type": "module:authenticate",
                    "data": {"token": _load_token()},
                }))
                print("overlay feed: connected to AIRI", flush=True)
                last_keepalive = time.monotonic()
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    except asyncio.TimeoutError:
                        raw = None
                    if time.monotonic() - last_keepalive > 30:
                        await ws.send(json.dumps({
                            "type": "module:authenticate",
                            "data": {"token": _load_token()},
                        }))
                        last_keepalive = time.monotonic()
                    if raw is None:
                        continue
                    msg = _unwrap(raw)
                    if msg.get("type") != "output:gen-ai:chat:complete":
                        continue
                    data = msg.get("data", {})
                    beat = (data.get("text") or "").strip()
                    message = data.get("message") or {}
                    cat = message.get("categorization") or {}
                    reply = (cat.get("speech")
                             or message.get("content") or "").strip()
                    if not beat.startswith("["):
                        continue  # only battle-feed replies, not stray chat
                    clean = _sanitize(reply)
                    if not clean:
                        continue
                    _latest.update(turn=_turn_of(beat), text=clean)
                    print(f"overlay feed -> {clean[:70]}", flush=True)
                    await _broadcast(json.dumps(_latest))
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as e:
            print(f"overlay feed: AIRI disconnected ({e!r}); retry 3s",
                  flush=True)
            await asyncio.sleep(3)


def _serve_http():
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(HERE))
    ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), handler).serve_forever()


async def main():
    Thread(target=_serve_http, daemon=True).start()
    print(f"overlay page: http://127.0.0.1:{HTTP_PORT}/overlay.html", flush=True)
    async with websockets.serve(_ws_handler, "127.0.0.1", WS_PORT):
        print(f"overlay ws:   ws://127.0.0.1:{WS_PORT}/", flush=True)
        await _airi_listener()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
