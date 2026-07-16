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


# --- structured HUD extraction from the beat text --------------------------
# gen9_player emits beats like:
#   [BATTLE T14] Last exchange: ... gholdengo (59% hp) vs cinderace (100% hp).
#   We go for switch zamazenta. Desk read: it's dead even right now, holding
#   steady. Bodies: us 5 standing, them 4.
# species are display names and can be multi-word ("Great Tusk", "Iron
# Valiant"); continuation words are always capitalized, so require that —
# the old single-word groups truncated ("KNOCKOUT · Tusk"). No '.' in the
# general class: it let a name swallow the sentence boundary ("Ting-Lu.
# Zamazenta"), so the four dotted species are explicit literals instead
_DOTTED = r"Mr\. Mime-Galar|Mr\. Mime|Mr\. Rime|Mime Jr\."
_NAME = rf"(?:{_DOTTED}|[A-Z][\w'\-]*(?: [A-Z][\w'\-]*)*)"
_ACTIVE_RE = re.compile(
    rf"({_NAME}) \((\d+)% hp\) vs ({_NAME}) \((\d+)% hp\)")
_BODIES_RE = re.compile(r"Bodies: us (\d+) standing, them (\d+)")
_LEFT_RE = re.compile(r"Left standing: us (\d+), them (\d+)")
_READ_RE = re.compile(r"Desk read: ([^.]+?)(?:\.|$)")
_TERA_RE = re.compile(rf"({_NAME}) Terastallized into an? (\w+) type")
# every KO phrasing gen9_player emits — attributed, residual, and the flat
# fallback lines — so the "big moment" banner never misses a knockout
_KO_RES = [
    re.compile(rf"knocked out ({_NAME})"),
    re.compile(rf"({_NAME}) went down"),
    re.compile(rf"Their ({_NAME}) (?:is|are) down"),
    re.compile(rf"We lost ({_NAME})"),
]
# read phrase -> momentum (our win share, 0=lost .. 1=won); mirrors
# gen9_player._read_phrase's bands
_READ_MOMENTUM = [
    ("all but sealed", 0.93), ("clearly ahead", 0.77), ("real edge", 0.62),
    ("dead even", 0.50), ("behind in this", 0.35), ("deep trouble", 0.18),
    ("nearly gone", 0.06),
]


def _parse_beat(beat: str) -> dict:
    """Pull scoreboard/momentum fields out of the beat text. Missing fields
    stay None so the panel can degrade gracefully (e.g. MATCH START has no
    active mons yet)."""
    hud: dict = {"us": None, "us_hp": None, "them": None, "them_hp": None,
                 "us_alive": None, "them_alive": None, "mom": None,
                 "read": None, "moment": None}
    m = _ACTIVE_RE.search(beat)
    if m:
        hud.update(us=m.group(1), us_hp=int(m.group(2)),
                   them=m.group(3), them_hp=int(m.group(4)))
    m = _BODIES_RE.search(beat)
    if m:
        hud.update(us_alive=int(m.group(1)), them_alive=int(m.group(2)))
    m = _LEFT_RE.search(beat)  # RESULT beat
    if m:
        hud.update(us_alive=int(m.group(1)), them_alive=int(m.group(2)))
    m = _READ_RE.search(beat)
    if m:
        read = m.group(1).strip()
        hud["read"] = read
        low = read.lower()
        hud["mom"] = next((v for k, v in _READ_MOMENTUM if k in low), 0.5)
    if beat.startswith("[RESULT]"):
        hud["mom"] = 1.0 if " WIN " in beat else 0.0 if " LOSS " in beat else 0.5
    # a "big moment" banner: prefer a KO, else a Terastallization
    ko = next((r.search(beat).group(1) for r in _KO_RES if r.search(beat)), None)
    tera = _TERA_RE.search(beat)
    if ko:
        hud["moment"] = f"KNOCKOUT · {ko}"
    elif tera:
        hud["moment"] = f"TERA · {tera.group(1)} → {tera.group(2)}"
    return hud


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
                    hud = _parse_beat(beat)
                    # the desk read is only spoken when it changes, so most
                    # beats carry none — hold the meter at its last value
                    # instead of snapping to center (reset on a new match)
                    if not beat.startswith("[MATCH START]"):
                        if hud["mom"] is None:
                            hud["mom"] = _latest.get("mom")
                        if hud["read"] is None:
                            hud["read"] = _latest.get("read")
                    _latest.clear()
                    _latest.update(turn=_turn_of(beat), text=clean, **hud)
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
