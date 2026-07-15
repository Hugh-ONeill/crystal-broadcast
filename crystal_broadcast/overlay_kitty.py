#!/usr/bin/env python3
"""Terminal lower-third for the Prism commentator.

WebKitGTK crashes on this Hyprland/wlroots setup, but kitty renders fine and
themes to the desktop — so the caption layer is just a small terminal client
that reads the overlay feed (commentary_overlay.py on ws://127.0.0.1:8130)
and reprints the current line, styled. Run it inside a borderless, semi-
transparent, pinned kitty window (overlay_kitty.sh) as a bottom strip over
the battle. No Showdown login, no server patch.

Run via:  showdown/overlay_kitty.sh
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import websockets

WS = "ws://127.0.0.1:8130/"

ACCENT = "\033[38;2;124;220;255m"   # cyan
VIOLET = "\033[38;2;177;140;255m"
DIM = "\033[38;2;150;160;180m"
WHITE = "\033[38;2;242;245;251m"
BOLD = "\033[1m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[3J\033[H"
HIDE_CURSOR = "\033[?25l"
RULE = "─"
DIAMOND = "◆"
DOT = "·"


def render(turn, text: str):
    size = shutil.get_terminal_size((100, 10))
    width = min(size.columns - 4, 118)
    lines = textwrap.wrap(text, width) or [""]
    tnum = f"  {VIOLET}{BOLD}T{turn}{RESET}" if turn is not None else ""
    rule = ACCENT + (RULE * min(width, size.columns - 4)) + RESET
    body = "".join(f"  {WHITE}{ln}{RESET}\n" for ln in lines[:4])
    header = (f"  {ACCENT}{BOLD}{DIAMOND} PRISM{RESET}"
              f"  {WHITE}{DOT}{RESET}  {DIM}ANALYST{RESET}{tnum}")
    sys.stdout.write(f"{CLEAR}\n{header}\n  {rule}\n\n{body}")
    sys.stdout.flush()


async def run():
    sys.stdout.write(HIDE_CURSOR)
    render(None, "waiting for the desk…")
    while True:
        try:
            async with websockets.connect(WS) as ws:
                async for msg in ws:
                    try:
                        d = json.loads(msg)
                    except Exception:
                        continue
                    render(d.get("turn"), d.get("text", ""))
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception:
            await asyncio.sleep(2)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        sys.stdout.write("\033[?25h")  # restore cursor
