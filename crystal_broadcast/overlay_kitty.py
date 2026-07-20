#!/usr/bin/env python3
"""Terminal broadcast panel for the Prism commentator.

WebKitGTK crashes on this Hyprland/wlroots setup, but kitty renders fine and
themes to the desktop — so the caption layer is a terminal client that reads
the overlay feed (commentary_overlay.py on ws://127.0.0.1:8130) and draws a
lower-third broadcast graphic: a ball tracker + HP bars + momentum meter +
big-moment banner + Prism's prose. Run it as a tiled kitty panel next to the
battle browser (overlay_kitty.sh). No Showdown login, no server patch.

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


def _c(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"


ACCENT = _c(124, 220, 255)   # cyan
VIOLET = _c(177, 140, 255)
GREEN = _c(126, 231, 135)
AMBER = _c(255, 204, 102)
RED = _c(255, 107, 107)
DIM = _c(150, 160, 180)
FAINT = _c(90, 100, 116)
WHITE = _c(242, 245, 251)
BOLD = "\033[1m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[3J\033[H"
HIDE_CURSOR = "\033[?25l"


def _pretty(name):
    return name[:1].upper() + name[1:] if name else "—"


def _hp_bar(pct, width=12):
    if pct is None:
        return DIM + "·" * width + RESET
    pct = max(0, min(100, pct))
    filled = round(pct / 100 * width)
    color = GREEN if pct > 50 else AMBER if pct > 20 else RED
    return (color + "█" * filled + FAINT + "░" * (width - filled) + RESET
            + f" {color}{pct:>3}%{RESET}")


def _balls(alive, total=6):
    if alive is None:
        alive = total
    alive = max(0, min(total, alive))
    return ACCENT + "●" * alive + FAINT + "○" * (total - alive) + RESET


def _momentum(mom, width=17):
    if mom is None:
        mom = 0.5
    pos = max(0, min(width - 1, round(mom * (width - 1))))
    mcolor = GREEN if mom >= 0.5 else RED
    cells = []
    for i in range(width):
        if i == pos:
            cells.append(f"{mcolor}{BOLD}●{RESET}")
        else:
            cells.append(f"{FAINT}─{RESET}")
    return f"{FAINT}├{RESET}" + "".join(cells) + f"{FAINT}┤{RESET}"


SHARD = _c(120, 180, 255)      # FRACTURE's crystal blue
_SPEAKER_COLOR = {"PRISM": ACCENT, "FRACTURE": SHARD}


def render(s: dict, connected: bool, history: list | None = None):
    cols, rows = shutil.get_terminal_size((110, 40))
    inner = min(cols - 4, 120)
    rule = ACCENT + "─" * inner + RESET

    # header: left brand, right status + turn
    dot = (f"{GREEN}●{RESET} {DIM}LIVE{RESET}" if connected
           else f"{AMBER}○{RESET} {DIM}reconnecting…{RESET}")
    turn = s.get("turn")
    ttag = f"{VIOLET}{BOLD}TURN {turn}{RESET}" if turn is not None else ""
    left_plain = "◆ PRISM · FRACTURE"
    right_plain = ("● LIVE" if connected else "○ reconnecting…") + \
                  (f"    TURN {turn}" if turn is not None else "")
    left = (f"{ACCENT}{BOLD}◆ PRISM{RESET}  {WHITE}·{RESET}  "
            f"{SHARD}{BOLD}FRACTURE{RESET}")
    right = f"{dot}    {ttag}" if ttag else dot
    pad = max(1, inner - len(left_plain) - len(right_plain))
    out = [CLEAR, HIDE_CURSOR, "\n", f"  {left}{' ' * pad}{right}\n",
           f"  {rule}\n\n"]

    # scoreboard rows
    us, them = s.get("us"), s.get("them")
    if us or them or s.get("us_alive") is not None:
        out.append(f"  {DIM}US  {RESET} {_balls(s.get('us_alive'))}   "
                   f"{WHITE}{_pretty(us):<13}{RESET}{_hp_bar(s.get('us_hp'))}\n")
        out.append(f"  {DIM}THEM{RESET} {_balls(s.get('them_alive'))}   "
                   f"{WHITE}{_pretty(them):<13}{RESET}"
                   f"{_hp_bar(s.get('them_hp'))}\n")
        # just the position clause (before the comma); the swing is in prose
        read = (s.get("read") or "").split(",")[0].strip()
        out.append(f"  {DIM}MOMENTUM{RESET} {_momentum(s.get('mom'))}  "
                   f"{DIM}{read[:28]}{RESET}\n")

    # big-moment banner
    moment = s.get("moment")
    if moment:
        mc = RED if moment.startswith("KNOCK") else VIOLET
        out.append(f"\n  {mc}{BOLD}▸ {moment}{RESET}\n")

    out.append(f"  {rule}\n\n")

    # caption block — use the panel's real height; a full reply wraps to
    # 5-6 lines and a hard 3-line cap silently ate the second half. With
    # the duo live, show the exchange: previous line dimmed, current line
    # bright, each with a color-coded speaker tag (the sub-second
    # speaker-identification requirement, solved visually).
    used = 6 + (3 if (us or them or s.get("us_alive") is not None) else 0) \
        + (2 if moment else 0)
    avail = max(3, rows - used - 1)
    entries = history or [(s.get("persona"), s.get("text") or "")]
    blocks = []
    for i, (persona, text) in enumerate(entries):
        if not text:
            continue
        current = i == len(entries) - 1
        color = WHITE if current else DIM
        tag = ""
        if persona:
            pc = _SPEAKER_COLOR.get(persona, WHITE)
            tag = f"{pc}{BOLD}{persona}{RESET}  "
        wrapped = textwrap.wrap(text, inner - (len(persona) + 2
                                               if persona else 0))
        block = []
        for j, ln in enumerate(wrapped):
            prefix = tag if j == 0 else " " * (len(persona) + 2
                                               if persona else 0)
            block.append(f"  {prefix}{color}{ln}{RESET}\n")
        blocks.append(block)
    # newest block gets space first; older ones fill what remains
    lines_left = avail
    kept = []
    for block in reversed(blocks):
        take = block[:max(0, lines_left)]
        if len(take) < len(block) and take:
            take[-1] = take[-1].rstrip("\n")[: inner - 2].rstrip() + " …\n"
        lines_left -= len(take) + 1  # +1 for the blank spacer
        kept.append(take)
    for block in reversed(kept):
        if block:
            out.extend(block)
            out.append("\n")

    sys.stdout.write("".join(out))
    sys.stdout.flush()


async def run():
    sys.stdout.write(HIDE_CURSOR)
    state = {"turn": None, "text": "waiting for the desk…"}
    history: list = []
    render(state, connected=False)
    while True:
        try:
            async with websockets.connect(WS) as ws:
                render(state, connected=True, history=history)
                async for msg in ws:
                    try:
                        state = json.loads(msg)
                    except Exception:
                        continue
                    entry = (state.get("persona"), state.get("text") or "")
                    # the hub re-sends the latest line to fresh connections;
                    # don't duplicate it in the exchange history
                    if entry[1] and (not history or history[-1] != entry):
                        history.append(entry)
                        del history[:-2]
                    render(state, connected=True, history=history)
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception:
            render(state, connected=False, history=history)
            await asyncio.sleep(2)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        sys.stdout.write("\033[?25h")  # restore cursor
