#!/usr/bin/env python3
"""Terminal broadcast panel for the PRISM + FRACTURE commentary duo.

WebKitGTK crashes on this Hyprland/wlroots setup, but kitty renders fine and
themes to the desktop — so the caption layer is a terminal client that reads
the overlay feed (commentary_overlay.py on ws://127.0.0.1:8130) and draws a
framed lower-third: a status card (ball tracker + HP bars + momentum meter),
the duo exchange, and a sources card citing the expert facts PRISM used. Run
it as a tiled kitty panel next to the battle browser (overlay_kitty.sh).

Alignment note: box borders are aligned on VISIBLE width (ANSI escapes have
zero display width), computed by _vis(); _selftest() asserts every boxed line
is equal width. Never pad on len() of a colored string.

Run via:  showdown/overlay_kitty.sh
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import websockets

WS = "ws://127.0.0.1:8130/"

# how many past caption lines to keep on screen at once. The two most recent
# read as "current" (bright/dim); older ones fade to FAINT but stay up long
# enough to actually be read during a fast exchange.
MAX_HISTORY = 4


def _c(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"


ACCENT = _c(124, 220, 255)   # PRISM cyan
SHARD = _c(120, 180, 255)    # FRACTURE crystal blue
VIOLET = _c(177, 140, 255)
GREEN = _c(126, 231, 135)
AMBER = _c(255, 204, 102)
RED = _c(255, 107, 107)
DIM = _c(150, 160, 180)
FAINT = _c(90, 100, 116)
BORDER = _c(70, 82, 100)
WHITE = _c(242, 245, 251)
BOLD = "\033[1m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[3J\033[H"
HIDE_CURSOR = "\033[?25l"

_SPEAKER_COLOR = {"PRISM": ACCENT, "FRACTURE": SHARD}
_ANSI = re.compile(r"\033\[[0-9;?]*[a-zA-Z]")

# spoken desk-read phrase -> compact meter label (matched by substring)
_READ_LABELS = [
    ("all but sealed", "all but sealed"), ("clearly ahead", "clearly ahead"),
    ("real edge", "real edge"), ("dead even", "dead even"),
    ("behind in this", "behind"), ("deep trouble", "deep trouble"),
    ("nearly gone", "nearly gone"),
]


def _vis(s: str) -> int:
    """Visible width: ANSI stripped; every glyph we draw (box-drawing,
    ●○◇, latin, é) is monospace width 1, so len() of the stripped string is
    the display width. (We use Nerd-Font symbols, never width-2 emoji.)"""
    return len(_ANSI.sub("", s))


def _fit(content: str, w: int) -> str:
    """Pad a colored string to visible width w, or hard-truncate (dropping
    color) if it overruns — a box border must never be pushed out of line."""
    v = _vis(content)
    if v <= w:
        return content + " " * (w - v)
    return _ANSI.sub("", content)[:w]


def _pretty(name):
    return name[:1].upper() + name[1:] if name else "—"


NAME_W = 15   # fixed name-column width; fits the longest gen9-OU display name
              # ('Slowking-Galar' = 14). Longer names hard-truncate.


def _namecol(name) -> str:
    """Species name padded (or truncated) to NAME_W so the HP bar and ball
    tracker line up between the US and THEM rows regardless of name length —
    a long name like Slowking-Galar used to shove those columns out of line."""
    return _pretty(name)[:NAME_W].ljust(NAME_W)


def _hp_bar(pct, width=10):
    if pct is None:
        return f"{FAINT}{'·' * width}{RESET}   —"
    pct = max(0, min(100, pct))
    filled = round(pct / 100 * width)
    color = GREEN if pct > 50 else AMBER if pct > 20 else RED
    return (f"{color}{'█' * filled}{FAINT}{'░' * (width - filled)}{RESET}"
            f" {color}{pct:>3}%{RESET}")


def _balls(alive, total=6):
    alive = total if alive is None else max(0, min(total, alive))
    return f"{ACCENT}{'●' * alive}{FAINT}{'○' * (total - alive)}{RESET}"


def _momentum(mom, width=15):
    if mom is None:
        mom = 0.5
    pos = max(0, min(width - 1, round(mom * (width - 1))))
    mcolor = GREEN if mom >= 0.5 else RED
    cells = [(f"{mcolor}{BOLD}●{RESET}" if i == pos else f"{FAINT}─{RESET}")
             for i in range(width)]
    return f"{FAINT}├{RESET}" + "".join(cells) + f"{FAINT}┤{RESET}"


# --- box drawing (all widths visible, self-tested) -------------------------

def _box_top(left: str, right: str, iw: int) -> str:
    """╭─ left ──…── right ─╮ ; interior visible width == iw. Box glyphs stay
    BORDER-colored; the titles keep their own colors (each title ends in a
    RESET, so the border color is re-applied around them explicitly)."""
    prefix = f"{BORDER}─ {RESET}{left}{BORDER} {RESET}"    # "─ " + left + " "
    suffix = (f"{BORDER} {RESET}{right}{BORDER} ─{RESET}"  # " " + right + " ─"
              if right else "")
    fill = max(0, iw - _vis(prefix) - _vis(suffix))
    return (f"{BORDER}╭{RESET}{prefix}{BORDER}{'─' * fill}{RESET}"
            f"{suffix}{BORDER}╮{RESET}\n")


def _box_row(content: str, iw: int) -> str:
    """│ content │ — content padded to iw-2 (one space of inset each side)."""
    return f"{BORDER}│{RESET} {_fit(content, iw - 2)} {BORDER}│{RESET}\n"


def _box_bottom(iw: int) -> str:
    return f"{BORDER}╰{'─' * iw}╯{RESET}\n"


def render(s: dict, connected: bool, history: list | None = None):
    cols, rows = shutil.get_terminal_size((110, 40))
    cw = min(cols - 6, 100)          # card content width
    iw = cw + 2                      # interior width (one inset space each side)
    out = [CLEAR, HIDE_CURSOR, "\n"]

    # --- status card ---------------------------------------------------
    dot = (f"{GREEN}●{RESET} {DIM}LIVE{RESET}" if connected
           else f"{AMBER}○{RESET} {DIM}reconnecting{RESET}")
    turn = s.get("turn")
    left = f"{ACCENT}{BOLD}◆ PRISM{RESET} {DIM}·{RESET} {SHARD}{BOLD}FRACTURE{RESET}"
    right = f"{dot}  {DIM}·{RESET}  {VIOLET}{BOLD}T{turn}{RESET}" if turn is not None else dot
    out.append(_box_top(left, right, iw))

    us, them = s.get("us"), s.get("them")
    if us or them or s.get("us_alive") is not None:
        out.append(_box_row(
            f"{DIM}US  {RESET} {_namecol(us)}{RESET} "
            f"{_hp_bar(s.get('us_hp'))}   {_balls(s.get('us_alive'))}", iw))
        out.append(_box_row(
            f"{DIM}THEM{RESET} {_namecol(them)}{RESET} "
            f"{_hp_bar(s.get('them_hp'))}   {_balls(s.get('them_alive'))}", iw))
        read = (s.get("read") or "").split(",")[0].strip()
        label = next((sh for key, sh in _READ_LABELS if key in read),
                     read[:24])
        out.append(_box_row(
            f"{DIM}MOMENTUM{RESET} {_momentum(s.get('mom'))}  "
            f"{DIM}{label}{RESET}", iw))
    moment = s.get("moment")
    if moment:
        mc = RED if moment.startswith("KNOCK") else VIOLET
        out.append(_box_row(f"{mc}{BOLD}▸ {moment}{RESET}", iw))
    out.append(_box_bottom(iw))
    out.append("\n")

    # --- caption (unboxed — variable length lives outside the frames) --
    citations = s.get("citations") or []
    reserved = len(out) + (len(citations) + 2 if citations else 0) + 2
    avail = max(3, rows - reserved - 1)
    entries = history or [(s.get("persona"), s.get("text") or "")]
    blocks = []
    twidth = cw - 2
    for i, (persona, text) in enumerate(entries):
        if not text:
            continue
        # recency ramp: most recent bright, the 2nd dim ("current" pair),
        # anything older faded to FAINT — visibly in the past but still legible
        depth = len(entries) - 1 - i
        color = WHITE if depth == 0 else DIM if depth == 1 else FAINT
        tag = (f"{_SPEAKER_COLOR.get(persona, WHITE)}{BOLD}{persona}{RESET}  "
               if persona else "")
        indent = len(persona) + 2 if persona else 0
        wrapped = textwrap.wrap(text, twidth - indent) or [""]
        block = [f"   {tag if j == 0 else ' ' * indent}{color}{ln}{RESET}\n"
                 for j, ln in enumerate(wrapped)]
        blocks.append(block)
    left_rows = avail
    kept = []
    for block in reversed(blocks):
        take = block[:max(0, left_rows)]
        if take and len(take) < len(block):
            take[-1] = take[-1].rstrip("\n") + f" {FAINT}…{RESET}\n"
        left_rows -= len(take) + 1
        kept.append(take)
    for block in reversed(kept):
        if block:
            out.extend(block)
            out.append("\n")

    # --- sources card (the expert facts PRISM cited) — PINNED to the
    # bottom of the panel so it reads as a broadcast lower-third footer,
    # not a card floating mid-panel with dead space beneath it ----------
    foot = []
    if citations:
        foot.append(_box_top(f"{DIM}sources{RESET}", "", iw))
        for cite in citations[:3]:
            label = cite.get("label", "?")
            corpus = cite.get("corpus", "")
            foot.append(_box_row(
                f"{ACCENT}◇{RESET} {WHITE}{label}{RESET}"
                f"{DIM}  ·  {corpus}{RESET}", iw))
        foot.append(_box_bottom(iw))
        used = "".join(out).count("\n")
        gap = max(1, rows - used - len(foot) - 1)
        out.append("\n" * gap)
        out.extend(foot)

    sys.stdout.write("".join(out))
    sys.stdout.flush()


def _selftest():
    """Assert every boxed line renders to the same visible width — the
    alignment guarantee. Run: python overlay_kitty.py --selftest"""
    import io
    import contextlib
    shutil.get_terminal_size = lambda fb=(110, 40): __import__(
        "os").terminal_size((110, 40))
    state = {"turn": 27, "text": "The Rapid Spin did nothing to Gholdengo.",
             "persona": "PRISM", "us": "Gholdengo", "us_hp": 28,
             "them": "Iron Treads", "them_hp": 37, "us_alive": 5,
             "them_alive": 4, "mom": 0.62, "read": "we hold a real edge",
             "moment": "KNOCKOUT · Moltres",
             "citations": [{"label": "Rapid Spin", "corpus": "Bulbapedia"}]}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        render(state, True, history=[
            ("PRISM", "Two turns ago we committed the hazards."),
            ("FRACTURE", "I have been screaming about this exact line."),
            ("FRACTURE", "MY spin read!"),
            ("PRISM", state["text"])])
    plain = _ANSI.sub("", buf.getvalue())
    box_widths = {len(ln) for ln in plain.split("\n")
                  if ln and ln[0] in "╭│╰"}
    assert len(box_widths) == 1, f"misaligned box lines: {sorted(box_widths)}"
    print(f"ok — all box lines width {box_widths.pop()}")


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
                    if entry[1] and (not history or history[-1] != entry):
                        history.append(entry)
                        del history[:-MAX_HISTORY]
                    render(state, connected=True, history=history)
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception:
            render(state, connected=False, history=history)
            await asyncio.sleep(2)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        sys.exit(0)
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        sys.stdout.write("\033[?25h")
