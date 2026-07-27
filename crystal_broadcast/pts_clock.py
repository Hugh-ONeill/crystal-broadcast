#!/usr/bin/env python3
"""PTS scheduling: release a beat when the VIEWER reaches its turn.

The engine resolves a turn in milliseconds; the client animates it over
seconds. Measured on a real broadcast match (69 turns, 2026-07-27):

    per-turn lag   min 1.00s   median 13.59s   max 158.56s   spread 157.56s

So commentary fired at engine time lands anywhere from a second to two and a
half minutes ahead of the picture, and no fixed `--airi-turn-pace` can be
right across that range. broadcast_clock.js reports when each line is actually
PRESENTED; this subscribes to that feed and lets the caster hold a finished
line until its turn is on screen.

WHERE THE HOLD GOES: after generation, before publish. Generation costs ~8s
and the lag is usually far larger, so generating first spends the lag we
already have instead of adding to it. The line is ready and waiting when the
viewer gets there.

INERT BY DEFAULT. With no --pts-url the caster never constructs one, and
publishing is byte-for-byte unchanged. A hold is also always bounded: if the
broadcast page is closed, or the viewer never reaches that turn, the beat goes
out at max_hold rather than never. Silence is worse than late.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import time

import websockets

DEFAULT_URL = "ws://127.0.0.1:8132/"
DEFAULT_MAX_HOLD = 180.0


class PresentationClock:
    """Tracks how far the viewer has got, and waits for a given turn."""

    def __init__(self, url: str = DEFAULT_URL,
                 max_hold: float = DEFAULT_MAX_HOLD):
        self.url = url
        self.max_hold = max_hold
        self.highest_turn: int | None = None
        self.ended = False
        self.connected = False
        self.holds = 0
        self.held_seconds = 0.0
        self.timeouts = 0
        self._pulse = asyncio.Event()
        self._task: asyncio.Task | None = None

    # --- feed ----------------------------------------------------------
    def ingest(self, ev: dict):
        """Fold one clock event. Split out from the socket so the wait logic
        is testable without a network."""
        if ev.get("kind") != "presented":
            return
        line = ev.get("line") or ""
        if line.startswith("|turn|"):
            try:
                turn = int(line.split("|")[2])
            except (IndexError, ValueError):
                return
            if self.highest_turn is None or turn > self.highest_turn:
                self.highest_turn = turn
                self._wake()
        elif line.startswith("|win|") or line.startswith("|tie"):
            self.ended = True
            self._wake()
        elif line.startswith("|start"):
            # a new battle: the viewer is back at the beginning
            self.highest_turn = None
            self.ended = False
            self._wake()

    def _wake(self):
        self._pulse.set()
        self._pulse.clear()

    async def _run(self):
        backoff = 0.5
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    self.connected = True
                    backoff = 0.5
                    print(f"pts: following {self.url}", flush=True)
                    async for msg in ws:
                        try:
                            self.ingest(json.loads(msg))
                        except (ValueError, TypeError):
                            continue
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            self.connected = False
            self._wake()   # unblock anyone waiting on a feed that just died
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 5.0)

    def start(self):
        self._task = asyncio.create_task(self._run())
        return self._task

    async def stop(self):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # --- the gate ------------------------------------------------------
    def reached(self, turn: int | None) -> bool:
        if turn is None:
            return True
        return self.highest_turn is not None and self.highest_turn >= turn

    async def wait_for(self, turn: int | None,
                       final: bool = False) -> float:
        """Block until the viewer reaches `turn` (or the battle ends, for a
        final beat). Returns seconds actually held. Never waits longer than
        max_hold: a closed broadcast page must not mute the commentary."""
        if turn is None and not final:
            return 0.0
        start = time.monotonic()
        while True:
            if final and self.ended:
                break
            if not final and self.reached(turn):
                break
            waited = time.monotonic() - start
            if waited >= self.max_hold:
                self.timeouts += 1
                print(f"pts: hold timed out after {waited:.1f}s "
                      f"(turn {turn}, viewer at {self.highest_turn}) — "
                      f"publishing anyway", flush=True)
                break
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._pulse.wait(),
                    timeout=min(1.0, self.max_hold - waited))
        held = time.monotonic() - start
        if held > 0.05:
            self.holds += 1
            self.held_seconds += held
        return held

    def summary(self) -> str:
        if not self.holds:
            return (f"pts: no holds (connected={self.connected}, "
                    f"viewer reached turn {self.highest_turn})")
        return (f"pts: held {self.holds} beat(s) for "
                f"{self.held_seconds:.1f}s total, mean "
                f"{self.held_seconds / self.holds:.1f}s, "
                f"{self.timeouts} timed out")
