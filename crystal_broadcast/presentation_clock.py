#!/usr/bin/env python3
"""Presentation-clock collector: the viewer's timeline, measured.

broadcast_clock.js (injected into the self-hosted client by serve_client.py)
reports both ends of every protocol line:

    queued     the line arrived from the sim server
    presented  the client started ANIMATING it for the viewer

This process receives those, pairs them by (battle id, queue index), and turns
them into the number the delay buffer needs: how far BEHIND the server the
viewer actually is, per line and per turn.

WHY IT MATTERS: that lag is not a constant. A six-event turn with a KO, weather
chip and a Tera animates far longer than a two-move turn, so commentary paced
off the server's clock drifts against the picture. `--airi-turn-pace` papered a
fixed offset over a fixed offset and measured zero drift only because the demo
turns were uniform. The spread reported in the summary below is the evidence
for (or against) needing PTS scheduling at all — if per-turn lag were actually
constant, a fixed pace would be fine and the buffer would be over-engineering.

Doubles as the feed a PTS scheduler subscribes to: every event is rebroadcast
to any other connected peer, so a consumer just connects and listens.

Run:  python crystal_broadcast/presentation_clock.py [--port 8132] [--log FILE]
Then load the broadcast composite; the hook connects on its own and retries
if this process starts later.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import websockets

DEFAULT_PORT = 8132
DEFAULT_LOG = "presentation_clock.jsonl"


def _pct(values, q):
    """Nearest-rank percentile; avoids numpy for one number."""
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(q * (len(ordered) - 1))))
    return ordered[k]


class ClockCollector:
    def __init__(self, log_path: Path | None):
        self.log = log_path.open("a") if log_path else None
        # (battle_id, queue_index) -> arrival timestamp (ms)
        self._queued: dict[tuple, int] = {}
        # battle_id -> [lag_ms] for every paired line
        self.line_lags: dict[str, list] = defaultdict(list)
        # battle_id -> [(turn, lag_ms)] for |turn|N lines only
        self.turn_lags: dict[str, list] = defaultdict(list)
        self.unpaired = 0
        self.hooked = False

    def ingest(self, ev: dict):
        if self.log:
            self.log.write(json.dumps(ev) + "\n")
            self.log.flush()

        kind = ev.get("kind")
        if kind == "hooked":
            self.hooked = True
            print("clock: client hook attached", flush=True)
            return

        bid, idx, t = ev.get("id"), ev.get("idx"), ev.get("t")
        if bid is None or idx is None or t is None:
            return

        if kind == "queued":
            # keep the FIRST arrival: a line's index is stable, and re-adds
            # (rejoin, replay) would otherwise overwrite the real arrival
            self._queued.setdefault((bid, idx), t)
            return

        if kind != "presented":
            return

        arrived = self._queued.pop((bid, idx), None)
        if arrived is None:
            # presented without a recorded arrival: backlog replayed before we
            # connected, or a preempt path. Not an error, just unmeasurable.
            self.unpaired += 1
            return

        lag = t - arrived
        self.line_lags[bid].append(lag)

        line = ev.get("line") or ""
        if line.startswith("|turn|"):
            try:
                turn = int(line.split("|")[2])
            except (IndexError, ValueError):
                turn = ev.get("turn")
            self.turn_lags[bid].append((turn, lag))
            print(f"clock: turn {turn} presented {lag/1000:.1f}s "
                  f"after the server sent it", flush=True)

        if line.startswith("|win|") or line.startswith("|tie"):
            self.summary(bid)

    def summary(self, only: str | None = None):
        ids = [only] if only else sorted(
            set(self.line_lags) | set(self.turn_lags))
        if not ids:
            print("clock: nothing measured "
                  f"(hook attached: {self.hooked}, unpaired: {self.unpaired})",
                  flush=True)
            return

        for bid in ids:
            turns = self.turn_lags.get(bid) or []
            lines = self.line_lags.get(bid) or []
            print(f"\n=== presentation clock: {bid} ===", flush=True)
            print(f"  lines paired: {len(lines)}  (unpaired: {self.unpaired})",
                  flush=True)
            if lines:
                print(f"  per-line lag  median {_pct(lines,.5)/1000:.2f}s  "
                      f"p90 {_pct(lines,.9)/1000:.2f}s  "
                      f"max {max(lines)/1000:.2f}s", flush=True)
            if len(turns) >= 2:
                vals = [lag for _, lag in turns]
                spread = max(vals) - min(vals)
                print(f"  turns measured: {len(turns)}", flush=True)
                print(f"  per-turn lag  min {min(vals)/1000:.2f}s  "
                      f"median {_pct(vals,.5)/1000:.2f}s  "
                      f"max {max(vals)/1000:.2f}s", flush=True)
                print(f"  SPREAD {spread/1000:.2f}s "
                      f"(stdev {statistics.pstdev(vals)/1000:.2f}s)",
                      flush=True)
                print("  ^ this is the number that decides PTS scheduling: a "
                      "fixed --airi-turn-pace can only ever be correct if the "
                      "spread is near zero.", flush=True)
                worst = max(turns, key=lambda p: p[1])
                best = min(turns, key=lambda p: p[1])
                print(f"  slowest turn {worst[0]} at {worst[1]/1000:.2f}s, "
                      f"fastest turn {best[0]} at {best[1]/1000:.2f}s",
                      flush=True)
            elif turns:
                print(f"  only {len(turns)} turn measured; need >=2 for spread",
                      flush=True)

    def close(self):
        if self.log:
            self.log.close()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--log", default=DEFAULT_LOG,
                    help="JSONL of raw events; '' to disable")
    args = ap.parse_args()

    collector = ClockCollector(Path(args.log) if args.log else None)
    peers: set = set()

    async def handler(conn):
        peers.add(conn)
        try:
            async for msg in conn:
                try:
                    ev = json.loads(msg)
                except Exception:
                    continue
                collector.ingest(ev)
                # rebroadcast so a PTS consumer can just subscribe here
                for p in list(peers):
                    if p is not conn:
                        try:
                            await p.send(msg)
                        except Exception:
                            pass
        except websockets.ConnectionClosed:
            pass
        finally:
            peers.discard(conn)

    stop = asyncio.get_running_loop().create_future()

    def _bye():
        collector.summary()
        collector.close()
        if not stop.done():
            stop.set_result(None)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _bye)

    async with websockets.serve(handler, "127.0.0.1", args.port):
        print(f"presentation clock: ws://127.0.0.1:{args.port}/ "
              f"(log {args.log or 'off'})", flush=True)
        await stop


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
