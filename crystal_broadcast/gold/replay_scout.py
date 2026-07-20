#!/usr/bin/env python3
"""Scout real replays for gold-set-pinnable moments.

Runs the ProtocolScanner over replays and prints every event of the
requested types with its replay id, turn, side (from p1's seat), and
prose — the shopping list for pinning gold entries to real games.

Usage:
  .venv/bin/python showdown/gold/replay_scout.py --types status_applied \
      --limit 200 [--dir showdown/replays/gen9ou] [--grep Will-O-Wisp]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from showdown.beat_director import ProtocolScanner
from showdown.commentary_eval import load_replay_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path,
                    default=Path("showdown/replays/gen9ou"))
    ap.add_argument("--types", required=True,
                    help="comma-separated event types")
    ap.add_argument("--grep", default=None,
                    help="only print events whose prose contains this")
    ap.add_argument("--limit", type=int, default=100,
                    help="replays to scan")
    ap.add_argument("--max-hits", type=int, default=40)
    args = ap.parse_args()

    want = set(args.types.split(","))
    hits = 0
    for path in sorted(args.dir.glob("*.json"))[:args.limit]:
        scanner = ProtocolScanner()
        try:
            blocks = load_replay_blocks(path)
        except Exception:
            continue
        for turn, batch in blocks:
            for ev in scanner.scan(batch, "p1"):
                if ev.type not in want:
                    continue
                if args.grep and args.grep.lower() not in ev.prose.lower():
                    continue
                print(f"{path.name} T{turn:<3} {ev.type:<16} "
                      f"side={ev.side or '-':<4} {ev.prose}")
                hits += 1
                if hits >= args.max_hits:
                    return


if __name__ == "__main__":
    main()
