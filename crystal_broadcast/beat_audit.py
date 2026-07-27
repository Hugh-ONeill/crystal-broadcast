#!/usr/bin/env python3
"""Turn-keyed audit: what HAPPENED vs what the director SAID happened.

Structure is turn -> the beats spawned from it, because that is the only view
in which misattribution is visible. A beat read on its own sounds fine; it is
only against the protocol of its own turn that you can see it credited the
wrong actor, or blended our action into the opponent's.

Two failure classes this is built to expose, both observed live 2026-07-27:
  * ACTOR      the beat names the wrong Pokemon as having done a thing
  * AGENCY     the beat is ambiguous about WHICH SIDE caused it, so the caster
               invents the framing. Real case: our Gholdengo used Trick to put
               a Choice Scarf on their Garganacl, and FRACTURE narrated it as
               the opponent having set her up. Also: an ability that fires off
               OUR contact move (Flame Body burning us) read as the opponent
               "using" the ability.
Note that neither is a caster hallucination in the usual sense — the facts are
real, the attribution is not. String-based gold checks cannot see this, which
is why actor-misattribution has stayed a known blind spot.

Input is any protocol capture:
  --clock FILE   presentation_clock.jsonl (broadcast_clock.js records every
                 line the client received, so a broadcast run is also a
                 protocol capture — no extra plumbing needed)
  --replay FILE  a Showdown replay log

Run:
  python showdown/beat_audit.py --clock /tmp/prism-demo/clock_clean.jsonl \
      --role p1 --turns 26-29
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from showdown.beat_director import ProtocolScanner, classify  # noqa: E402


def lines_from_clock(path: Path, battle: str | None = None) -> list[str]:
    """Protocol lines in arrival order. 'queued' is when the line reached the
    client, so it preserves server order; 'presented' would be animation order
    and is deduplicated differently.

    A long-lived collector logs SEVERAL battles to one file, so without a
    battle filter the turns of two games interleave and the audit is
    nonsense. Defaults to the LAST battle seen."""
    out = []
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        if ev.get("kind") == "queued" and ev.get("line"):
            out.append((ev.get("id"), ev["line"]))
    if not out:
        return []
    target = battle or out[-1][0]
    return [ln for bid, ln in out if bid == target]


def lines_from_replay(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln.startswith("|")]


def blocks_by_turn(lines: list[str]):
    """[(turn, [split_message, ...])] — scan() wants messages already split on
    '|', the way poke-env hands them over."""
    blocks, turn, cur = [], 0, []
    for ln in lines:
        if ln.startswith("|turn|"):
            if cur:
                blocks.append((turn, cur))
            try:
                turn = int(ln.split("|")[2])
            except (IndexError, ValueError):
                pass
            cur = []
        cur.append(ln.split("|"))
    if cur:
        blocks.append((turn, cur))
    return blocks


def parse_range(spec: str | None):
    if not spec:
        return None
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return range(int(lo), int(hi) + 1)
    return range(int(spec), int(spec) + 1)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--clock", help="presentation_clock.jsonl")
    src.add_argument("--replay", help="Showdown replay log")
    ap.add_argument("--role", default="p1", help="which side is US (p1/p2)")
    ap.add_argument("--battle", default=None,
                    help="battle id in a multi-battle clock log "
                         "(default: the last one seen)")
    ap.add_argument("--turns", help="e.g. 27 or 26-29 (default: all)")
    ap.add_argument("--only-beats", action="store_true",
                    help="skip turns that produced no beat")
    args = ap.parse_args()

    lines = (lines_from_clock(Path(args.clock), args.battle) if args.clock
             else lines_from_replay(Path(args.replay)))
    if not lines:
        raise SystemExit("no protocol lines found")

    wanted = parse_range(args.turns)
    scanner = ProtocolScanner()
    shown = 0

    # scan sequentially so the scanner keeps its species/nickname state, but
    # only PRINT the requested turns
    for turn, batch in blocks_by_turn(lines):
        events = scanner.scan(batch, args.role)
        beats = [(ev, classify(ev)) for ev in events]
        if wanted is not None and turn not in wanted:
            continue
        if args.only_beats and not any(b for _, b in beats):
            continue
        shown += 1
        print(f"\n=== T{turn} ===")
        print("  protocol:")
        for msg in batch:
            joined = "|".join(msg).rstrip()
            if joined.strip("|"):
                print(f"    {joined[:150]}")
        print("  events -> beats:")
        for ev, beat in beats:
            side = ev.side or "-"
            print(f"    [{ev.type} side={side}] {ev.prose}")
            if beat:
                print(f"        BEAT persona={beat.persona} "
                      f"register={beat.register}")
                print(f"        {beat.prose}")
            else:
                print("        (no beat)")
    if not shown:
        print("no turns matched", file=sys.stderr)


if __name__ == "__main__":
    main()
