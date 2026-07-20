#!/usr/bin/env python3
"""FRACTURE's Book of Grudges: a persistent record of which opponent
Pokemon have KO'd our mons across past games.

The caster injects a grudge line for the current opponent active so
FRACTURE can only cite REAL, statistically-justified vendettas — the
character contract forbids inventing grudges, and this is what makes the
paranoia grounded (the whole joke). Sourced from the same faint
attributions loss_trace already parses: every faint records the killer
species, so counting killers-of-our-mons by species IS the ledger.

The ledger is a plain JSON file, additive across games. Rebuild or top it
up from replay logs:

  .venv/bin/python showdown/grudge_ledger.py build \
      showdown/replays/gen9ou/*.json --our CBGen9 -o showdown/grudges.json
  .venv/bin/python showdown/grudge_ledger.py show showdown/grudges.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# canonical species id for merging nickname/forme spellings into one grudge
from showdown.name_mapping import _normalize

_PLAYER_RE = re.compile(r"\|player\|(p[12])\|([^|]*)\|")


def parse_replay_games(paths: list[Path], our_name: str) -> list[dict]:
    """Parse downloaded replay JSONs (the `log` field's raw protocol) into
    the faint-attribution shape merge_faints wants — but resolved to
    SPECIES, not nicknames. A grudge is against the Pokemon, not the pet
    name a laddering opponent gave it, so nickname->species is tracked from
    switch lines exactly like the director's scanner does.

    Faint killer = the mover of the most recent move by the OTHER side
    (matches loss_trace's last-hit attribution)."""
    games = []
    our_norm = _normalize(our_name)
    for path in paths:
        try:
            log = json.loads(path.read_text())["log"]
        except Exception:
            continue
        players: dict[str, str] = {}
        species: dict[str, str] = {}   # position 'p1a' -> species
        last_move: dict[str, str] = {}  # side 'p1' -> mover species
        faints: list = []
        winner = None
        for line in log.split("\n"):
            sm = line.split("|")
            if len(sm) < 2:
                continue
            tag = sm[1]
            if tag == "player" and len(sm) > 3:
                players[sm[2]] = sm[3]
            elif tag in ("switch", "drag", "replace") and len(sm) > 3:
                species[sm[2].split(":")[0]] = sm[3].split(",")[0]
            elif tag == "move" and len(sm) > 3:
                pos = sm[2].split(":")[0]
                last_move[pos[:2]] = species.get(pos, sm[2].split(": ")[-1])
            elif tag == "faint" and len(sm) > 2:
                pos = sm[2].split(":")[0]
                side = pos[:2]
                other = "p2" if side == "p1" else "p1"
                victim = species.get(pos, sm[2].split(": ")[-1])
                faints.append((side, victim, 0,
                               (last_move.get(other), None)))
            elif tag == "win" and len(sm) > 2:
                winner = sm[2].strip()
        our_role = next((r for r, n in players.items()
                         if _normalize(n) == our_norm), None)
        if our_role is None or winner is None:
            continue
        games.append({"our_role": our_role, "faints": faints,
                      "winner": winner, "we_won": winner == players[our_role]})
    return games


def merge_faints(ledger: dict, games: list[dict]) -> dict:
    """Fold games' faint attributions into `ledger` (mutated + returned).

    ledger schema: { species_id: {"name": display, "kos": int,
                                  "victims": {victim: count},
                                  "moves": {move: count}} }
    Only OUR mons' deaths count — the grudge is against what kills us.
    Games use loss_trace's shape: faints = [(side, victim, turn,
    (killer_mon, killer_move))], plus our_role."""
    for g in games:
        our = g.get("our_role")
        if not our:
            continue
        for side, victim, _turn, (kmon, kmove) in g["faints"]:
            if side != our or not kmon or kmon == "?":
                continue
            key = _normalize(kmon)
            rec = ledger.setdefault(
                key, {"name": kmon, "kos": 0, "victims": {}, "moves": {}})
            rec["name"] = kmon  # keep the freshest display spelling
            rec["kos"] += 1
            rec["victims"][victim] = rec["victims"].get(victim, 0) + 1
            if kmove and kmove != "?":
                rec["moves"][kmove] = rec["moves"].get(kmove, 0) + 1
    return ledger


class GrudgeLedger:
    """Load a grudge file and answer 'what's the beef with this mon'. A
    missing/empty file yields no grudges — graceful, never raises."""

    def __init__(self, ledger: dict | None = None):
        self.ledger = ledger or {}

    @classmethod
    def load(cls, path: str | Path | None):
        if not path:
            return cls({})
        p = Path(path)
        if not p.exists():
            return cls({})
        try:
            return cls(json.loads(p.read_text()))
        except Exception:
            return cls({})

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.ledger, indent=2))

    def record(self, games: list[dict]):
        merge_faints(self.ledger, games)

    def grudge_for(self, species_display: str | None,
                   min_kos: int = 2) -> str | None:
        """A one-line grudge cue for the caster to inject, or None when the
        record is too thin to justify a vendetta. Threshold keeps a single
        unlucky KO from becoming lore."""
        if not species_display:
            return None
        rec = self.ledger.get(_normalize(species_display))
        if not rec or rec["kos"] < min_kos:
            return None
        victims = Counter(rec["victims"])
        top = victims.most_common(3)
        vic_str = ", ".join(f"{v} ({c}x)" if c > 1 else v for v, c in top)
        move = None
        if rec["moves"]:
            move = Counter(rec["moves"]).most_common(1)[0][0]
        line = (f"GRUDGE LEDGER — {rec['name']}: has KO'd our mons "
                f"{rec['kos']} times across past games "
                f"(victims: {vic_str}"
                + (f"; usually with {move}" if move else "") + "). "
                f"This is a real, recorded vendetta you MAY cite.")
        return line


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build/extend a ledger from replays")
    b.add_argument("replays", nargs="+")
    b.add_argument("--our", default="CBGen9",
                   help="our player name in the logs")
    b.add_argument("-o", "--out", required=True)

    s = sub.add_parser("show", help="print a ledger, worst offenders first")
    s.add_argument("ledger")
    args = ap.parse_args()

    if args.cmd == "build":
        games = parse_replay_games([Path(p) for p in args.replays], args.our)
        led = GrudgeLedger.load(args.out)
        led.record(games)
        led.save(args.out)
        print(f"ledger: {len(games)} games folded, "
              f"{len(led.ledger)} species with grudges -> {args.out}")
    elif args.cmd == "show":
        led = GrudgeLedger.load(args.ledger)
        rows = sorted(led.ledger.values(), key=lambda r: -r["kos"])
        for r in rows:
            vic = ", ".join(f"{v}x{c}" for v, c
                            in Counter(r["victims"]).most_common(4))
            print(f"{r['kos']:>3}  {r['name']:<16} {vic}")


if __name__ == "__main__":
    main()
