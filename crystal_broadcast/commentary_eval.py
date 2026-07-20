#!/usr/bin/env python3
"""Commentary gold-set eval runner.

Drives the SAME classes the live broadcast runs — ProtocolScanner ->
Director (and, at caster level, the Caster's persona generation) — against
the executable gold set (showdown/gold/commentary_gold.yaml), fully
offline. Philosophy ported from grounded-rag's run_eval: machine-checkable
assertions, per-dimension report, nonzero exit on any miss.

Two levels:
  --level director   (default) deterministic, no LLM: beat detection,
                     attribution (persona/priority/register/handoff),
                     faithfulness of the composed beat text, silence
                     precision. Cheap enough for a pre-push hook.
  --level caster     everything above PLUS real generations through the
                     caster prompts: must_mention on the spoken lines,
                     per-persona checks on dual beats, contract forbids
                     (plain strings, statistics_or_citations,
                     ungrounded_entity). Needs the ollama proxy up;
                     generation wobbles — reread miss lines before calling
                     a fail real (the grounded-rag rule).

Usage:
  .venv/bin/python showdown/commentary_eval.py
  .venv/bin/python showdown/commentary_eval.py --level caster -e gc-0017
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from showdown.beat_director import Director, ProtocolScanner, TurnContext

GOLD_DEFAULT = Path(__file__).parent / "gold" / "commentary_gold.yaml"

_STATS_RE = re.compile(r"\d+(?:\.\d+)?\s*%|\[\d+\]|\bpercent\b", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


class GameData:
    """Lazy GenData wrappers: stats for the director's burn split and the
    entity index for the ungrounded-entity contract check."""

    def __init__(self):
        self._gen = None

    @property
    def gen(self):
        if self._gen is None:
            from poke_env.data import GenData
            self._gen = GenData.from_gen(9)
        return self._gen

    def stats(self, display_name: str):
        entry = self.gen.pokedex.get(_norm(display_name))
        if entry and "baseStats" in entry:
            bs = entry["baseStats"]
            return bs.get("atk", 0), bs.get("spa", 0)
        return None

    def entity_names(self) -> list[str]:
        names = [e["name"] for e in self.gen.pokedex.values() if "name" in e]
        names += [e["name"] for e in self.gen.moves.values() if "name" in e]
        # multi-word first so "Iron Valiant" wins over "Iron"
        return sorted(set(names), key=len, reverse=True)


DATA = GameData()


def _mention_ok(text: str, spec: str) -> bool:
    """Case-insensitive; 'a|b' means any alternative suffices."""
    low = text.lower()
    return any(alt.strip().lower() in low for alt in spec.split("|"))


def _ctx(raw: dict) -> TurnContext:
    kw = dict(raw)
    for key in ("ours_fainted", "theirs_fainted"):
        if key in kw:
            kw[key] = frozenset(kw[key])
    return TurnContext(**kw)


def run_director(entry: dict) -> tuple[list, object, list[str]]:
    """Feed the fixture through fresh scanner+director; return (decisions,
    final decision, misses)."""
    fx = entry["fixture"]
    scanner = ProtocolScanner()
    director = Director(stats_fn=DATA.stats)
    for batch in fx.get("batches", []):
        director.observe(scanner.scan(batch, fx.get("role")))
    decisions = [director.decide(_ctx(c)) for c in fx["ctx"]]
    final = decisions[-1]
    misses = []

    if entry.get("silence"):
        if not final.silence or final.text is not None:
            misses.append(f"expected silence, got: {final.text!r}")
        return decisions, final, misses

    if final.text is None:
        misses.append("decision was silent, expected a beat")
        return decisions, final, misses

    want_beat = entry.get("beat")
    if want_beat:
        matching = [b for b in final.beats if b.beat == want_beat]
        if not matching:
            misses.append(
                f"no '{want_beat}' beat fired "
                f"(got: {[b.beat for b in final.beats]})")
        else:
            b = matching[0]
            for field in ("persona", "priority", "register", "handoff"):
                want = entry.get(field)
                if want is not None and getattr(b, field) != want:
                    misses.append(f"{field}: want {want!r}, "
                                  f"got {getattr(b, field)!r}")
    for spec in entry.get("must_mention", []):
        if not _mention_ok(final.text, spec):
            misses.append(f"beat text missing {spec!r}")
    return decisions, final, misses


def run_caster(entry: dict, final, upstream: str, model: str) -> list[str]:
    """Generate real lines for the final decision and check the spoken
    layer: who spoke, what they said, what they must never say."""
    from showdown.caster import Caster

    caster = Caster(upstream, model)
    spoken: list[tuple[str, str]] = []

    async def collect(beat_text, persona, line, hud):
        spoken.append((persona, line))

    caster.publish = collect
    item = {"text": final.text, "beats": [asdict(b) for b in final.beats],
            "hud": {"turn": entry["fixture"]["ctx"][-1].get("turn")}}
    asyncio.run(caster.speak(item))
    misses = []
    by = {p: ln for p, ln in spoken}

    # subset semantics: the entry's persona must have spoken; extra voices
    # are legitimate when other beats share the decision (a KO's gremlin
    # scream alongside the analyst's contradiction call)
    persona = entry.get("persona")
    want = {"gremlin": ["FRACTURE"], "analyst": ["PRISM"],
            "both": ["FRACTURE", "PRISM"]}.get(persona)
    if persona == "either":
        if not spoken:
            misses.append("no voice spoke")
    elif want is not None and not set(want) <= set(by):
        misses.append(f"speakers: want {want} to speak, got {sorted(by)}")

    all_lines = " ".join(ln for _, ln in spoken)
    for spec in entry.get("must_mention", []):
        if not _mention_ok(all_lines, spec):
            misses.append(f"spoken lines missing {spec!r}")
    for suffix, who in (("gremlin", "FRACTURE"), ("analyst", "PRISM")):
        for spec in entry.get(f"must_mention_{suffix}", []):
            if who not in by:
                misses.append(f"{who} never spoke ({spec!r} unchecked)")
            elif not _mention_ok(by[who], spec):
                misses.append(f"{who} line missing {spec!r}: {by[who]!r}")

    allowed_text = (final.text or "") + " " + " ".join(
        entry.get("allowed_entities", []))
    for forbid in entry.get("forbid", []):
        if forbid == "statistics_or_citations":
            for who, ln in spoken:
                if who == "FRACTURE" and _STATS_RE.search(ln):
                    misses.append(f"FRACTURE quoted a statistic: {ln!r}")
        elif forbid == "ungrounded_entity":
            pass  # checked globally below
        else:
            for who, ln in spoken:
                if forbid.lower() in ln.lower():
                    misses.append(f"forbidden {forbid!r} in {who}: {ln!r}")

    # global ungrounded-entity check (the grounded-rag 0-ungrounded-mentions
    # metric, ported): any known species/move named in a line must appear in
    # the beat text. Lowercase occurrences are skipped so common-word moves
    # ("rest", "protect") in prose don't false-flag.
    allowed_low = allowed_text.lower()
    for who, ln in spoken:
        low = ln.lower()
        for name in DATA.entity_names():
            if len(name) < 4:
                continue
            nl = name.lower()
            if nl in low and nl not in allowed_low and name in ln:
                misses.append(
                    f"ungrounded entity {name!r} in {who}: {ln!r}")
                break
    return misses


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--gold", type=Path, default=GOLD_DEFAULT)
    ap.add_argument("--level", choices=("director", "caster"),
                    default="director")
    ap.add_argument("-e", "--entries", default=None,
                    help="comma-separated gc-ids to run (default: all)")
    ap.add_argument("--upstream", default="http://127.0.0.1:11435")
    ap.add_argument("--model", default="gemma4:26b-a4b-it-q4_K_M")
    args = ap.parse_args()

    entries = yaml.safe_load(args.gold.read_text())
    if args.entries:
        keep = set(args.entries.split(","))
        entries = [e for e in entries if e["id"] in keep]
    if not entries:
        raise SystemExit("no entries selected")

    dims = {"beat/attribution": [0, 0], "faithfulness(text)": [0, 0],
            "silence": [0, 0]}
    if args.level == "caster":
        dims["spoken lines"] = [0, 0]
    failed = []

    for entry in entries:
        _, final, misses = run_director(entry)
        dim = "silence" if entry.get("silence") else "beat/attribution"
        text_misses = [m for m in misses if m.startswith("beat text")]
        attr_misses = [m for m in misses if not m.startswith("beat text")]
        dims[dim][1] += 1
        dims[dim][0] += 0 if attr_misses else 1
        if not entry.get("silence"):
            dims["faithfulness(text)"][1] += 1
            dims["faithfulness(text)"][0] += 0 if text_misses else 1

        if args.level == "caster" and not misses and not entry.get("silence"):
            c_misses = run_caster(entry, final, args.upstream, args.model)
            dims["spoken lines"][1] += 1
            dims["spoken lines"][0] += 0 if c_misses else 1
            misses += c_misses

        status = "ok " if not misses else "MISS"
        print(f"{status} {entry['id']}"
              + (f"  ({'; '.join(misses)})" if misses else ""))
        if misses:
            failed.append(entry["id"])

    print()
    for dim, (hit, total) in dims.items():
        if total:
            print(f"{dim:<20} {hit}/{total}")
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        sys.exit(1)
    print("\nall entries green")


if __name__ == "__main__":
    main()
