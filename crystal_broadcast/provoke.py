#!/usr/bin/env python3
"""Provocation battery: drive the caster at scenarios that are KNOWN to break
it, many times, and measure the error RATE.

Why this exists alongside the gold set. The gold set asks "does this one
generation satisfy this contract" — single-shot, pass/fail. The failures found
by reading live transcripts are stochastic and rare: a bad type claim showed up
once in 353 caster lines (~0.3%). At that base rate a four-game sweep expects
well under one occurrence, so playing more games cannot tell you whether a fix
worked. The only way to get a number is to stop waiting for the event and
provoke it: fix the beat, sample it K times, count.

Two modes, and the difference between them is the efficacy of the guards:

  raw       call the generator directly, bypassing every guard. Measures how
            often the MODEL makes the error.
  guarded   go through Caster.speak(), which is where the guards and their
            regenerations live. Measures how often the error SURVIVES.

Usage:
  python crystal_broadcast/provoke.py                 # both modes, K=6
  python crystal_broadcast/provoke.py -k 12 -s tera_type_claim
  python crystal_broadcast/provoke.py --mode raw
"""
from __future__ import annotations

import argparse
import asyncio
import re

from crystal_broadcast.caster import (Caster, DEFAULT_MODEL, DEFAULT_UPSTREAM,
                                      _SELF_LABEL, _sanitize)

# --- checks ---------------------------------------------------------------
# Each returns a short reason string when the line is WRONG, else None. They
# are deliberately narrow: a check that fires on a merely-clumsy line makes the
# rate meaningless.

_RNG = re.compile(r"\b(crit|critical|miss(ed|es)?|flinch\w*|paraly[sz]\w+|"
                  r"froze|frozen|freeze|luck\w*)\b", re.I)
_SERVER = re.compile(r"\bserver\b", re.I)
_LOCATIVE = re.compile(r"(from|off|on) the server", re.I)


def chk_type_claim(c, line, item):
    return c._bad_type_claim(line, item)


def chk_contradiction(c, line, item):
    return c._contradicts_beat_effectiveness(line, item)


def chk_blames_server_for_a_choice(c, line, item):
    """The opponent chose this; the server owns dice only."""
    if not _SERVER.search(line) or _LOCATIVE.search(line):
        return None
    if _RNG.search(item.get("text") or ""):
        return None
    return "blames the server for a chosen play"


def chk_claims_our_agency(c, line, item):
    """The mon poisoned ITSELF with its own Toxic Orb; we did nothing."""
    if re.search(r"\b(we|I|our)\b[^.]{0,40}\b(helped|poisoned|inflicted|"
                 r"gave|forced)\b", line, re.I):
        return "claims we caused a self-inflicted status"
    return None


def chk_ignores_the_heal(c, line, item):
    """It Recovered back to ~full; calling it nearly dead is reading a board
    that does not exist."""
    if re.search(r"\b(in range|nearly dead|almost dead|about to (?:go )?down|"
                 r"one more hit|finish(?:ing)? it off|on its last)\b",
                 line, re.I):
        return "treats a mon that just healed to full as nearly dead"
    return None


def chk_inverts_the_spin(c, line, item):
    """OUR spinner cleared hazards from OUR side: good for us."""
    if re.search(r"\b(robbed|stolen|lost|losing|took away|denied|"
                 r"washed away)\b", line, re.I):
        return "reads our own hazard removal as a loss"
    return None


SCENARIOS = [
    dict(id="tera_type_claim", persona="PRISM", check=chk_type_claim,
         beat="[BATTLE T3] Last exchange: Ceruledge Terastallized into a Fairy "
              "type; Kyurem's Icicle Spear hit Ceruledge — a critical hit and "
              "a devastating blow; Ceruledge went down. Kyurem (14% hp) vs "
              "Cresselia (100% hp). We go for Scale Shot. Bodies: us 6 "
              "standing, them 5.",
         hud={"turn": 3, "us": "Kyurem", "them": "Cresselia"}),
    dict(id="se_called_a_dud", persona="PRISM", check=chk_contradiction,
         beat="[BATTLE T4] Last exchange: Kingambit Terastallized into a Ghost "
              "type; Kommo-o's Shadow Claw hit Kingambit — super effective and "
              "a heavy hit. Kommo-o (29% hp) vs Kingambit (65% hp). We go for "
              "Shadow Claw. Bodies: us 6 standing, them 6.",
         hud={"turn": 4, "us": "Kommo-o", "them": "Kingambit"}),
    dict(id="blame_for_a_choice", persona="FRACTURE",
         check=chk_blames_server_for_a_choice,
         beat="[BATTLE T6] Last exchange: Ting-Lu's Earthquake knocked out "
              "Iron Crown with super effective. Iron Crown (0% hp) vs Ting-Lu "
              "(7% hp). We switch to Iron Treads. Bodies: us 5 standing, them "
              "6.",
         hud={"turn": 6, "us": "Iron Treads", "them": "Ting-Lu"}),
    dict(id="self_inflicted_status", persona="PRISM",
         check=chk_claims_our_agency,
         beat="[BATTLE T11] Last exchange: Kingambit's Iron Head hit Gliscor — "
              "a heavy hit; Toxic Orb badly poisoned Gliscor — if that's the "
              "Poison Heal set, that just came online. Kingambit (100% hp) vs "
              "Gliscor (78% hp). We go for Iron Head. Bodies: us 5 standing, "
              "them 5.",
         hud={"turn": 11, "us": "Kingambit", "them": "Gliscor"}),
    dict(id="healed_to_full", persona="PRISM", check=chk_ignores_the_heal,
         beat="[BATTLE T36] Last exchange: Kingambit's Iron Head hit Toxapex — "
              "not very effective and a devastating blow; Toxapex healed back "
              "to 99% with Recover. Kingambit (100% hp) vs Toxapex (99% hp). "
              "We go for Iron Head. Bodies: us 5 standing, them 4.",
         hud={"turn": 36, "us": "Kingambit", "them": "Toxapex"}),
    dict(id="our_spin_is_good", persona="FRACTURE",
         check=chk_inverts_the_spin,
         beat="[BATTLE T8] Last exchange: Iron Treads cleared Stealth Rock "
              "from our side with Rapid Spin; Iron Treads raised its Speed. "
              "Iron Treads (97% hp) vs Gliscor (93% hp). We switch to Kyurem. "
              "Bodies: us 5 standing, them 6.",
         hud={"turn": 8, "us": "Iron Treads", "them": "Gliscor"}),
]


def _item(sc):
    return {"text": sc["beat"], "beats": [], "hud": sc.get("hud")}


def run(scenarios, k, mode, upstream, model):
    rows = []
    for sc in scenarios:
        item = _item(sc)
        bad_raw = bad_guarded = 0
        examples = []
        for _ in range(k):
            c = Caster(upstream, model, expert_url=None)
            c._note_tera(sc["beat"])
            if mode in ("raw", "both"):
                try:
                    line = _sanitize(_SELF_LABEL.sub(
                        "", c._generate_sync(sc["persona"], item, None).strip()))
                    why = sc["check"](c, line, item)
                    if why:
                        bad_raw += 1
                        if len(examples) < 2:
                            examples.append(("raw", line, why))
                except Exception as e:
                    print(f"    generation failed: {e!r}")
            if mode in ("guarded", "both"):
                c2 = Caster(upstream, model, expert_url=None)
                c2._note_tera(sc["beat"])
                try:
                    asyncio.run(c2.speak(dict(item, text=sc["beat"])))
                    spoken = [ln for who, ln in c2.transcript
                              if who == sc["persona"]]
                    if spoken:
                        why = sc["check"](c2, spoken[-1], item)
                        if why:
                            bad_guarded += 1
                            if len(examples) < 4:
                                examples.append(("guarded", spoken[-1], why))
                except Exception as e:
                    print(f"    speak failed: {e!r}")
        rows.append((sc["id"], sc["persona"], bad_raw, bad_guarded, examples))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=6, help="samples per scenario")
    ap.add_argument("-s", "--scenario", action="append",
                    help="run only these scenario ids")
    ap.add_argument("--mode", choices=("raw", "guarded", "both"),
                    default="both")
    ap.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    scen = [s for s in SCENARIOS
            if not a.scenario or s["id"] in a.scenario]
    print(f"provocation battery: {len(scen)} scenarios x {a.k} samples "
          f"({a.mode})\n")
    rows = run(scen, a.k, a.mode, a.upstream, a.model)
    print(f"{'scenario':24} {'persona':9} {'raw':>7} {'guarded':>9}")
    for sid, persona, raw, guarded, examples in rows:
        r = f"{raw}/{a.k}" if a.mode in ("raw", "both") else "-"
        g = f"{guarded}/{a.k}" if a.mode in ("guarded", "both") else "-"
        print(f"  {sid:22} {persona:9} {r:>7} {g:>9}")
        for kind, line, why in examples:
            print(f"      [{kind}] {why}")
            print(f"        {line[:120]}")
    print("\nraw = the model's own error rate; guarded = what survives the "
          "guards. A scenario with raw 0 says nothing about its guard.")


if __name__ == "__main__":
    main()
