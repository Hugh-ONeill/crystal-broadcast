#!/usr/bin/env python3
"""Caster service: the duo's voice box.

Speaks the beat protocol on ws://127.0.0.1:8131 (see caster_bridge.py for
its shape and why it looks the way it does):
  * gen9_player --airi --airi-url ws://127.0.0.1:8131/ws delivers beats
    through the stock CasterBridge (module:authenticate -> input:text);
    structured director beats + HUD ride in extra data fields alongside
    the beat text.
  * commentary_overlay / caster_bridge --watch subscribe here and receive
    output:gen-ai:chat:complete envelopes (superjson-wrapped) per finished
    line, with data.persona attached.

For each beat the caster picks the speaking persona(s) from the director's
routing (handoff order for dual beats), builds a per-persona prompt (the
contract file + a bounded duo transcript + the beat + register direction),
and generates through the ollama no-think proxy. The duo transcript is
shared, so PRISM sees FRACTURE's line before correcting it — the
correction loop is an ordered pair of generations, not a prompt prayer.

Latency policy is skip-don't-queue: one pending slot per priority class;
a newer turn beat replaces an unspoken older one. MATCH START / RESULT
always speak.

Run:  .venv/bin/python crystal_broadcast/caster.py [--port 8131]
      [--upstream http://127.0.0.1:11434] [--model ...]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import websockets

import difflib
import re

from crystal_broadcast.caster_bridge import _sanitize, _unwrap
from crystal_broadcast.grudge_ledger import GrudgeLedger
from crystal_broadcast.pts_clock import PresentationClock

# the model mimics the transcript format and prefixes its own line with a
# speaker label (sometimes stacked: "PRISM: PRISM: ..."); strip them all
_SELF_LABEL = re.compile(r"^\s*(?:(?:PRISM|FRACTURE)\s*:\s*)+", re.I)
# the same tic MID-line: take 50 T22 aired "...look stupid! FRACTURE: I'M
# LITERALLY RUNNING OUT OF OPTIONS" — the model restarts the transcript
# format inside its own line. The colon is what marks it as a label; a
# vocative ("Prism, watch this") has none and survives.
_MID_LABEL = re.compile(r"\s+(?:PRISM|FRACTURE)\s*:\s*", re.I)


def _clean(raw: str) -> str:
    """The full line hygiene pass: leading self-labels, mid-line label
    restarts, then the bridge sanitizer."""
    return _sanitize(_MID_LABEL.sub(" ", _SELF_LABEL.sub("", raw.strip())))

PERSONA_DIR = Path(__file__).parent / "personas"
DEFAULT_PORT = 8131
# Ollama direct. We used to go through ollama_nothink_proxy.py on :11435,
# which existed ONLY because AIRI would not send `reasoning_effort` on its
# /v1 calls and gemma4 defaults to thinking-on (reasoning leaked into the
# spoken line and ate the token budget). The caster is our own client, so it
# just sends the field — see _generate_sync. Needs Ollama >= 0.32, which is
# where /v1 started mapping reasoning_effort:"none" to thinking off.
DEFAULT_UPSTREAM = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma4:26b-a4b-it-q4_K_M"
DEFAULT_GRUDGES = Path(__file__).parent / "grudges.json"
DEFAULT_EXPERT = "http://127.0.0.1:8001"
# corpus id -> display name for the on-screen source chip
_CORPUS_NAME = {"bulbapedia": "Bulbapedia", "pokeapi": "PokéAPI",
                "crystal_battle": "Smogon", "smogon": "Smogon",
                "monotype": "Smogon"}

# curated "non-obvious mechanics" whose effect PRISM tends to reason wrong
# from the name alone — when one appears in a beat, pull the real effect
# from the grounded-rag expert (/retrieve) and inject it. Grows as new
# hallucinations surface (seeded from the demos: Good as Gold, Drain Punch,
# Rapid Spin). FRACTURE never gets facts — she's no-citations by contract.
_MECHANICS = frozenset({
    # abilities
    "good as gold", "poison heal", "guts", "magic guard", "levitate",
    "regenerator", "intimidate", "protosynthesis", "quark drive", "unaware",
    "prankster", "multiscale", "flash fire", "water absorb", "volt absorb",
    "storm drain", "lightning rod", "thick fat", "well-baked body",
    "purifying salt", "mold breaker", "sheer force", "magic bounce",
    "defiant", "competitive", "beads of ruin", "sword of ruin",
    "tablets of ruin", "vessel of ruin", "supreme overlord",
    "poison puppeteer", "toxic debris", "guard dog", "orichalcum pulse",
    "hadron engine", "contrary", "unburden", "weak armor", "stamina",
    "anger shell", "opportunist", "flame body", "static", "rough skin",
    "iron barbs", "quick feet", "marvel scale", "flare boost", "toxic boost",
    # moves
    "knock off", "trick", "switcheroo", "court change", "rapid spin",
    "mortal spin", "defog", "tidy up", "drain punch", "parabolic charge",
    "sucker punch", "u-turn", "volt switch", "flip turn", "teleport",
    "encore", "taunt", "destiny bond", "perish song", "whirlwind", "roar",
    "dragon tail", "circle throw", "leech seed", "curse", "psychic noise",
    "salt cure", "ceaseless edge", "stone axe", "syrup bomb", "tar shot",
    "glaive rush", "ruination", "population bomb", "gigaton hammer",
    "bitter blade", "revival blessing", "future sight", "wish", "trick room",
})

# curated mechanics keyed by their spaceless form, so an ability id resolved
# by the player ("goodasgold") maps back to the display name ("good as gold")
# _gather_facts / the expert / the citation matcher all expect
_MECH_BY_KEY = {re.sub(r"[^a-z0-9]", "", m): m for m in _MECHANICS}


def _canon_mechanic(token: str | None) -> str | None:
    """Map a raw ability/move token (id like 'goodasgold' or display like
    'Good as Gold') to its curated display name, or None if not curated."""
    if not token:
        return None
    low = token.lower()
    if low in _MECHANICS:
        return low
    return _MECH_BY_KEY.get(re.sub(r"[^a-z0-9]", "", low))


# the caption-mode residual: restating the chosen move ('the search is opting
# for X') or reciting the desk read back ('the desk read shows Y') — the
# mode-collapse the variety pass reduced but didn't kill. Deliberately narrow:
# it must NOT catch the SANCTIONED qualitative attribution the persona file
# encourages ('the search likes this line', 'it sees one line') — only the
# move-caption template, whose tell is 'opting'/'opts for'.
_CAPTION_RE = re.compile(
    r"\bis opting\b|\bopting for\b|\bopts for\b|"
    r"\bthe desk read (?:shows|says|reads)\b", re.I)

# generation knobs per persona: FRACTURE runs hot and short, PRISM cool
# and a touch longer. frequency_penalty pushes against echoing the duo
# transcript (which is in-context), the main driver of same-y lines.
_GEN = {
    "FRACTURE": {"temperature": 1.0, "max_tokens": 90,
                 "frequency_penalty": 0.4},
    "PRISM": {"temperature": 0.85, "max_tokens": 140,
              "frequency_penalty": 0.4},
}
_PERSONA_FILE = {"PRISM": "prism.txt", "FRACTURE": "fracture.txt"}

# status-synergy ability names (lowercased); a caster naming one the beat
# never flagged is inventing a mechanic — see _fabricated_synergy
_SYNERGY_ABILITIES = ("poison heal", "guts", "quick feet", "marvel scale",
                      "flare boost", "toxic boost")

# analytic angles rotated across PRISM's plain turn updates: the beat text
# is a fixed template, and a fixed task on top of it collapses him into
# caption mode ("the search is opting for X, the desk read shows Y" every
# line — measured, an entire match of it). A rotating lens changes the
# TASK per line, which changes the sentence shapes with it.
# Registers the director only ever assigns to DICE: a crit, a miss, full
# paralysis, a freeze. Everything else reached the board because a human chose
# it. FRACTURE may rage at the server for these and must blame the opponent for
# the rest — see the BLAME THE RIGHT ENEMY rule in personas/fracture.txt. Kept
# in sync with beat_director's _LUCK_REGISTERS and the crit/freeze classifiers.
_DICE_REGISTERS = frozenset({"persecution", "delight", "rejoicing"})

_PRISM_ANGLES = [
    "name the one thing that actually changed this turn",
    "say what this positions us for two or three turns out",
    "price the trade that just happened: what it cost, what it bought",
    "note what the opponent is trying to do and whether it is working",
    "one dry observation, a single short sentence",
]

# stall-repeat detection. FRACTURE reuses whatever thinking-beat image she
# landed on ("threading a needle") every few beats; a generic "vary it" rule
# can't stop it. Compare distinctive-word bigrams: a repeated IMAGE shares one
# ('threading needle'), while a recurring verb with a new object ('cooking a
# masterpiece' vs 'cooking something transcendent') does not.
_STALL_STOP = frozenset(
    "a an the and or but to of in on at i im am is are was be been this that "
    "it its my we our you your here now just literally single every any right "
    "through into over with for so as not no do dont one".split())


def _turn_of(beat: str):
    """Turn number out of a beat's '[BATTLE T14]' tag, or None (MATCH START
    has no turn). Same shape commentary_overlay.py parses."""
    m = re.search(r"\bT(\d+)\b", beat)
    return int(m.group(1)) if m else None


# A capitalised word of 5+ letters: the shape of a species name. Apostrophes
# are excluded so a possessive keeps its "'s" while the stem is corrected.
_SPECIES_TOKEN = re.compile(r"\b[A-Z][A-Za-z\-]{4,}\b")


def _fix_species_spelling(line: str, item: dict) -> str:
    """Correct near-miss species spellings against the mons actually on the
    field.

    Measured 2026-07-27: PRISM reliably wrote "Gargancl" / "Garganyl" for
    Garganacl even with the name in the beat twice AND in the on-field
    grounding block — an unusual name a 4B-active model mangles. A misspelled
    species on a broadcast lower-third is exactly the sort of thing viewers
    notice, and deterministic correction beats asking the model to try harder.

    Deliberately narrow: candidates are only the two actives, the cutoff is
    high, and multi-word species (Great Tusk, Iron Valiant) are out of scope
    because a single token can't be matched against them safely.
    """
    hud = item.get("hud") or {}
    known = [n for n in (hud.get("us"), hud.get("them"))
             if n and " " not in n]
    if not known or not line:
        return line

    # match case-INSENSITIVELY: difflib scores "DONDONZO" against "Dondozo"
    # far below the cutoff, so every shouted misspelling sailed through. That
    # silently exempted FRACTURE's whole register, which is mostly caps —
    # measured live 2026-07-28, "DONDONZO" survived twice in one game while
    # the identical title-case slip was corrected.
    canon = {k.lower(): k for k in known}

    def repl(m):
        tok = m.group(0)
        if tok in known:
            return tok
        near = difflib.get_close_matches(tok.lower(), list(canon),
                                         n=1, cutoff=0.8)
        if not near:
            return tok
        fixed = canon[near[0]]
        # keep the shout: swapping DONDONZO for Dondozo mid-yell reads as a
        # case glitch rather than a correction
        return fixed.upper() if tok.isupper() else fixed

    return _SPECIES_TOKEN.sub(repl, line)


def _content_bigrams(line: str) -> set:
    words = re.findall(r"[a-z']+", line.lower())
    content = [w for w in words if len(w) > 2 and w not in _STALL_STOP]
    return set(zip(content, content[1:]))


# director persona tag -> speaker(s). "either" resolves by priority: the
# gremlin owns fast reactions, the desk owns considered ones (the docs'
# default flow).
def _speakers(beats: list[dict], text: str) -> list[str]:
    if text.startswith("[MATCH START]"):
        # preview: analyst leads, gremlin color (template taxonomy)
        return ["PRISM", "FRACTURE"]
    if text.startswith("[RESULT]"):
        # recap handoff: gremlin celebrates/deflects first, analyst walks
        # the trace (gc-0042)
        return ["FRACTURE", "PRISM"]
    if not beats:
        return ["PRISM"]  # plain turn update: the desk narrates
    top = beats[0]
    persona = top.get("persona", "analyst")
    if persona == "both":
        order = top.get("handoff") or ["gremlin", "analyst"]
        return [{"gremlin": "FRACTURE", "analyst": "PRISM"}[p]
                for p in order]
    if persona == "none":
        return []
    if persona == "either":
        voices = [("FRACTURE" if top.get("priority") == "interrupt"
                   else "PRISM")]
    else:
        voices = [{"gremlin": "FRACTURE", "analyst": "PRISM"}[persona]]
    # when another interrupt beat belongs to the OTHER persona (a KO and a
    # desk contradiction landing together), both voices speak — gremlin
    # reacts first, the desk follows with meaning
    for b in beats[1:]:
        other = {"gremlin": "FRACTURE", "analyst": "PRISM"}.get(
            b.get("persona"))
        if (other and other not in voices
                and b.get("priority") == "interrupt"):
            voices.append(other)
    if voices == ["PRISM", "FRACTURE"]:
        voices = ["FRACTURE", "PRISM"]  # fast reaction leads
    return voices[:2]


class Caster:
    def __init__(self, upstream: str, model: str,
                 grudge_path: str | None = None,
                 expert_url: str | None = DEFAULT_EXPERT,
                 speech_budget: float | None = None,
                 duration_fn=None,
                 speech=None,
                 pts=None):
        self.upstream = upstream
        self.model = model
        # PTS scheduling: a PresentationClock, or None to publish at engine
        # time exactly as before. See pts_clock.py — the hold sits AFTER
        # generation so the ~8s of generation is spent inside the lag we
        # already have rather than added to it.
        self.pts = pts
        # Speech pacing, both off by default so behaviour is unchanged until a
        # speech layer exists. duration_fn(persona, line) -> seconds is the only
        # contract the TTS side has to meet; it is called ON THE EVENT LOOP, so
        # a synth that runs in a thread should return an already-known duration
        # (the wav length, or the 12Hz codec frame count) rather than block.
        # speech_budget is the per-beat wall-clock allowance, i.e. the beat
        # floor, and gates the SECOND voice of a handoff pair only: the first
        # speaker always gets to finish, since silence is worse than overrun.
        self.speech_budget = speech_budget
        self.duration_fn = duration_fn
        # optional voice (crystal_broadcast.speech.Speech). Absent by
        # default and fail-soft when present: audio is opt-in, and a
        # missing synth must never cost a line.
        self.speech = speech
        self.prompts = {p: (PERSONA_DIR / f).read_text()
                        for p, f in _PERSONA_FILE.items()}
        self.grudges = GrudgeLedger.load(grudge_path)
        self.expert_url = expert_url          # None disables fact injection
        self._fact_cache: dict = {}           # mechanic -> fact (abilities
        self._expert_up: bool | None = None   # /moves don't change)
        self._warm_task = None                # team-preview cache-warm task
        # per-match RAG instrument (reset at MATCH START, logged at RESULT):
        # is the expert doing anything, and is warming actually killing the
        # cold in-game round-trips it's supposed to?
        self._fact_stats: dict = {}
        self._reset_fact_stats()
        # per-match pacing instrument (reset at MATCH START, logged at RESULT).
        # skip-don't-queue has never been measured: nothing records how old a
        # beat is by the time it is voiced, nor how many are dropped unspoken to
        # keep up. Both are invisible today because publishing text is instant,
        # and both start mattering the moment speech makes a beat occupy real
        # wall-clock time. Instrument first, so the speech work has a before.
        self._pace_stats: dict = {}
        self._reset_pace_stats()
        # monotonic deadline for when queued speech finishes. EVENT LOOP ONLY:
        # the 2026-07-25 freeze was a worker thread racing shared caster state,
        # so a threaded synth layer must marshal back (call_soon_threadsafe)
        # rather than touch this directly.
        self._speaking_until: float = 0.0
        self.transcript: deque = deque(maxlen=12)
        # FRACTURE's deep-think stall lines THIS match — she fixates on one
        # image ("threading a needle") and reuses it; the shared transcript
        # scrolls past between spaced-out stalls, so track them separately for
        # the WHOLE match (not just the last few) so a distant repeat 80 turns
        # later is still caught (see _stall_repeats). Reset at MATCH START.
        self._match_stalls: list = []
        self.clients: set = set()
        # skip-don't-queue: newest unspoken turn beat wins; framing beats
        # (MATCH START / RESULT) queue separately and always speak
        self._pending_turn: dict | None = None
        self._pending_framing: deque = deque()
        # Under PTS the policy INVERTS to queue-don't-skip. Skip-don't-queue
        # exists because a stale beat is worse than no beat — but once the
        # caster is deliberately holding lines for the viewer, "behind" is
        # correct, and the single pending slot then discards exactly the turns
        # the viewer is about to watch. Measured 2026-07-27: while speak() sat
        # in an 83s hold, every arriving beat replaced the last, so a 33-turn
        # game produced 5 spoken beats (T1, T6, T12, T23). The queue drains
        # faster than the viewer advances (generation ~8s vs ~13s per
        # presented turn), so it self-corrects rather than growing.
        # Beat texts seen this match, oldest first. The ungrounded-entity
        # guard grounds against these as well as the current beat: a line
        # recalling something from earlier ("that Spore is still on Gliscor")
        # is legitimate, and production really did show it. Deliberately the
        # BEATS and not self.transcript — grounding on our own past lines
        # would let one hallucination legitimise its repeats.
        self._beat_history: deque = deque(maxlen=40)
        self._pending_queue: deque = deque()
        self.PENDING_QUEUE_MAX = 120
        # species -> Tera type, for the type-claim guard. Tera REPLACES typing,
        # so this is not a decoration: checking a claim against the dex entry
        # of a Terastallized mon checks a typing that left the field.
        self._tera: dict = {}
        # consecutive speech drops per persona, so a starved voice can take
        # the lead back (see the rotation in speak())
        self._drops: dict = {}
        # Per-match spoken-line tallies and the lines themselves. _spoken
        # drives the deficit lead-swap: take 27 ran 15:8 with every PRISM
        # line in the first 13 turns, because the pre-flight budget cuts the
        # SECOND voice and the gremlin-first convention makes that PRISM
        # 5:1. The consecutive-drops rotation above can't see it — his
        # counter resets every time a solo beat lets him speak. _match_lines
        # backs the claimed-call guard (_stolen_call). Reset at MATCH START.
        self._spoken: dict = {}
        self._match_lines: dict = {}
        # spoken-line gap at which the trailing voice takes the lead on dual
        # beats (lead speaks unconditionally, so leading = speaking)
        self.DEFICIT_SWAP = 3
        # (us, them) actives from the last beat, for the switch consult —
        # a changed pair means a fresh matchup worth asking the expert about
        self._last_actives: tuple | None = None
        # Consecutive drops before a voice takes the lead back. MEASURED at
        # 2 over two full takes: 1.6:1 and 1.3:1 gremlin-to-analyst, against a
        # 2:1 ideal and 3:1 acceptable — so 2 already lands slightly MORE
        # balanced than wanted. Briefly set to 1 off a mid-take sample that
        # read 4:11; the completed games were 25:40 and 19:24, and alternating
        # would push further toward the analyst. RAISE this (3+) to give the
        # gremlin more of the contested beats, do not lower it.
        self.STARVED_AFTER = 2
        # how many grounded facts ride along on a beat, and how many of those
        # are held back for the two actives' abilities (see _gather_facts).
        # The cap is a latency budget: each fact is an expert round-trip.
        self.FACT_CAP = 3
        self.FACT_ABILITY_SLOTS = 1
        self._wake = asyncio.Event()

    # --- grounded facts (PRISM only) -----------------------------------
    def _speech_seconds(self, persona: str, line: str) -> float | None:
        """How long this line will occupy once voiced, or None when no speech
        layer is wired (which leaves pacing exactly as it is today). Deliberately
        NOT estimated from character count: PRISM runs a slowed paperwork cadence
        and FRACTURE is a motormouth, so the same text is a different duration in
        each voice and a length heuristic would be mistuned for one of them by
        construction. The synth knows the real number."""
        if self.duration_fn is None:
            return None
        try:
            secs = float(self.duration_fn(persona, line))
        except Exception as e:  # noqa: BLE001
            print(f"caster: duration_fn failed for {persona}: {e!r}", flush=True)
            return None
        return secs if secs > 0 else None

    def _reset_pace_stats(self):
        """Zero the per-match pacing counters (call at MATCH START)."""
        self._pace_stats = {"voiced": 0, "dropped": 0, "stale_ms": [],
                            "turnaround_ms": [], "speech_s": [],
                            "preflight_drops": 0, "overruns": 0,
                            "pts_held_s": []}

    def _speaking_backlog(self) -> float:
        """Seconds of queued speech still to play. 0.0 with no speech layer."""
        return max(0.0, self._speaking_until - time.monotonic())

    # How much queued speech is tolerable before a beat is dropped unspoken,
    # as a MULTIPLE of the per-beat budget. The gate used to compare a
    # multi-line backlog against the single-beat budget, which is a category
    # error: with an 8s budget and lines running 5-7s, two queued lines sit
    # at 8.1-8.6s — already over — so every following beat was dropped for
    # BOTH voices, and since a drop adds no speech the silence sustained
    # itself until the TTS drained. Measured on take 74 (a WINNING take):
    # beats T10-T15 went unspoken, taking a KO, a Tera + burn, and the snow
    # going up off the broadcast entirely, with the transcript jumping
    # T9 -> T17. Two lines of backlog is normal operation, not congestion.
    BACKLOG_LIMIT_FACTOR = 2.5
    # Interrupt-class beats (ko / tera / status / set_reveal / desk_swing)
    # buy extra room: when something actually happened, the silence should
    # land on housekeeping instead. Still bounded — nothing outruns this.
    BACKLOG_HARD_FACTOR = 4.0

    def _backlog_limit(self, item: dict) -> float | None:
        """Seconds of backlog this beat is allowed to speak over, or None
        when there is no budget to scale from. Priority bypass: a beat
        carrying an interrupt-class beat gets the higher ceiling."""
        if self.speech_budget is None:
            return None
        factor = self.BACKLOG_LIMIT_FACTOR
        for b in item.get("beats") or []:
            if (b or {}).get("priority") == "interrupt":
                factor = self.BACKLOG_HARD_FACTOR
                break
        return self.speech_budget * factor

    def _log_pace_summary(self):
        """One-line per-match pacing report, the counterpart to the RAG one.
        The number that matters is beat age at voicing: how stale the thing
        being said is by the time it is said. Dropped counts beats discarded
        unspoken by skip-don't-queue, which is the price paid for that
        freshness, and the two only trade against each other once speech
        occupies wall-clock time."""
        s = self._pace_stats
        if not s["voiced"]:
            return

        def pct(vals, p):
            if not vals:
                return 0.0
            ordered = sorted(vals)
            return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

        stale, turn = s["stale_ms"], s["turnaround_ms"]
        msg = (f"caster: pacing this match — voiced {s['voiced']}; dropped "
               f"{s['dropped']} unspoken; beat age at voicing median "
               f"{pct(stale, 0.5):.0f}ms / p90 {pct(stale, 0.9):.0f}ms / max "
               f"{max(stale) if stale else 0:.0f}ms; generation median "
               f"{pct(turn, 0.5):.0f}ms / p90 {pct(turn, 0.9):.0f}ms")
        held = s.get("pts_held_s") or []
        if held:
            msg += (f"; PTS held {len(held)} beat(s), median "
                    f"{pct(held, 0.5):.1f}s / max {max(held):.1f}s")
        if s["speech_s"]:
            msg += (f"; speech median {pct(s['speech_s'], 0.5):.2f}s / max "
                    f"{max(s['speech_s']):.2f}s; budget overruns "
                    f"{s['overruns']}; preflight drops {s['preflight_drops']}")
        print(msg, flush=True)

    def _reset_fact_stats(self):
        """Zero the per-match RAG counters (call at MATCH START)."""
        self._fact_stats = {"warmed": 0, "injected": 0, "beats_with_facts": 0,
                            "cache_hit": 0, "cold_fetch": 0, "miss": 0}

    def _ping_expert(self) -> bool | None:
        """Best-effort reachability probe for the grounded-rag expert. Any HTTP
        response (even a 404 on the base path) means the server answered ->
        reachable; connection-refused / timeout -> down. None if no expert
        configured. Purely diagnostic — never affects fact injection."""
        if not self.expert_url:
            return None
        try:
            urllib.request.urlopen(self.expert_url, timeout=3)
            return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            return False

    def _log_fact_summary(self):
        """One-line per-match RAG report so 'we think it helps' becomes a
        number: was the expert up, how many facts were warmed at preview, how
        many landed in beats, and — the key signal — how many in-game
        retrievals were cold round-trips vs warmed cache hits."""
        if not self.expert_url:
            return
        s = self._fact_stats
        up = ("reachable" if self._expert_up else
              "UNREACHABLE" if self._expert_up is False else "unknown")
        print(f"caster: RAG this match — expert {up}; warmed {s['warmed']}; "
              f"injected {s['injected']} fact(s) into {s['beats_with_facts']} "
              f"beat(s); in-game retrievals {s['cache_hit']} cache-hit / "
              f"{s['cold_fetch']} cold / {s['miss']} miss", flush=True)

    def _warm_cache(self, blob: str) -> int:
        """Pre-fetch every curated mechanic named in the team-preview blob (our
        paste + the predicted opponent paste) so the FIRST time PRISM narrates
        it mid-battle it's already a cache hit, not a cold round-trip on the
        critical path. Same substring match as _gather_facts, so a warmed name
        is exactly one a beat will hit. Worker-thread only; returns the count
        actually warmed. Bounded so a giant blob can't blast the expert."""
        if not self.expert_url or not blob:
            return 0
        low = blob.lower()
        hits = sorted({m for m in _MECHANICS if m in low},
                      key=len, reverse=True)
        warmed = sum(1 for name in hits[:24]
                     if self._retrieve_fact(name, warm=True))
        self._fact_stats["warmed"] = warmed
        return warmed

    async def _warm(self, blob: str):
        """Warm the fact cache off the event loop; log the count. A down or
        absent expert just warms nothing — never disturbs the match."""
        try:
            n = await asyncio.to_thread(self._warm_cache, blob)
            if n:
                print(f"caster: warmed {n} mechanic(s) from team preview",
                      flush=True)
        except Exception as e:
            print(f"caster: warm-cache failed: {e!r}", flush=True)

    def _retrieve_fact(self, name: str, warm: bool = False,
                       question: str | None = None):
        """Pull a mechanic's real effect from the expert (/retrieve) ->
        (fact_text, citation) or None. citation = {'label','corpus'} for the
        on-screen source chip. Cached (mechanics are static); a down/absent
        expert degrades to None so PRISM reasons as before — never raises.
        `warm=True` marks a team-preview pre-fetch: it fills the cache but is
        kept out of the in-game counters, so 'cold_fetch' measures only the
        round-trips warming failed to pre-empt.
        `question` overrides the definitional template — the strategy
        consults ask 'why', not 'what' (see _strategy_consults). The cache
        keys on the QUESTION, so a mon's tera consult and its switch consult
        cache separately; `name` stays the citation-match label either way."""
        q = question or f"what does {name} do in Pokemon"
        if q in self._fact_cache:
            if not warm:
                self._fact_stats["cache_hit"] += 1
            return self._fact_cache[q]
        result = None
        try:
            body = json.dumps({"question": q}).encode()
            req = urllib.request.Request(
                f"{self.expert_url}/retrieve", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.load(r)
            passages = data.get("passages") or data.get("results") or []
            texts = []
            for p in passages[:2]:
                t = " ".join((p.get("content") or p.get("text") or "").split())
                if t:
                    texts.append(t[:180])
            if texts:
                top = passages[0]
                title = (top.get("title") or name).split(" (")[0]
                title = title.split(" §")[0].strip()
                corpus = _CORPUS_NAME.get(top.get("corpus", ""),
                                          (top.get("corpus") or "").title())
                result = (" | ".join(texts),
                          {"label": title, "corpus": corpus})
                self._fact_cache[q] = result   # cache successes only
        except Exception:
            result = None
        # warm successes are tallied by _warm_cache; keep them out of the
        # in-game buckets so cold_fetch stays a clean 'warming missed this'
        if result and not warm:
            self._fact_stats["cold_fetch"] += 1
        elif not result and not warm:
            self._fact_stats["miss"] += 1
        return result

    def _gather_facts(self, beat_text: str, abilities=(),
                      consults=()) -> list:
        """(name, fact_text, citation) for each curated mechanic named in the
        beat PLUS the two active mons' abilities (already resolved upstream to
        single-known values), capped so prompt and latency stay bounded.
        Injecting the active abilities is the fix for PRISM reasoning from a
        NAME the beat didn't state — blaming Good as Gold for a Ghost's
        spinblock. `consults` are (label, question) strategy pulls from
        _strategy_consults; ONE leads the list per beat (it is the beat's
        point — a tera or a fresh matchup), the definitional facts fill what
        the cap leaves. Worker-thread only."""
        if not self.expert_url:
            return []
        facts = []
        for cname, cq in list(consults)[:1]:   # one strategy pull per beat
            got = self._retrieve_fact(cname, question=cq)
            if got:
                facts.append((cname, got[0], got[1]))
        low = beat_text.lower()
        # what's happening THIS turn leads; abilities ride along as context
        beat_hits = [m for m in _MECHANICS if m in low]
        ability_hits = []
        for ab in abilities:
            name = _canon_mechanic(ab)
            if name and name not in beat_hits and name not in ability_hits:
                ability_hits.append(name)
        # longest first within each group so 'quick feet' wins over a substring
        beat_hits.sort(key=len, reverse=True)
        ability_hits.sort(key=len, reverse=True)
        # Reserve a slot for the actives' abilities rather than taking a flat
        # (beat + ability)[:CAP]: beat mechanics sort first, so a turn naming
        # three of them dropped BOTH active abilities — exactly the busy turn
        # where knowing the mon has Unburden or Poison Heal explains the moment.
        # Abilities still fill spare room when the beat names little.
        cap = self.FACT_CAP - len(facts)
        n_ability = min(len(ability_hits), self.FACT_ABILITY_SLOTS, cap)
        picked = beat_hits[:cap - n_ability]
        picked += ability_hits[:cap - len(picked)]
        for name in picked:
            got = self._retrieve_fact(name)
            if got:
                facts.append((name, got[0], got[1]))
        if facts:
            self._fact_stats["injected"] += len(facts)
            self._fact_stats["beats_with_facts"] += 1
        return facts

    def _strategy_consults(self, item) -> list:
        """(label, question) strategy pulls for this beat — the first PULL
        instance of the expert integration (user-requested): the definitional
        template answers 'what does X do', these ask WHY a play makes sense,
        which is what the Smogon set-analysis corpus actually holds.

          tera    'why does {mon} run Tera {type}' — the 2-entity retrievable
                  form; the in-game 'against Z' application stays PRISM's own
                  reasoning, now anchored to the set's stated purpose.
          switch  a fresh matchup (one active changed, the other stayed) asks
                  why the incoming mon is a good switch-in against the one it
                  came in on — Checks-and-Counters territory.

        Board state stays the director's: these questions carry meta
        knowledge only, per the scope limit on the pull-based TODO item.
        Updates the last-actives tracker, so call it exactly once per beat.
        Latency: one consult per beat survives _gather_facts' cap, runs
        pre-generation off the event loop, and caches on the question — a
        repeat matchup is a cache hit."""
        out = []
        for b in item.get("beats") or []:
            if b.get("beat") == "tera":
                d = b.get("data") or {}
                mon, tt = d.get("mon"), d.get("tera_type")
                if mon and tt:
                    out.append((mon, f"why does {mon} run Tera {tt} in "
                                     f"competitive Pokemon"))
        hud = item.get("hud") or {}
        us, them = hud.get("us"), hud.get("them")
        if us and them:
            prev = self._last_actives
            self._last_actives = (us, them)
            if prev and prev != (us, them):
                if us != prev[0] and them == prev[1]:
                    out.append((us, f"why is {us} a good switch-in against "
                                    f"{them} in competitive Pokemon"))
                elif them != prev[1] and us == prev[0]:
                    out.append((them, f"why is {them} a good switch-in "
                                      f"against {us} in competitive Pokemon"))
                # both changed at once (double replacement): no single
                # incoming mon to ask about — skip rather than guess
        return out

    # --- intake (beat-protocol server) ---------------------------------
    async def handle(self, ws):
        self.clients.add(ws)
        try:
            async for raw in ws:
                try:
                    msg = _unwrap(raw)
                except Exception:
                    continue
                t = msg.get("type")
                if t == "module:authenticate":
                    # accept anyone local; the ack envelope shape is what
                    # CasterBridge's handshake succeeds unchanged
                    await ws.send(json.dumps({"json": {
                        "type": "module:authenticated",
                        "data": {"authenticated": True}}}))
                elif t == "input:text":
                    data = msg.get("data", {})
                    text = (data.get("text") or "").strip()
                    if not text:
                        continue
                    item = {"text": text,
                            "beats": data.get("beats") or [],
                            "hud": data.get("hud"),
                            "_queued": time.monotonic()}
                    if text.startswith("[MATCH START]"):
                        # fresh match: reset the RAG instrument and warm the
                        # fact cache from the preview blob (own + predicted
                        # opponent paste) off the event loop, so the first
                        # in-battle narration of each mechanic is a cache hit
                        self._reset_fact_stats()
                        self._reset_pace_stats()
                        self._spoken = {}
                        self._match_lines = {}
                        self._last_actives = None
                        blob = data.get("preview_text")
                        if blob and self.expert_url:
                            self._warm_task = asyncio.create_task(
                                self._warm(blob))
                    if (text.startswith("[MATCH START]")
                            or text.startswith("[RESULT]")):
                        self._pending_framing.append(item)
                    else:
                        if self.pts is not None:
                            # queue-don't-skip: every turn the viewer will
                            # watch gets its line
                            if len(self._pending_queue) >= self.PENDING_QUEUE_MAX:
                                self._pending_queue.popleft()
                                self._pace_stats["dropped"] += 1
                            self._pending_queue.append(item)
                        else:
                            if self._pending_turn is not None:
                                # the cost side of skip-don't-queue, counted
                                self._pace_stats["dropped"] += 1
                            self._pending_turn = item  # replace unspoken older
                    self._wake.set()
        finally:
            self.clients.discard(ws)

    # --- output (envelopes) ----------------------------------------------
    async def publish(self, beat_text: str, persona: str, line: str,
                      hud: dict | None, citations: list | None = None):
        envelope = json.dumps({"json": {
            "type": "output:gen-ai:chat:complete",
            "data": {"text": beat_text, "persona": persona, "hud": hud,
                     "citations": citations or [],
                     "message": {"content": line}}}})
        dead = []
        for c in list(self.clients):
            try:
                await c.send(envelope)
            except Exception:
                dead.append(c)
        for c in dead:
            self.clients.discard(c)

    # --- generation ------------------------------------------------------
    def _prompt(self, persona: str, item: dict,
                nudge: str | None = None) -> list[dict]:
        beats = item["beats"]
        # each voice anchors to ITS OWN beat: on a KO + desk-contradiction
        # turn, FRACTURE reacts to the KO while PRISM addresses the
        # contradiction — one shared anchor pulled both voices to the KO
        own_key = "analyst" if persona == "PRISM" else "gremlin"
        owned = [b for b in beats
                 if b.get("persona") in (own_key, "both", "either")]
        pool = owned or beats
        reg_beat = next((b for b in pool if b.get("register")), None)
        register = reg_beat.get("register") if reg_beat else None
        transcript = "\n".join(f"{p}: {ln}" for p, ln in self.transcript)
        direction = f"You are {persona}."
        if register:
            direction += f" Register: {register}."
        # anchor the line to its event or it floats free — measured: a
        # despair line about a burn that never said "burn", a Tera
        # analysis that never named the mon. Register-less beats need the
        # anchor just as much as registered ones.
        anchor = ((reg_beat or (pool[0] if pool else {})) or {}).get("prose")
        if anchor:
            direction += (f" You are reacting to THIS event: {anchor}. "
                          f"Name the Pokemon involved and the event itself "
                          f"(the move, the status, the crit) in your line.")
        elif persona == "PRISM" and not beats:
            # plain turn update: rotate the analytic lens so consecutive
            # tasks (and therefore sentence shapes) differ
            turn = (item.get("hud") or {}).get("turn") or 0
            direction += f" Angle: {_PRISM_ANGLES[turn % len(_PRISM_ANGLES)]}."
        # Tell her WHICH enemy this one belongs to. The contract states the
        # rule; this makes it mechanical, because deciding "was that RNG or a
        # read?" from prose is exactly the judgement she gets wrong — measured
        # live 2026-07-28, an opponent simply clicking Close Combat became
        # "the server literally decided Kyurem had to die for the plot".
        if persona == "FRACTURE" and (reg_beat or pool):
            kind = ((reg_beat or pool[0]) or {}).get("beat")
            beat_txt = item.get("text") or ""
            # Only a move that FAILED to do its job is the matchup story. A
            # super-effective hit is somebody's attack working, which is a
            # choice — routing that here told her "the enemy is our own
            # position" about THEIR Earthquake killing our Iron Crown.
            walled = bool(self._RESIST_RE.search(beat_txt)
                          or self._IMMUNE_RE.search(beat_txt))
            if kind == "crit_luck" or register in _DICE_REGISTERS:
                direction += (" This one genuinely WAS the dice: rage at the "
                              "server/RNG if you want.")
            elif walled:
                # whose move got walled decides the emotion, and getting it
                # backwards produced despair over THEIR Hurricane being
                # resisted by our own Iron Treads
                direction += (" A move got WALLED here — that is the type "
                              "chart, never the server, so do not blame it. "
                              "Check the beat for WHOSE move failed: if it "
                              "was OURS, the enemy is our own position and "
                              "having nothing better to throw, and that is "
                              "your bitterest register. If it was THEIRS, our "
                              "mon just ate it and you should be gloating.")
            else:
                direction += (" This was a CHOSEN play, not a dice roll. If "
                              "it was theirs, blame THEM by name — never the "
                              "server. If it was ours and it worked, take the "
                              "credit.")
            if "ability went off" in beat_txt:
                # the device wording alone did not hold her: takes 48 and 49
                # both aired "THEY CLICKED STATIC" over a beat that said the
                # ability went off. The routing has to say it in her terms.
                direction += (" An ABILITY went off this exchange — that is "
                              "a trap firing, not a play: NOBODY clicked it "
                              "and it cannot be clicked. Rage at the contact "
                              "or the luck, never say they used or clicked "
                              "the ability.")
        # FRACTURE fixates on one stall image; show her the ones already used
        # this match so she reaches for a new one (the reactive guard in
        # speak() is the backstop when this isn't enough)
        if (persona == "FRACTURE" and register == "deliberating"
                and self._match_stalls):
            used = "; ".join(f'"{s}"' for s in self._match_stalls[-8:])
            direction += (f" You have ALREADY used these stalls this match: "
                          f"{used}. Invent a totally different image — reuse "
                          "none of their metaphors or wording.")
        # Who is ACTUALLY on the field. Without this the model reaches into
        # the duo transcript for a name and reuses a mon that is long gone:
        # measured live 2026-07-27, both voices kept calling the opposing
        # active "Gholdengo" — OUR mon, fainted many turns earlier — while
        # their Great Tusk was in.
        hud = item.get("hud") or {}
        if hud.get("us") or hud.get("them"):
            ours = hud.get("us") or "unknown"
            theirs = hud.get("them") or "unknown"
            direction += (f" ON THE FIELD RIGHT NOW: ours is {ours}, theirs "
                          f"is {theirs}. Those are the only two Pokemon in "
                          f"play — never describe any other mon as active, "
                          f"and never attribute this turn's actions to one.")
        # A beat with no "Last exchange:" reports an INTENDED move whose
        # result is not known yet. Measured twice on 2026-07-27: on "We go
        # for Stone Edge" both voices invented a miss (the next beat said it
        # landed), and on "We go for Rapid Spin" PRISM said it "successfully
        # removed the entry hazards" before it had resolved. Naming the
        # uncertainty is cheaper than a guard per outcome type.
        if "Last exchange:" not in (item.get("text") or ""):
            direction += (" This beat states the move we are ABOUT to make; "
                          "its result is NOT known yet. Do not say whether it "
                          "hit, missed, knocked out, or worked — react to the "
                          "decision and the position instead.")
        elif ("We go for" in (item.get("text") or "")
              or "We switch to" in (item.get("text") or "")):
            # the exchange-less gate above missed beats that carry BOTH a
            # resolved exchange and a chosen move: take 28 T28 "The Sucker
            # Punch connects" and take 30 T8 "I absolutely crushed them with
            # it" were outcomes narrated for the 'We go for' move. 'We
            # switch to' is the same unresolved decision — take 48 T14 "The
            # Great Tusk lead has been neutralized by a switch" aired before
            # the switch had happened.
            direction += (" The 'We go for' / 'We switch to' line is our "
                          "NEXT play — it has not happened yet. Say nothing "
                          "about its outcome.")
        if " failed" in (item.get("text") or ""):
            # the move_failed narration states THAT it failed; the WHY is
            # mechanics, and inventing it produced "the Sucker Punch failed
            # because Kingambit couldn't bypass that Tera Ghost flip" (take
            # 30 T5 — wrong mon, wrong mechanic, and Dark hits Ghost anyway)
            direction += (" A move FAILED this exchange. Fails have "
                          "mechanical causes (a Sucker Punch fails when the "
                          "target is not attacking). Do not invent a reason "
                          "— state the fail plainly, or explain it only from "
                          "a listed grounded fact. If you name the reason "
                          "for a priority-move fail, it is that the TARGET "
                          "chose a NON-attacking move (a status move, a "
                          "switch) — NEVER that someone was 'pressing an "
                          "attack'; that is the exact opposite of how it "
                          "works.")
        direction += (" One or two short spoken sentences, react now. "
                      "Output only the line itself.")
        if nudge:
            direction += f" {nudge}"
        user = ""
        # PRISM's grounded facts: the real effect of any non-obvious
        # mechanic in the beat, pulled from the expert. Stops the
        # reason-from-the-name hallucinations (Good as Gold, Drain Punch).
        # PRISM only — FRACTURE is no-citations by contract.
        if persona == "PRISM" and item.get("_facts"):
            lines = "\n".join(f"- {n}: {f}" for n, f, _c in item["_facts"])
            user += ("GROUNDED FACTS — the real general behavior of the "
                     "mechanics in the beat AND the true abilities of the two "
                     "Pokemon active right now. Two hard rules: (1) NEVER name "
                     "an ability or mechanic that isn't listed here. (2) When "
                     "the beat says a move had 'no effect' / was immune, that "
                     "is a TYPE matchup (a Ghost ignores Rapid Spin; the "
                     "hazards stay up) UNLESS the beat ITSELF names the ability "
                     "that blocked it (Levitate, Volt Absorb, Flash Fire) — "
                     "credit an ability for an immunity ONLY when the beat "
                     "names it, never otherwise, even for an ability listed "
                     "below. (3) A fact's example Pokemon and moves come from "
                     "the corpus, NOT this game — apply the fact's REASONING "
                     "to what is actually on the field, and never name a "
                     "corpus example as if it were in play (a fact citing "
                     "'Body Press Corviknight' grounds a Tera read, but the "
                     "Fighting move to name is the one THIS opponent has "
                     "shown). Use these facts to add meaning to what DID "
                     "happen — why a mon survived, why a status is actually "
                     "a boon — never to invent a reason a move failed. "
                     "Context, not a mandate: reach for them only when they "
                     f"explain the moment, not every line:\n{lines}\n\n")
        # FRACTURE's Book of Grudges: inject the real vendetta for the mon
        # on the field so she can cite it. Only a recorded grudge appears
        # here, which is the whole point — her paranoia has to be earned,
        # never invented. Injected as available context, not a command:
        # she references it when it fits the moment, not every line.
        if persona == "FRACTURE":
            them = (item.get("hud") or {}).get("them")
            grudge = self.grudges.grudge_for(them)
            if grudge:
                user += (f"{grudge} Reference it only if it fits this "
                         f"moment; never invent a grudge not stated here.\n\n")
        if transcript:
            user += f"Broadcast so far:\n{transcript}\n\n"
        user += f"New beat from the director:\n{item['text']}\n({direction})"
        return [{"role": "system", "content": self.prompts[persona]},
                {"role": "user", "content": user}]

    def _generate_sync(self, persona: str, item: dict,
                       nudge: str | None = None,
                       temp_boost: float = 0.0) -> str:
        knobs = dict(_GEN[persona])
        knobs["temperature"] = knobs["temperature"] + temp_boost
        body = json.dumps({
            "model": self.model,
            "messages": self._prompt(persona, item, nudge=nudge),
            "stream": False,
            # thinking OFF. gemma4 is thinking-capable and Ollama defaults it
            # ON, which leaked reasoning into the spoken line and truncated
            # replies by eating the token budget. Only the NATIVE /api/chat
            # honours `think:false`; on /v1 the lever is reasoning_effort,
            # and only "none" (Ollama >= 0.32) turns it off.
            "reasoning_effort": "none",
            **knobs,
        }).encode()
        req = urllib.request.Request(
            f"{self.upstream}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.load(resp)
        return out["choices"][0]["message"]["content"]

    @staticmethod
    def _fabricated_crit(line: str, item: dict) -> bool:
        """True when the spoken line claims a crit the beat never reported —
        a facts-of-record violation (a super-effective/heavy hit narrated as
        a 'crit'). The beat text carries 'critical hit' only when one really
        landed, so a crit word without it in the beat is invented."""
        if not re.search(r"\bcrit(?:ical|s)?\b", line, re.I):
            return False
        return "critical" not in (item.get("text") or "").lower()

    _OVERLORD_POWER = re.compile(
        r"\bsupreme\s+overlord\b", re.I)
    _POWER_VOCAB = re.compile(
        r"\b(?:stack\w*|boost\w*|power\w*|advantage|online|matter\w*|"
        r"stronger|damage)\b", re.I)
    _BODIES = re.compile(r"Bodies: us (\d+) standing, them (\d+)")

    def _overlord_state_claim(self, line: str, item: dict) -> bool:
        """The first per-claim ABILITY-STATE evaluator. Take 71 T2: 'the
        advantage remains with us because of the Supreme Overlord stacks' —
        at 6-6 bodies. Supreme Overlord scales with FAINTED ALLIES; with
        nobody fainted on either side, zero stacks exist for either
        Kingambit, so any power-claim about it is false by arithmetic the
        beat itself carries. Precision-first: fires only when BOTH sides
        are untouched (no binding needed to know whose stacks are zero);
        a bare ability mention without power vocabulary passes."""
        if not (self._OVERLORD_POWER.search(line)
                and self._POWER_VOCAB.search(line)):
            return False
        m = self._BODIES.search(item.get("text") or "")
        return bool(m and m.group(1) == "6" and m.group(2) == "6")

    # --- field-state guards: weather/screens/boost claims checked against
    # the state footer the director stamps into every beat ("Weather: rain.",
    # "Screens: our Reflect.", "Boosts: their Volcarona +1 Special Attack.")
    # plus the exchange prose around it. All three share the overlord guard's
    # contract: default-pass, fire only when a present-tense claim has ZERO
    # support anywhere in the beat text — so transitions ("Drought set harsh
    # sun up"), retrospectives ("the rain is gone") and hypotheticals ("if
    # they get screens up") never trip them.

    @staticmethod
    def _claim_sentence(line: str, pos: int) -> str:
        """The sentence of `line` containing offset `pos` — the window the
        not-now/speculation scans run over. Abbreviation periods (K.O) only
        shrink the window, which errs toward firing; the speculation tokens
        are broad enough to absorb that."""
        start = max(line.rfind(".", 0, pos), line.rfind("!", 0, pos),
                    line.rfind("?", 0, pos)) + 1
        ends = [i for i in (line.find(".", pos), line.find("!", pos),
                            line.find("?", pos)) if i != -1]
        return line[start:min(ends) if ends else len(line)]

    _WX_CLAIM = re.compile(
        r"\b(?:in|under|through)\s+(?:th(?:is|e|at)\s+)?"
        r"(?:rain|sun(?:light|shine)?|sandstorm|snow|hail)\b"
        r"|\bthe\s+(?:rain|sun(?:light|shine)?|sand(?:storm)?|snow|hail)\s+"
        r"(?:is|stays|keeps|remains|continues|pelts|batters|chips|falls|"
        r"beats|pounds|rages)\b"
        r"|\b(?:rain|sun|sand|snow|hail)-boosted\b", re.I)
    _WX_NOT_NOW = re.compile(
        r"\b(?:gone|cleared|clears|fade[sd]?|over|end(?:s|ed)?|expir\w+|"
        r"ran out|runs out|dried|down|without|no longer|lost|"
        r"if|could|would|might|may|once|when|unless|before|soon|"
        r"coming|incoming|hop\w+|want\w*|need\w*|threat\w*|"
        r"set(?:s|ting)?\s+up|summon\w*|bring\w*|"
        r"every\w*\s+\w+\s+under\s+the\s+sun)\b", re.I)

    def _weather_state_claim(self, line: str, item: dict) -> bool:
        """True when the line treats a weather as ACTIVE and no form of that
        weather appears anywhere in the beat — footer or exchange. 'in this
        sun' with no sun up is the same invention as the phantom crit, just
        about the field instead of the dice."""
        text = (item.get("text") or "").lower()
        for m in self._WX_CLAIM.finditer(line):
            if self._WX_NOT_NOW.search(self._claim_sentence(line, m.start())):
                continue
            tok = m.group(0).lower()
            fam = ("rain" if "rain" in tok else
                   "sand" if "sand" in tok else
                   "sun" if "sun" in tok else "snow")
            roots = ("snow", "hail") if fam == "snow" else (fam,)
            if not any(r in text for r in roots):
                return True
        return False

    _SCR_CLAIM = re.compile(
        r"\b(?:behind|under)\s+(?:the\s+|a\s+|our\s+|their\s+|dual\s+|"
        r"both\s+)?(?:screens?|reflect|light\s+screen|aurora\s+veil|veil)\b"
        r"|\b(?:screens?|reflect|light\s+screen|aurora\s+veil|veil)\s+"
        r"(?:is|are)\s+(?:up|active|still|online|holding)\b"
        r"|\b(?:the|our|their|dual|both)\s+screens\b", re.I)
    _SCR_NOT_NOW = re.compile(
        r"\b(?:broke\w*|break\w*|shatter\w*|gone|wore|worn|fade[sd]?|"
        r"expir\w+|down|end(?:s|ed)?|cleared|removed|blown|without|no|"
        r"none|lost|if|could|would|might|may|once|when|unless|want\w*|"
        r"need\w*|hop\w+|before|threat\w*|set(?:s|ting)?\s+up|"
        r"go(?:es)?\s+up|put(?:s|ting)?\s+up|coming|veil\s+of)\b", re.I)

    def _screens_state_claim(self, line: str, item: dict) -> bool:
        """True when the line treats screens as ACTIVE and no screen is in
        evidence anywhere in the beat. Fires only in the nothing-up case:
        when a Screens: footer exists at all, side-binding a bare 'behind
        screens' is guesswork and the guard stays out of it."""
        m = self._SCR_CLAIM.search(line)
        if not m:
            return False
        if self._SCR_NOT_NOW.search(self._claim_sentence(line, m.start())):
            return False
        text = (item.get("text") or "").lower()
        return not any(w in text for w in ("screen", "reflect", "veil"))

    _STAGE_CLAIM = re.compile(
        r"(?:^|[\s(])\+\d\b"
        r"|\bplus[- ](?:one|two|three|four|five|six|\d)\b"
        r"|\bminus[- ](?:one|two|three|\d)\b", re.I)
    _STAGE_NOT_NOW = re.compile(
        r"\b(?:if|could|would|might|may|once|when|unless|after|gets?|"
        r"reach\w*|want\w*|threat\w*|imagine|before|risk\w*|fish\w*|"
        r"looking|priorit\w+)\b", re.I)
    _STAGE_SUPPORT = ("boosts:", "raise", "rose", "boost", "maxed",
                      "dropped", "cut", "fell", "lowered", "stat",
                      "baton pass", "psych up")

    def _boost_state_claim(self, line: str, item: dict) -> bool:
        """True when the line states a numeric stat stage ('at +2') and the
        beat carries neither a Boosts: footer nor any stat-change language —
        a power state invented from nothing, the Supreme Overlord shape with
        a number attached. Baton Pass / Psych Up hand-offs the footer can't
        see are escaped by name."""
        m = self._STAGE_CLAIM.search(line)
        if not m:
            return False
        if self._STAGE_NOT_NOW.search(self._claim_sentence(line, m.start())):
            return False
        text = (item.get("text") or "").lower()
        return not any(w in text for w in self._STAGE_SUPPORT)

    # The director deliberately words a consumed item by what the
    # consumption DID ("Booster Energy kicked in"), because "used up its X"
    # is loss-coded before the caster ever reads it — measured live
    # 2026-07-28 as "The Booster Energy is gone, so we lost our speed
    # advantage", 4 occurrences in 147 beats, every one framed as a loss.
    # Spending it is what switches Quark Drive ON. The prose fix removed
    # the invitation; this guard rules on the claim itself.
    _ITEM_ACTIVATED = re.compile(
        r"([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*)\s+"
        r"(?:kicked in|activated|fired|undid the stat drops|"
        r"skipped the charge turn|shook off the restriction)")
    # Real item denials keep their own prose and ARE losses — never rule
    # against a line grieving one of these.
    _ITEM_DENIED = re.compile(
        r"\b(?:knocked off|swiped|popped|stole|took)\b", re.I)
    _ITEM_LOSS_FRAME = re.compile(
        r"\b(?:gone|lost|losing|lose|wasted|burnt|burned through|"
        r"used up|spent|no longer|without|stripped|down an item|"
        r"deprived|robbed of|missing)\b", re.I)

    def _item_polarity_claim(self, line: str, item: dict) -> bool:
        """True when an item that ACTIVATED is narrated as a loss. The beat
        says the Booster Energy kicked in — that is Quark Drive coming
        online — and a line reading it as 'the Booster Energy is gone, so
        we lost our speed advantage' inverts the mechanic.

        Precision-first: the item must be named in the line, the loss frame
        must sit in the same sentence, and any real denial in the beat
        (Knock Off, theft, a popped Air Balloon) makes the guard abstain —
        those genuinely are losses and the desk should say so."""
        text = item.get("text") or ""
        if self._ITEM_DENIED.search(text):
            return False
        for m in self._ITEM_ACTIVATED.finditer(text):
            name = m.group(1)
            # strip a leading possessive holder ("our Iron Valiant's
            # Booster Energy" -> the capture starts at the holder)
            item_name = name.split("'s ")[-1].strip()
            if len(item_name) < 4 or item_name.lower() not in line.lower():
                continue
            idx = line.lower().index(item_name.lower())
            if self._ITEM_LOSS_FRAME.search(
                    self._claim_sentence(line, idx)):
                return True
        return False

    _KO_IN_BEAT = re.compile(
        r"(\w[\w'-]*(?:\s+[A-Z][\w'-]*)?)'s\s+([A-Z][\w' -]*?)\s+"
        r"knocked out\s+(?:our|their)?\s*([A-Z][\w-]*)")
    # The move did nothing. These assert ineffectiveness directly and are
    # not normally negated, so they convict on sight.
    _DISMISS_FRAME = re.compile(
        r"\b(?:did|does|do)\s+(?:absolutely\s+|literally\s+)?nothing\b"
        r"|\bno\s+damage\b|\bbarely\s+scratch\w*\b", re.I)
    # The victim lived. These are routinely NEGATED to say the opposite —
    # "that Tera flip didn't change the matchups enough to survive" is a
    # correct reading of a KO, and the first version of this guard read the
    # word "survive" and dropped the line twice (take 82 T19, a good PRISM
    # line lost). Negation flips their meaning, so check for it.
    _SURVIVE_FRAME = re.compile(
        r"\btook\s+(?:that|it|the)\b"
        r"|\btank(?:ed|s)\b|\blike\s+a\s+champ\b|\bsurviv\w+\b"
        r"|\bwalk(?:ed)?\s+it\s+off\b|\bshrug\w*\s+(?:it\s+)?off\b"
        r"|\bheld\s+on\b|\bstill\s+standing\b|\bunfazed\b"
        r"|\bbrush\w*\s+(?:it\s+)?off\b", re.I)
    _SURVIVE_NEGATED = re.compile(
        r"\b(?:didn't|did\s+not|doesn't|does\s+not|couldn't|could\s+not|"
        r"wasn't|was\s+not|weren't|were\s+not|isn't|is\s+not|won't|"
        r"will\s+not|not|never|no|failed\s+to|unable\s+to|"
        r"nowhere\s+near)\b|n't\b", re.I)

    def _ko_dismissal_claim(self, line: str, item: dict) -> bool:
        """True when a KO is narrated as the victim SURVIVING, or as the
        killing move doing nothing.

        Take 76 T8 (user-flagged): the beat reads 'our Kyurem's Icicle
        Spear knocked out their Cinderace with not very effective' and the
        body count drops to 4 — and FRACTURE aired 'Cinderace just TOOK
        THAT HIT like a champ! That Icicle Spear did NOTHING'. She bound to
        the effectiveness tag and dropped the knockout, so the record says
        dead and the broadcast says fine. Take 73 T20 was the same shape
        ('A Hurricane that does NOTHING?' over a Hurricane that KO'd Iron
        Treads) and was wrongly written off as rhetoric.

        A not-very-effective KO is exactly the trap: the dismissive read is
        RIGHT about the multiplier and catastrophically wrong about the
        outcome. Bound tight — the dismissal must name the mon that died or
        the move that killed it, so a 'did nothing' about some other move
        in a crowded beat passes untouched."""
        for m in self._KO_IN_BEAT.finditer(item.get("text") or ""):
            move, victim = m.group(2).strip(), m.group(3).strip()
            for frame, negatable in ((self._DISMISS_FRAME, False),
                                     (self._SURVIVE_FRAME, True)):
                for fm in frame.finditer(line):
                    # The dismissal must be IN THE SAME SENTENCE as the mon
                    # that died or the move that killed it. Scoping this to
                    # the whole line convicted a true statement (take 85
                    # T19): the beat had "their Zapdos's Thunder Wave FAILED"
                    # and, separately, "Tachyon Cutter knocked out their
                    # Zapdos", and "THEY CLICKED THUNDER WAVE AND IT DID
                    # NOTHING! I had their Zapdos exactly where I wanted it"
                    # bound the victim's name in one sentence to a dismissal
                    # of an entirely different move in another.
                    sent = self._claim_sentence(line, fm.start())
                    low = sent.lower()
                    if not ((len(victim) >= 4 and victim.lower() in low)
                            or (len(move) >= 4 and move.lower() in low)):
                        continue
                    if negatable and self._SURVIVE_NEGATED.search(sent):
                        continue
                    return True
        return False

    _TYPE_MECHANISM = re.compile(
        r"\b(?:immun\w+|no\s+effect|does(?:n't|\s+not)\s+affect|"
        r"resist\w*|super[- ]effective|not\s+very\s+effective|"
        r"bypass\w*|get\s+(?:past|through)|go\s+through|"
        r"typ(?:e|ing)\s+(?:advantage|matchup|change|flip)|"
        r"miss(?:ed|es)?|dodg\w+|evad\w+|evasion|whiff\w*)\b", re.I)

    def _fail_mechanism_claim(self, line: str, item: dict) -> bool:
        """True when a move FAILURE is explained by a type interaction, an
        immunity or a miss — three distinct mechanics the desk conflates.

        The founding case of this whole guard family, take 30 T5: 'the
        Sucker Punch failed because Kingambit couldn't bypass that Tera
        Ghost flip'. Sucker Punch fails when the TARGET did not attack;
        typing has nothing to do with it, and Dark into Ghost is super
        effective anyway. The director states the failure as bare fact and
        deliberately offers no reason — so a mechanism supplied by the line
        is the caster's invention, not the record's.

        Same narrow shape as _miss_for_immunity: fires only when the beat
        has a failure and NO real immunity or miss anywhere in it to be
        talking about. Legitimate fail reasoning ('the target chose a
        non-attacking move', 'Protect twice in a row') carries none of this
        vocabulary and passes untouched."""
        text = (item.get("text") or "").lower()
        if "failed" not in text:
            return False
        if "had no effect" in text or "missed" in text:
            return False
        if not re.search(r"\bfail(?:ed|s|ing)?\b", line, re.I):
            return False
        return bool(self._TYPE_MECHANISM.search(
            self._claim_sentence(line, re.search(
                r"\bfail(?:ed|s|ing)?\b", line, re.I).start())))

    # "Boosts: our Kyurem -1 Defense, +1 Speed; their Zapdos -1 Special
    # Attack." — the footer names the side, the mon and the sign, which is
    # everything needed to tell a debuff we INFLICTED from one we took.
    _BOOST_FOOTER = re.compile(r"Boosts: ([^.]*)\.")
    _DROP_WORD = re.compile(
        r"\b(?:drop|dropped|cut|gutted|lowered|slashed|sapped|"
        r"weakened|debuff\w*|reduc\w+|tank\w*)\b", re.I)
    _HARM_TO_US = re.compile(
        r"\b(?:has|have|got|leaves?|left)\s+(?:us|me)\b"
        r"|\b(?:us|me|our|my)\s+(?:reeling|crippled|gutted|ruined)\b"
        r"|\bcrippl\w+\s+(?:us|me|our|my)\b"
        r"|\bagainst\s+(?:us|me)\b", re.I)
    _HARM_TO_THEM = re.compile(
        r"\b(?:has|have|got|leaves?|left)\s+them\b"
        r"|\bthem\s+(?:reeling|crippled|gutted|ruined)\b"
        r"|\bcrippl\w+\s+(?:them|their)\b", re.I)

    def _boost_polarity_claim(self, line: str, item: dict) -> bool:
        """True when a stat-DROP is credited to the wrong side. Take 74 T3:
        the footer read 'Boosts: their Zapdos -1 Special Attack' — OUR
        Moonblast cut THEIR Zapdos, a debuff in our favour — and FRACTURE
        aired 'Zapdos has us REELING with that Special Attack drop!',
        turning our own successful debuff into an injury. Same family as
        the luck-polarity inversion: the record names the side and the
        sign, so the inversion is false by arithmetic.

        Precision-first, like every guard here: fires only when negative
        stages sit on EXACTLY ONE side of the footer, so there is no
        binding to guess at."""
        m = self._BOOST_FOOTER.search(item.get("text") or "")
        if not m or not self._DROP_WORD.search(line):
            return False
        ours = theirs = False
        for frag in m.group(1).split(";"):
            # a NEGATIVE STAGE, not any hyphen — species names carry them
            # ("their Slowking-Galar +1 Attack" is not a drop)
            if not re.search(r"-\d", frag):
                continue
            if frag.strip().startswith("our "):
                ours = True
            elif frag.strip().startswith("their "):
                theirs = True
        if ours == theirs:              # both sides or neither: abstain
            return False
        if theirs:
            return bool(self._HARM_TO_US.search(line))
        return bool(self._HARM_TO_THEM.search(line))

    _HAZ_CLAIM = re.compile(
        r"\b(?:the|those|these|our|their)\s+(?:rocks|spikes|webs?|hazards)\b"
        r"|\bstealth\s+rock\b|\btoxic\s+spikes\b|\bsticky\s+web\b"
        r"|\bsteelsurge\b"
        r"|\bhazards?\s+(?:are|is)\s+(?:up|set|there|active)\b"
        r"|\b(?:chip|damage)\s+from\s+the\s+(?:rocks|spikes|web|hazards)\b",
        re.I)
    _HAZ_NOT_NOW = re.compile(
        r"\b(?:gone|cleared|removed|spun|spin\w*|defog\w*|blown|break\w*|"
        r"broke\w*|no|none|without|lost|if|could|would|might|may|once|"
        r"when|unless|want\w*|need\w*|hop\w+|before|threat\w*|"
        r"set(?:s|ting)?\s+up|go(?:es)?\s+up|put(?:s|ting)?\s+up|coming|"
        r"on\s+the\s+rocks)\b", re.I)

    def _hazard_state_claim(self, line: str, item: dict) -> bool:
        """True when the line treats entry hazards as ON THE FIELD and no
        hazard is in evidence anywhere in the beat — footer or exchange.
        The take-26 class ('the hazards are gone' from a boost Rapid Spin)
        is guarded on the CLEAR side by _fabricated_hazard_clear; this is
        the presence side ('the rocks are chipping them' over an empty
        field). When a Hazards: footer exists at all the support check
        passes everything — side-binding 'the rocks' is guesswork."""
        m = self._HAZ_CLAIM.search(line)
        if not m:
            return False
        if self._HAZ_NOT_NOW.search(self._claim_sentence(line, m.start())):
            return False
        text = (item.get("text") or "").lower()
        return not any(w in text for w in
                       ("hazard", "stealth rock", "spikes", "sticky web",
                        "steelsurge"))

    # Miss prose always leads with the MOVER's possessive ("their Zapdos's
    # Hurricane missed our Iron Crown"), and clauses are ";"-joined, so the
    # first side-word inside the clause is the side the dice went against.
    _MISS_IN_BEAT = re.compile(r"\b(our|their)\s+[^;.!:]*?\bmissed\b")
    _DICE_WORD = re.compile(r"\b(?:dice|server|rng|luck|hax)\b", re.I)
    _LUCK_AGAINST_US = re.compile(
        r"\b(?:against\s+(?:us|me)|stop(?:ping)?\s+(?:my|our|me)\b|"
        r"rob(?:bed|bing)?\s+(?:us|me)|screw(?:ed|ing)?\s+(?:us|me)|"
        r"punish(?:ing|ed)?\s+(?:us|me)|out\s+to\s+get\s+(?:us|me)|"
        r"hates?\s+(?:us|me)|cheat(?:ed|ing)?\s+(?:us|me))", re.I)
    _LUCK_FAVOR_US = re.compile(
        r"\b(?:paying\s+(?:us|me)\s+back|against\s+them|"
        r"on\s+(?:our|my)\s+side|going\s+(?:our|my)\s+way|"
        r"love[sd]?\s+(?:us|me)|owed|repaid)", re.I)

    def _beat_miss_directions(self, item: dict) -> set:
        """Which side(s) the beat's misses went against — the MOVER's side,
        'our' or 'their'."""
        return set(self._MISS_IN_BEAT.findall(item.get("text") or ""))

    def _luck_polarity_claim(self, line: str, item: dict) -> bool:
        """True when a dice-grievance inverts the polarity of the beat's
        miss. Take 72 T14: 'their Zapdos's Hurricane missed our Iron Crown'
        — luck against THEM, and the ledger counted it that way — but
        FRACTURE aired 'the server DECIDED that Hurricane should MISS! THE
        DICE are TRYING to stop my Iron Crown!' — their miss rendered as
        our persecution. Whose move missed is stated in the beat, so the
        inversion is false by the record; the mirror case (celebrating a
        payback over OUR OWN miss) is the same arithmetic. Abstains when
        the beat's misses point both ways, or when its ledger suffix names
        a same-beat luck event on the other side."""
        dirs = self._beat_miss_directions(item)
        if len(dirs) != 1:
            return False
        if not (self._DICE_WORD.search(line)
                and re.search(r"\bmiss\w*\b", line, re.I)):
            return False
        text = (item.get("text") or "").lower()
        if dirs == {"their"}:
            return bool(self._LUCK_AGAINST_US.search(line)
                        and "dice have gone against us" not in text)
        return bool(self._LUCK_FAVOR_US.search(line)
                    and "dice have gone against them" not in text)

    @staticmethod
    def _miss_for_immunity(line: str, item: dict) -> bool:
        """True when the line narrates a no-effect as a miss/dodge — three
        sightings in one hunt: 'the Thunder Wave missed because of Iron
        Treads' immunity' (take 52 T24), 'that evasion' for a Ground
        immunity (take 48 T24). An immunity is the type chart working;
        a miss is the dice — conflating them corrupts both the luck ledger
        and the mechanics. Narrow on purpose: fires only when the beat has
        a no-effect and NO real miss to be talking about."""
        text = (item.get("text") or "").lower()
        if "had no effect" not in text or "missed" in text:
            return False
        return bool(re.search(r"\b(?:miss(?:ed|es)?|dodg\w+|evad\w+|"
                              r"evasion|whiff\w*)\b", line, re.I))

    @staticmethod
    def _fabricated_recoil(line: str, item: dict) -> bool:
        """True when the line blames 'recoil' the beat never reported — take
        49 T16/T19: Headlong Rush's self-stat-drops narrated as recoil,
        twice, with the real effect stated in the beat both times. Same
        shape as _fabricated_crit: an event word with no beat support."""
        if not re.search(r"\brecoil\b", line, re.I):
            return False
        return "recoil" not in (item.get("text") or "").lower()

    def _ungrounded_entity(self, line: str, item: dict) -> str | None:
        """The species/move a line names must be in evidence. Returns the
        offending name, or None.

        Ported from the gold set's contract check, which the LIVE caster never
        had — it only carried three narrow guards (crit / synergy / immunity),
        so a plausible-but-unevidenced mechanic sailed through. Measured live
        2026-07-27: PRISM said "The halved damage from Multiscale was likely
        intended to keep Roost viable" on a beat that never mentions
        Multiscale. Dragonite really does have it, which is exactly what makes
        it dangerous: true-sounding, unsupported, invisible to every other
        guard.

        Grounded against the beats seen THIS MATCH, the actives and their
        known abilities, and any injected expert facts. Case-sensitivity is
        the trick carried over from the eval: only a properly capitalised
        occurrence counts, so prose uses of common-word moves ("rest",
        "protect", "will you") never false-flag. No-ops if poke_env is absent.
        """
        try:
            from crystal_broadcast.game_data import DATA
            names = DATA.entity_names()
        except Exception:
            return None
        hud = item.get("hud") or {}
        allowed = " ".join(self._beat_history)
        allowed += " " + " ".join(str(hud.get(k) or "") for k in
                                  ("us", "them", "us_ability", "them_ability"))
        if item.get("_facts"):
            allowed += " " + " ".join(n for n, _f, _c in item["_facts"])
        # a status on the board grounds the whole family of moves that inflict
        # it: naming the move behind a status we are reacting to is not a
        # hallucination
        codes = {b.get("data", {}).get("status")
                 for b in (item.get("beats") or [])}
        codes = {c for c in codes if c}
        if codes & {"tox", "psn"}:
            codes |= {"tox", "psn"}
        if codes:
            smoves = DATA.status_moves()
            for c in codes:
                allowed += " " + " ".join(smoves.get(c, ()))
        low, allowed_low = line.lower(), allowed.lower()
        for name in names:
            if len(name) < 4:
                continue
            nl = name.lower()
            if nl in low and nl not in allowed_low and name in line:
                return name
        return None

    # Up to two words may sit between "I" and the claim verb: FRACTURE
    # speaks in intensifiers by contract, and "I absolutely called that
    # Icicle Spear would clean them up" sailed past the adjacent-only form
    # live on take 29 — her first line of the match. Negated sentences
    # ("I never said...") are excluded in _stolen_call, not here.
    _CLAIM_RE = re.compile(
        r"\b(?:i\s+(?:\w+(?:'\w+)?\s+){0,2}?(?:told\s+you|said|"
        r"called\s+(?:it|that|this)|promised(?:\s+you)?)|"
        r"like\s+i\s+(?:said|told\s+you)|"
        r"i(?:'ve|\s+have)\s+been\s+saying)\b", re.I)
    # third-person self-citation: "was predicted by the desk" (take 49 T5,
    # about a Tera flip nobody had predicted) evades the first-person forms.
    # Desk claims verify against BOTH voices' prior lines — the desk is
    # either of them.
    _DESK_CLAIM_RE = re.compile(
        r"\b(?:(?:predicted|called|foreseen|expected)\s+by\s+the\s+desk|"
        r"the\s+desk\s+(?:predicted|called|saw|expected)(?:\s+(?:it|this|"
        r"that))?|as\s+the\s+desk\s+said)\b", re.I)

    def _stolen_call(self, line: str, persona: str) -> str | None:
        """The entity behind an 'I told you / I called it' the speaker never
        previously mentioned — or None.

        Take 27 T14: PRISM named the Icicle Spear plan on T13; one beat
        later FRACTURE opened 'I TOLD YOU THAT ICICLE SPEAR WAS THE FINAL
        NAIL IN THE COFFIN' — her first mention of the move, the call her
        desk mate's. A fabricated past is checkable against the on-screen
        transcript, which is what makes it worse than ordinary bravado.

        Binding is entity-based like _ungrounded_entity and scoped to the
        claim-bearing SENTENCE, so a subject-free 'I called it!' (the
        set-reveal bit, part of her contract) never fires, and an innocent
        first mention elsewhere in the line doesn't either. FRACTURE shouts
        in caps, which destroys the capitalisation signal the entity guard
        leans on — so all-caps lines match case-insensitively instead.
        Verified against the speaker's OWN lines this match (_match_lines):
        claiming a call she really made is the bit working as intended."""
        if not (self._CLAIM_RE.search(line)
                or self._DESK_CLAIM_RE.search(line)):
            return None
        try:
            from crystal_broadcast.game_data import DATA
            names = DATA.entity_names()
        except Exception:
            return None
        own_prior = " ".join(self._match_lines.get(persona, ())).lower()
        all_prior = " ".join(l for ls in self._match_lines.values()
                             for l in ls).lower()
        checks = []          # (claim sentence, the prior pool it must clear)
        for s in re.split(r"(?<=[.!?])\s+", line):
            # a denial is not a claim: the loosened regex would otherwise
            # read "I never said X" as claiming X
            if re.search(r"\b(?:never|not)\b|n't\b", s, re.I):
                continue
            if self._CLAIM_RE.search(s):
                checks.append((s, own_prior))
            elif self._DESK_CLAIM_RE.search(s):
                checks.append((s, all_prior))
        for sent, prior in checks:
            caps_blind = sent.isupper()
            low = sent.lower()
            for name in names:
                if len(name) < 4:
                    continue
                nl = name.lower()
                if (nl in low and (caps_blind or name in sent)
                        and nl not in prior):
                    return name
            # Tera tokens bind too: "the Tera Ghost flip was predicted by
            # the desk" names no dex entity, only the tera — which is
            # exactly as checkable against the transcript
            for tok in re.findall(r"\btera[- ][a-z]+\b", low):
                if tok not in prior:
                    return tok
        return None

    # effectiveness vocabulary -> the multiplier the claim implies
    _RESIST_RE = re.compile(
        r"\b(resists?|resisted|resisting|not very effective|walls?|walled|"
        r"shrugs? off|tanks?)\b", re.I)
    _IMMUNE_RE = re.compile(
        r"\b(immune|immunity|no effect|does nothing|doesn't do anything)\b",
        re.I)
    _SUPER_RE = re.compile(r"\b(super[- ]effective|weak to|weakness to)\b",
                           re.I)
    _TERA_RE = re.compile(
        r"([A-Z][\w'.-]*(?:-[A-Z][\w'.-]*)?) Terastallized into an? (\w+) type")

    def _note_tera(self, beat_text: str) -> None:
        """Remember what a mon Terastallized into, for the type-claim guard.
        Tera REPLACES the typing, so a claim checked against the dex entry is
        checked against a typing that is no longer on the field."""
        for mon, ttype in self._TERA_RE.findall(beat_text or ""):
            self._tera[mon.lower()] = ttype.title()

    # a move dismissed as bad//ineffective. Kept separate from _RESIST_RE
    # because these are not type words — they are verdicts on the move, and
    # they are what slipped past the chart check.
    _DUD_RE = re.compile(
        r"\b(liabilit(?:y|ies)|useless|pointless|ineffective|wasted|"
        r"did nothing|does nothing|accomplished nothing|no good)\b", re.I)

    def _contradicts_beat_effectiveness(self, line: str,
                                        item: dict) -> str | None:
        """True when the line calls a move ineffective that the beat just
        reported as super effective, or vice versa.

        Cheaper and stricter than the chart check and catches a case it
        cannot: measured live 2026-07-28, beat "Kommo-o's Shadow Claw hit
        Kingambit — super effective and a heavy hit" -> PRISM said "The Ghost
        Tera on Kingambit turned Shadow Claw into a liability". Ghost hits
        Ghost for 2x and the beat SAYS so in the same sentence, but "liability"
        is not type vocabulary so `_bad_type_claim` never looked.

        Binds on the MOVE, not the species: the beat states a polarity per
        move, so the move name is the one key that ties a claim to a fact.
        """
        beat = item.get("text") or ""
        if not beat or not line:
            return None
        try:
            from crystal_broadcast.game_data import DATA
            moves = [m["name"] for m in DATA.gen.moves.values() if "name" in m]
        except Exception:
            return None
        polarity = {}
        for mv in moves:
            if len(mv) < 4 or mv not in beat:
                continue
            # One move can be graded TWICE in a turn against different targets:
            # "Icicle Spear landed not very effective on Cinderace; Icicle Spear
            # knocked out Zapdos with super effective". Reading only the first
            # clause flagged a correct line about the second. Collect every
            # occurrence and refuse to rule when they disagree.
            seen = set()
            start = 0
            while True:
                i = beat.find(mv, start)
                if i < 0:
                    break
                start = i + len(mv)
                seg = beat[i:].split(";")[0]
                if self._SUPER_RE.search(seg):
                    seen.add("super")
                elif self._RESIST_RE.search(seg) or self._IMMUNE_RE.search(seg):
                    seen.add("weak")
            if len(seen) == 1:
                polarity[mv] = seen.pop()
        # the claim must attach to exactly one move the beat graded, or there
        # is no way to know which fact it contradicts
        named = [mv for mv in polarity if mv in line]
        named = [m for m in named if not any(m != o and m in o for o in named)]
        if len(named) != 1:
            return None
        mv = named[0]
        said_dud = bool(self._DUD_RE.search(line) or self._RESIST_RE.search(line)
                        or self._IMMUNE_RE.search(line))
        said_super = bool(self._SUPER_RE.search(line))
        if polarity[mv] == "super" and said_dud and not said_super:
            return (f"the beat reports {mv} as SUPER EFFECTIVE; the line calls "
                    f"it ineffective")
        if polarity[mv] == "weak" and said_super and not said_dud:
            return (f"the beat reports {mv} as resisted/ineffective; the line "
                    f"calls it super effective")
        return None

    def _bad_type_claim(self, line: str, item: dict) -> str | None:
        """True when the line asserts a type matchup the chart contradicts.

        The gap this closes: every other gate checks NOUNS or EVENTS — that a
        name exists, that a crit happened. A line can name only real entities,
        report only real events, and still be exactly backwards about WHY.
        Measured live 2026-07-28: "The Tera-Fairy on Ceruledge was a desperate
        attempt to resist the Icicle Spear crits" — Tera Fairy took Ice from
        0.5x to 1.0x, i.e. it DOUBLED the damage; it was blanking Scale Shot,
        which Fairy is immune to. Every noun in that sentence is real.

        Deliberately high-precision, low-recall: it fires a regeneration, so a
        false positive costs latency and a worse line. It only rules when the
        binding is unambiguous — exactly one move and one species named — and
        stays silent on anything it cannot resolve.
        """
        claim_resist = bool(self._RESIST_RE.search(line))
        claim_immune = bool(self._IMMUNE_RE.search(line))
        claim_super = bool(self._SUPER_RE.search(line))
        if not (claim_resist or claim_immune or claim_super):
            return None
        # more than one kind of claim in one line: cannot bind them apart
        if sum((claim_resist, claim_immune, claim_super)) > 1:
            return None
        # If the BEAT already asserts this polarity, the line is quoting the
        # record and the species it happens to name may be there for another
        # reason entirely. Measured on the corpus: beat "Iron Valiant's
        # Moonblast landed not very effective on Moltres. We switch to
        # Kommo-o" -> FRACTURE echoed "NOT VERY EFFECTIVE?" and named Kommo-o
        # as the SWITCH TARGET, and binding (Moonblast, Kommo-o) flagged a
        # correct line. Costs recall — a wrong claim about a third mon on a
        # turn that already reported an effectiveness slips through — which is
        # the right trade for a gate that forces a regeneration.
        beat_text = item.get("text") or ""
        if ((claim_resist and self._RESIST_RE.search(beat_text))
                or (claim_immune and self._IMMUNE_RE.search(beat_text))
                or (claim_super and self._SUPER_RE.search(beat_text))):
            return None
        try:
            from crystal_broadcast.game_data import DATA
            moves = [m["name"] for m in DATA.gen.moves.values() if "name" in m]
            species = [p["name"] for p in DATA.gen.pokedex.values()
                       if "name" in p]
        except Exception:
            return None
        named_moves = [m for m in moves if len(m) >= 4 and m in line]
        named_mons = [s for s in species if len(s) >= 4 and s in line]
        # drop names contained in a longer match ("Ice Spinner" vs "Ice Beam")
        named_moves = [m for m in named_moves
                       if not any(m != o and m in o for o in named_moves)]
        named_mons = [s for s in named_mons
                      if not any(s != o and s in o for o in named_mons)]
        # A claim about "the Tera-Fairy" often never names the mon — dropping
        # "on Ceruledge" was enough to escape the check entirely. When the beat
        # Terastallized exactly one mon and the line is talking about a Tera,
        # the subject is unambiguous, so bind to it.
        if not named_mons and re.search(r"\bTera\b|\bTera-", line, re.I):
            tera_here = self._TERA_RE.findall(item.get("text") or "")
            if len(tera_here) == 1:
                named_mons = [tera_here[0][0]]
        if len(named_moves) != 1 or len(named_mons) != 1:
            return None
        move, mon = named_moves[0], named_mons[0]
        atk = DATA.move_type(move)
        # a Terastallized mon IS its tera type; fall back to the dex entry
        tera = self._tera.get(mon.lower())
        dtypes = [tera] if tera else DATA.species_types(mon)
        mult = DATA.effectiveness(atk, dtypes)
        if mult is None:
            return None
        typing = tera + " (Tera)" if tera else "/".join(dtypes)
        if claim_resist and mult >= 1:
            return f"{mon} does NOT resist {move} ({atk} into {typing} is {mult}x)"
        if claim_immune and mult > 0:
            return f"{mon} is NOT immune to {move} ({atk} into {typing} is {mult}x)"
        if claim_super and mult <= 1:
            return f"{move} is NOT super effective on {mon} ({atk} into {typing} is {mult}x)"
        return None

    _CLEARED_RE = re.compile(
        r"\b(cleared|clearing|clear|removed|removing|remove|removal|spun|"
        r"spinning|swept|sweeping|gone|blown away)\b", re.I)
    _HAZARD_WORD_RE = re.compile(
        r"\b(hazards?|stealth rock|spikes|toxic spikes|sticky web|rocks)\b",
        re.I)

    @staticmethod
    def _fabricated_hazard_clear(line: str, item: dict) -> bool:
        """True when the line says hazards came off and the beat reports no
        such thing.

        Measured on take 26: Iron Treads clicked Rapid Spin for the chip and
        the Speed boost across a long attrition stretch with NOTHING on the
        field, and both voices narrated a hazard clear six times over — "we
        cleared the hazards", "the hazards are gone". The move's NAME was the
        only evidence, which is the Good as Gold failure wearing a different
        hat. The director now states when a spin had nothing to clear; this is
        the backstop for when it says nothing at all.
        """
        # "spin them away" reads as a clear, but bare "spin" cannot: the move
        # is named legitimately all the time ("the search is choosing Rapid
        # Spin"), so require the phrase, not the word.
        spun_away = re.search(r"\bspin\w*\b[^.]{0,30}\baway\b", line, re.I)
        if not ((Caster._CLEARED_RE.search(line) or spun_away)
                and Caster._HAZARD_WORD_RE.search(line)):
            return False
        # Ground on whether the beat mentions hazards AT ALL, not on whether
        # it says "cleared". A beat reporting rocks going UP makes "Rapid Spin
        # to clear them" a correct statement of intent — flagging that was a
        # false positive on the corpus. Only a beat that never mentions a
        # hazard leaves the claim unsupported, which is exactly the take-26
        # case: the record was silent and the move's name filled the gap.
        beat = item.get("text") or ""
        # A completed clear claimed while the Hazards: footer still lists
        # hazards is false on its face — the rocks are right there in the
        # record. This branch exists because adding that footer (2026-07-30)
        # made the "mentions hazards at all" test below abstain on EVERY
        # beat with hazards up, silently widening the hole it was guarding:
        # grounding text counts as a hazard mention. The footer repays that
        # with certainty the guard never had, so use it directly. Requires a
        # PAST-TENSE clear (intent stays legal, which was the original false
        # positive) and no real clearing event in the beat — when one side
        # genuinely got spun the beat says so and the other side's leftover
        # rocks must not convict a true line.
        if (re.search(r"\bHazards:", beat)
                and not re.search(r"\bclear\w*|\bspun\b|\bblown away\b|"
                                  r"\bswept\b|\bremov\w+", beat, re.I)
                and re.search(r"\b(?:cleared|removed|spun|swept|gone|"
                              r"blown away)\b", line, re.I)):
            return True
        return not Caster._HAZARD_WORD_RE.search(beat)

    @staticmethod
    def _fabricated_miss(line: str, item: dict) -> bool:
        """True when the line claims a move MISSED and the beat never said so.

        Measured live 2026-07-27: on a pre-move beat carrying no outcome at
        all ("We go for Stone Edge. Desk read: this is nearly gone"), BOTH
        voices invented a miss — "THAT MISSED!?" and "The miss on Stone Edge
        was all we had left" — and the next turn's beat confirmed the move
        LANDED. Same facts-of-record shape as _fabricated_crit: an outcome
        asserted that the record does not contain. A pre-move beat is the
        dangerous case, because the desk read tempts a narration of failure.
        """
        if not re.search(r"\b(missed|miss|whiff(?:ed)?)\b", line, re.I):
            return False
        beat = (item.get("text") or "").lower()
        return not re.search(r"\bmiss(?:ed)?\b|\bavoided\b|\bprotected\b",
                             beat)

    @staticmethod
    def _fabricated_synergy(line: str, item: dict) -> bool:
        """True when the line names a status-synergy ability the beat never
        flagged. The director only writes an ability name into the beat when
        a status genuinely fed it (Toxic on Poison Heal), so naming one the
        beat doesn't is an invented mechanic — measured live: the synergy
        framing leaked from an earlier turn's transcript onto Pecharunt,
        which has no such ability."""
        beat = (item.get("text") or "").lower()
        for ability in _SYNERGY_ABILITIES:
            if ability in line.lower() and ability not in beat:
                return True
        return False

    @staticmethod
    def _fabricated_immunity(line: str, item: dict) -> bool:
        """True when the beat is a no-effect outcome and the line credits an
        ability the beat did NOT name. A TYPE immunity (a Ghost ignoring Rapid
        Spin) carries no ability, so blaming Good as Gold is invented; but an
        ABILITY immunity names its cause in the beat ('Rotom's Levitate
        blocked it'), and crediting THAT is correct — so the check is
        name-in-line AND name-not-in-beat, exactly like _fabricated_synergy.
        The one hallucination the ability injection could reopen; regen keeps
        the original if the retry still trips."""
        beat = (item.get("text") or "").lower()
        if not any(k in beat for k in
                   ("no effect", "immune", "doesn't affect", "didn't affect")):
            return False
        low = line.lower()
        return any(name in low and name not in beat
                   for name, _f, _c in (item.get("_facts") or []))

    @staticmethod
    def _caption_phrasing(line: str) -> bool:
        """True for the caption-mode residual — restating the chosen move
        ('the search is opting for X') or reciting the desk read back. NOT the
        sanctioned attribution ('the search likes this line'); only the
        move-caption template. Recurs a few times a match even after the
        variety pass, so it gets a mechanical regen — PRISM's tic."""
        return bool(_CAPTION_RE.search(line))

    def _same_opener(self, persona: str, line: str, words: int = 4) -> bool:
        """True when `line` opens with the same first words as this
        persona's most recent line — the measured mode-collapse signature
        ('The search is opting...' x13 in one match)."""
        prev = next((ln for p, ln in reversed(self.transcript)
                     if p == persona), None)
        if prev is None:
            return False
        opener = lambda s: [w.lower().strip(".,!?") for w in s.split()[:words]]
        return opener(prev) == opener(line) and len(opener(line)) == words

    def _stall_repeats(self, line: str) -> bool:
        """True when a deep-think stall line reuses a distinctive image from
        ANY prior stall THIS MATCH (shares a content-word bigram) — the
        'threading a needle every third beat' problem a prompt rule alone
        couldn't stop, now caught across the whole match rather than a window
        of the last few (which a distant recurrence would slip past)."""
        bg = _content_bigrams(line)
        return any(bg & _content_bigrams(prev) for prev in self._match_stalls)

    async def speak(self, item: dict):
        if item["text"].startswith("[MATCH START]"):
            self.transcript.clear()
            self._beat_history.clear()
            self._match_stalls.clear()
            self._tera.clear()
        # record any Tera BEFORE the line is generated and checked: the beat
        # that announces it is usually the same beat being reacted to
        self._note_tera(item.get("text") or "")
        if item["text"].startswith("[RESULT]"):
            self._log_fact_summary()
            self._log_pace_summary()
        print(f"BEAT: {item['text']}", flush=True)
        self._beat_history.append(item["text"] or "")
        started = time.monotonic()
        if item.get("_queued") is not None:
            # how stale this beat is at the moment it starts being said
            self._pace_stats["stale_ms"].append(
                1000 * (started - item["_queued"]))
        self._pace_stats["voiced"] += 1
        spent = 0.0        # speech seconds committed by this beat
        speakers = _speakers(item["beats"], item["text"])
        # Both speech gates cut the LATER voice, and the handoff convention
        # puts the gremlin first on interrupts — so "drop the second" silently
        # meant "always drop PRISM". Measured on take 22: 25 PRISM drops
        # against 4, and a match that was 20 FRACTURE lines to 5. Rotate the
        # lead when the trailing voice has been starved, so the cost of a
        # tight budget lands on whoever has been speaking, not on whoever the
        # convention happens to put second.
        # Two triggers, one remedy. STARVED_AFTER catches a run of
        # consecutive cuts; it missed take 27 (5:1 PRISM cuts, all his lines
        # in the first 13 turns) because solo beats kept resetting his
        # counter between cuts. The DEFICIT_SWAP trigger reads the per-match
        # TALLY instead, which nothing resets. Speech mode only: in text
        # mode both voices always air, so the tally gap is content (solo
        # beats), not a budget artifact, and ordering stays byte-identical.
        starved = (self._drops.get(speakers[-1], 0) >= self.STARVED_AFTER)
        behind = (self.speech is not None
                  and self._spoken.get(speakers[0], 0)
                  - self._spoken.get(speakers[-1], 0) >= self.DEFICIT_SWAP)
        if len(speakers) > 1 and (starved or behind):
            speakers = list(reversed(speakers))
        deliberating = any(b.get("beat") == "deep_think"
                           for b in item.get("beats") or [])
        # fetch grounded facts once per beat if PRISM will speak (off the
        # event loop; cached across turns so it's usually free)
        if "PRISM" in speakers and item.get("_facts") is None:
            hud = item.get("hud") or {}
            abilities = [hud.get("us_ability"), hud.get("them_ability")]
            item["_facts"] = await asyncio.to_thread(
                self._gather_facts, item["text"],
                [a for a in abilities if a],
                self._strategy_consults(item))
        for idx, persona in enumerate(speakers):
            # Dual-beat pre-flight: a handoff pair is two utterances back to
            # back, so it can outlast the beat floor and push every later beat
            # late. The first voice always speaks (silence is worse than
            # overrun); the second is gated below, once its real length is
            # known. This cheap check only skips a generation that cannot
            # possibly fit. Both are inert without a duration source, so
            # text-only pacing is byte-for-byte unchanged.
            # CROSS-BEAT gate. The per-beat budget below only trims a handoff
            # pair; it cannot see speech still in flight from EARLIER beats,
            # and on take 13 that is what stacked up — busy stretches queued
            # faster than they could be spoken, so lines landed on top of
            # each other and drifted away from the picture. If audio is
            # already backed up past the floor, this line would be late as
            # well as overlapping, so drop it. MATCH START and RESULT are
            # exempt: missing the opening or the verdict is worse than late.
            limit = self._backlog_limit(item)
            if (self.speech is not None and self.speech_budget is not None
                    and limit is not None
                    and self._speaking_backlog() >= limit
                    and not item["text"].startswith(("[MATCH START]",
                                                     "[RESULT]"))):
                self._pace_stats["preflight_drops"] += 1
                print(f"caster: backlog dropped {persona} — "
                      f"{self._speaking_backlog():.1f}s of speech still "
                      f"queued (limit {limit:.1f}s)", flush=True)
                self._drops[persona] = self._drops.get(persona, 0) + 1
                break
            if (idx and self.speech_budget is not None
                    and spent >= self.speech_budget):
                self._pace_stats["preflight_drops"] += 1
                print(f"caster: pre-flight dropped {persona} — {spent:.2f}s "
                      f"already fills the {self.speech_budget:.1f}s budget",
                      flush=True)
                self._drops[persona] = self._drops.get(persona, 0) + 1
                break
            try:
                raw = await asyncio.to_thread(self._generate_sync,
                                              persona, item)
            except Exception as e:
                print(f"caster: generation failed for {persona}: {e!r}",
                      flush=True)
                continue
            line = _clean(raw)
            line = _fix_species_spelling(line, item)
            # facts-of-record guard: a fabricated crit is the common one
            # (a super-effective/heavy hit narrated as a "crit" that never
            # happened). If the line claims a crit the beat never stated,
            # regenerate once forbidding it.
            bad = self._ungrounded_entity(line, item) if line else None
            if bad:
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        f"Do NOT mention {bad} — nothing in the beat "
                        f"establishes it. Name only Pokemon, moves and "
                        f"abilities the beat itself reports.")
                    retry = _clean(raw)
                    retry = _fix_species_spelling(retry, item)
                    if retry and not self._ungrounded_entity(retry, item):
                        line = retry
                    else:
                        print(f"caster: ungrounded {bad!r} survived a regen "
                              f"({persona})", flush=True)
                except Exception:
                    pass
            if line and self._fabricated_miss(line, item):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "Do NOT say the move missed or whiffed — nothing in "
                        "the beat reports a miss. State only what the beat "
                        "reports.")
                    retry = _clean(raw)
                    retry = _fix_species_spelling(retry, item)
                    if retry and not self._fabricated_miss(retry, item):
                        line = retry
                except Exception:
                    pass
            # cheapest first: the beat's own words contradict the line
            contra = (self._contradicts_beat_effectiveness(line, item)
                      if line else None)
            if contra:
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        f"WRONG: {contra}. Read the beat again and do not "
                        "reverse what it reports.")
                    retry = _clean(raw)
                    retry = _fix_species_spelling(retry, item)
                    if retry and not self._contradicts_beat_effectiveness(
                            retry, item):
                        line = retry
                    else:
                        print(f"caster: beat contradiction survived a regen "
                              f"({contra})", flush=True)
                except Exception:
                    pass
            # type-claim guard: the chart contradicts the stated matchup
            bad_type = self._bad_type_claim(line, item) if line else None
            if bad_type:
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        f"WRONG: {bad_type}. Do not claim that matchup. State "
                        "only what the beat reports, and do not explain the "
                        "moment with a type interaction unless the beat says "
                        "so.")
                    retry = _clean(raw)
                    retry = _fix_species_spelling(retry, item)
                    if retry and not self._bad_type_claim(retry, item):
                        line = retry
                    else:
                        print(f"caster: bad type claim survived a regen "
                              f"({bad_type})", flush=True)
                except Exception:
                    pass
            # invented hazard clear: the move's name is not evidence
            if line and self._fabricated_hazard_clear(line, item):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "Do NOT say hazards were cleared, removed or spun "
                        "away — the beat reports no hazard leaving the field. "
                        "A move called Rapid Spin does not prove one was "
                        "there. React to what the beat actually reports.")
                    retry = _clean(raw)
                    retry = _fix_species_spelling(retry, item)
                    if retry and not self._fabricated_hazard_clear(retry, item):
                        line = retry
                    else:
                        print("caster: invented hazard clear survived a regen",
                              flush=True)
                except Exception:
                    pass
            if line and self._fabricated_crit(line, item):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "Do NOT call this a critical hit or crit — nothing "
                        "in the beat says a critical hit happened. State only "
                        "what the beat reports.")
                    retry = _clean(raw)
                    if retry and not self._fabricated_crit(retry, item):
                        line = retry
                except Exception:
                    pass
            # ability-state guard: Supreme Overlord power-claims with nobody
            # fainted (take 71 T2 — zero stacks existed); regen with the rule
            if line and self._overlord_state_claim(line, item):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "Supreme Overlord has ZERO stacks right now — it "
                        "scales with fainted allies and nobody has fainted. "
                        "Do not credit it with any power yet.")
                    retry = _clean(raw)
                    if retry and not self._overlord_state_claim(retry, item):
                        line = retry
                except Exception:
                    pass
            # field-state guards: weather/screens/stage claims the record
            # doesn't support anywhere in the beat; regen with the rule
            if line and self._weather_state_claim(line, item):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "Do NOT narrate active weather — the beat reports "
                        "none of the weather you named. Past or hypothetical "
                        "weather must be clearly marked as past or "
                        "hypothetical. React to what the beat actually "
                        "reports.")
                    retry = _clean(raw)
                    if retry and not self._weather_state_claim(retry, item):
                        line = retry
                except Exception:
                    pass
            if line and self._screens_state_claim(line, item):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "Do NOT say screens are up — the beat reports no "
                        "Reflect, Light Screen or Aurora Veil active on "
                        "either side. React to what the beat actually "
                        "reports.")
                    retry = _clean(raw)
                    if retry and not self._screens_state_claim(retry, item):
                        line = retry
                except Exception:
                    pass
            if line and self._boost_state_claim(line, item):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "Do NOT state a stat-stage number like +2 — the "
                        "beat reports no active stat boosts. State only "
                        "what the beat reports.")
                    retry = _clean(raw)
                    if retry and not self._boost_state_claim(retry, item):
                        line = retry
                except Exception:
                    pass
            # KO-dismissal guard: the record says dead, the line says fine
            if line and self._ko_dismissal_claim(line, item):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "That move KNOCKED THE TARGET OUT — it did not "
                        "survive, it was not unfazed, and the move did not "
                        "do nothing. A not-very-effective hit can still be "
                        "lethal. React to the knockout.")
                    retry = _clean(raw)
                    if retry and not self._ko_dismissal_claim(retry, item):
                        line = retry
                except Exception:
                    pass
            # item-polarity guard: an item that ACTIVATED read as a loss
            if line and self._item_polarity_claim(line, item):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "That item was CONSUMED BY WORKING — spending it is "
                        "what turns its effect ON. Nothing was lost and "
                        "nothing was taken. Do not frame it as a loss, a "
                        "waste, or an advantage gone.")
                    retry = _clean(raw)
                    if retry and not self._item_polarity_claim(retry, item):
                        line = retry
                except Exception:
                    pass
            # fail-semantics guard: a failure explained by a type
            # interaction, an immunity or a miss — the founding case of
            # this family (take 30 T5), which the prompt fence alone
            # did not hold
            if line and self._fail_mechanism_claim(line, item):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "The move FAILED — that is not a miss, not an "
                        "immunity and not a type matchup, and the beat "
                        "gives no reason for it. Do not explain the "
                        "failure with typing, immunity or the dice. Say "
                        "only that it failed, or reason from what the "
                        "TARGET did.")
                    retry = _clean(raw)
                    if retry and not self._fail_mechanism_claim(retry, item):
                        line = retry
                except Exception:
                    pass
            # boost-polarity guard: a stat drop credited to the wrong side
            if line and self._boost_polarity_claim(line, item):
                m = self._BOOST_FOOTER.search(item.get("text") or "")
                frags = (m.group(1) if m else "")
                if "their" in frags:
                    note = ("The stat drop in this beat is on THEIR mon — "
                            "WE cut it, and that is damage WE did. Do not "
                            "frame it as harming us.")
                else:
                    note = ("The stat drop in this beat is on OUR mon — "
                            "THEY cut it. Do not frame it as harming them.")
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item, note)
                    retry = _clean(raw)
                    if retry and not self._boost_polarity_claim(retry, item):
                        line = retry
                except Exception:
                    pass
            if line and self._hazard_state_claim(line, item):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "Do NOT talk about entry hazards being on the "
                        "field — the beat reports no Stealth Rock, Spikes, "
                        "Toxic Spikes or Sticky Web up on either side. "
                        "React to what the beat actually reports.")
                    retry = _clean(raw)
                    if retry and not self._hazard_state_claim(retry, item):
                        line = retry
                except Exception:
                    pass
            # luck-polarity guard: a dice grievance pointed the wrong way
            # round (their miss rendered as our persecution, or ours as a
            # payback) — the beat states whose move missed
            if line and self._luck_polarity_claim(line, item):
                if "their" in self._beat_miss_directions(item):
                    note = ("The miss in this beat went against THEM — "
                            "THEIR move missed. That is luck in OUR favor. "
                            "Do not frame it as the dice or the server "
                            "working against us.")
                else:
                    note = ("The miss in this beat was OURS — OUR move "
                            "missed. That is luck against US. Do not frame "
                            "it as the dice paying us back or favoring us.")
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item, note)
                    retry = _clean(raw)
                    if retry and not self._luck_polarity_claim(retry, item):
                        line = retry
                except Exception:
                    pass
            # miss-for-immunity guard: a no-effect narrated as a miss/dodge
            # corrupts both the luck ledger and the mechanics (3 sightings
            # in one hunt); regen once with the distinction spelled out
            if line and self._miss_for_immunity(line, item):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "Nothing MISSED — the beat says the move had NO "
                        "EFFECT, which is a type immunity: the type chart, "
                        "not the dice and not a dodge. Say it was immune or "
                        "that it did nothing.")
                    retry = _clean(raw)
                    if retry and not self._miss_for_immunity(retry, item):
                        line = retry
                except Exception:
                    pass
            # fabricated-recoil guard: take 49 T16/T19, PRISM blamed
            # "recoil" for Headlong Rush twice — it has none, it drops the
            # user's defenses, and the beat stated the real effect both
            # times. Same facts-of-record shape as the crit guard above.
            if line and self._fabricated_recoil(line, item):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "Do NOT mention recoil — nothing in the beat reports "
                        "recoil damage. The beat states the move's real "
                        "effects; use those.")
                    retry = _clean(raw)
                    if retry and not self._fabricated_recoil(retry, item):
                        line = retry
                except Exception:
                    pass
            # fabricated-synergy guard: a status-boon read the beat never
            # flagged (the framing leaks from an earlier turn's transcript)
            if line and self._fabricated_synergy(line, item):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "Do NOT say any ability turns this status into an "
                        "advantage — the beat does not report that. Treat the "
                        "status as an ordinary status.")
                    retry = _clean(raw)
                    if retry and not self._fabricated_synergy(retry, item):
                        line = retry
                except Exception:
                    pass
            # fabricated-immunity guard: a no-effect/immune outcome blamed on
            # the defender's (real, now-listed) ability — the exact spinblock
            # hallucination the ability injection could reopen
            if line and self._fabricated_immunity(line, item):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "Do NOT credit any ability for this — the move had no "
                        "effect because of a TYPE matchup, not the defender's "
                        "ability. Just report that it did nothing.")
                    retry = _clean(raw)
                    if retry and not self._fabricated_immunity(retry, item):
                        line = retry
                except Exception:
                    pass
            # claimed-call guard: "I told you / I called it" about something
            # the speaker never previously mentioned — a fabricated PAST,
            # same family as the invented turn numbers. Take 27 T14:
            # FRACTURE's first-ever mention of Icicle Spear opened "I TOLD
            # YOU THAT ICICLE SPEAR WAS THE FINAL NAIL" — the call was
            # PRISM's, one beat earlier.
            stolen = self._stolen_call(line, persona) if line else None
            if stolen:
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        f"You have NOT previously said anything about "
                        f"{stolen} this match — do not claim you told "
                        f"anyone, called it, or said so earlier. React to "
                        f"it fresh, in the present.")
                    retry = _clean(raw)
                    retry = _fix_species_spelling(retry, item)
                    if retry and not self._stolen_call(retry, persona):
                        line = retry
                    else:
                        print(f"caster: stolen call ({stolen}) survived a "
                              f"regen", flush=True)
                except Exception:
                    pass
            # caption-mode guard: PRISM restating the move or reciting the
            # desk read back ('the search is opting for X') — regen once for
            # meaning over caption; keep the retry only if it clears
            if line and persona == "PRISM" and self._caption_phrasing(line):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "Do NOT say 'the search is opting for' or recite the "
                        "desk read back — the audience already sees the move "
                        "and the meter. Say what it MEANS, don't caption it.",
                        0.2)
                    retry = _clean(raw)
                    if retry and not self._caption_phrasing(retry):
                        line = retry
                except Exception:
                    pass
            # opener-repetition guard: one hotter retry with an explicit
            # nudge; keep whatever the retry gives (never loop)
            if line and self._same_opener(persona, line):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "Do NOT start with the words your previous line "
                        "started with; change the sentence shape entirely.",
                        0.3)
                    retry = _clean(raw)
                    if retry:
                        line = retry
                except Exception:
                    pass
            # stall-repeat guard: block FRACTURE reusing a recent thinking-beat
            # image ('threading a needle' every few beats); regenerate once
            if (line and deliberating and persona == "FRACTURE"
                    and self._stall_repeats(line)):
                try:
                    used = "; ".join(f'"{s}"' for s in self._match_stalls[-8:])
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        f"You already used these stall images this match: "
                        f"{used}. Use a COMPLETELY different metaphor — reuse "
                        "none of their wording.", 0.3)
                    retry = _clean(raw)
                    if retry and not self._stall_repeats(retry):
                        line = retry
                except Exception:
                    pass
            # FACTS-OF-RECORD BACKSTOP (user call, 2026-07-30): every guard
            # above regenerates once and, when the retry ALSO violated, used
            # to air the original — a known-false line reaching the
            # audience. The contract is "never invented", so on double-fail
            # the line is dropped with pre-flight-drop semantics instead
            # (no transcript, no ledger, no speech — no trace). Style
            # guards (caption, opener, stall) stay lenient on purpose: a
            # stale metaphor is not a lie.
            if line:
                fact_checks = (
                    ("ungrounded entity",
                     lambda: self._ungrounded_entity(line, item)),
                    ("beat contradiction",
                     lambda: self._contradicts_beat_effectiveness(line, item)),
                    ("bad type claim",
                     lambda: self._bad_type_claim(line, item)),
                    ("fabricated miss",
                     lambda: self._fabricated_miss(line, item)),
                    ("invented hazard clear",
                     lambda: self._fabricated_hazard_clear(line, item)),
                    ("fabricated crit",
                     lambda: self._fabricated_crit(line, item)),
                    ("miss-for-immunity",
                     lambda: self._miss_for_immunity(line, item)),
                    ("overlord state claim",
                     lambda: self._overlord_state_claim(line, item)),
                    ("weather state claim",
                     lambda: self._weather_state_claim(line, item)),
                    ("screens state claim",
                     lambda: self._screens_state_claim(line, item)),
                    ("boost state claim",
                     lambda: self._boost_state_claim(line, item)),
                    ("luck polarity claim",
                     lambda: self._luck_polarity_claim(line, item)),
                    ("hazard state claim",
                     lambda: self._hazard_state_claim(line, item)),
                    ("boost polarity claim",
                     lambda: self._boost_polarity_claim(line, item)),
                    ("fail mechanism claim",
                     lambda: self._fail_mechanism_claim(line, item)),
                    ("item polarity claim",
                     lambda: self._item_polarity_claim(line, item)),
                    ("ko dismissal claim",
                     lambda: self._ko_dismissal_claim(line, item)),
                    ("fabricated recoil",
                     lambda: self._fabricated_recoil(line, item)),
                    ("fabricated synergy",
                     lambda: self._fabricated_synergy(line, item)),
                    ("fabricated immunity",
                     lambda: self._fabricated_immunity(line, item)),
                    ("stolen call",
                     lambda: self._stolen_call(line, persona)),
                )
                # Several guards return WHAT offended (the ungrounded
                # species, the false type claim) rather than a bare True,
                # and throwing that away made drops undiagnosable: a
                # MATCH START line reading "Kyurem holds the lead..." was
                # dropped for an ungrounded entity that sat past the 90-char
                # excerpt, so the log could not say whether the fire was
                # even correct. Name the offender and show more of the line.
                offense = None
                for name, chk in fact_checks:
                    found = chk()
                    if found:
                        offense = (name if found is True
                                   else f"{name}: {found}")
                        break
                if offense:
                    print(f"caster: DROPPED {persona} line — {offense} "
                          f"survived regeneration: {line[:160]!r}",
                          flush=True)
                    self._drops[persona] = self._drops.get(persona, 0) + 1
                    continue
            if not line:
                print(f"caster: {persona} line sanitized to empty, "
                      f"dropped: {raw[:90]!r}", flush=True)
                continue
            secs = self._speech_seconds(persona, line)
            # The real pre-flight: only now is this line's true length known,
            # so gate on what it WOULD cost rather than on what is already
            # spent. Checking "budget already exhausted" instead lets a pair
            # sail past the floor whenever the first voice fits on its own,
            # which is the common case and the one worth catching.
            #
            # This MUST run before any state below it. A dropped line has to
            # leave no trace: the duo transcript is shared so the other voice
            # can answer it, and a correction that never aired would otherwise
            # get answered on air next beat. Same for the stall ledger, which
            # would burn a metaphor FRACTURE never actually said.
            if (idx and secs is not None and self.speech_budget is not None
                    and spent + secs > self.speech_budget):
                self._pace_stats["preflight_drops"] += 1
                print(f"caster: pre-flight dropped {persona} — {spent:.2f}s + "
                      f"{secs:.2f}s exceeds the {self.speech_budget:.1f}s "
                      f"budget", flush=True)
                break
            self._drops[persona] = 0      # spoke: no longer starved
            self._spoken[persona] = self._spoken.get(persona, 0) + 1
            self._match_lines.setdefault(persona, []).append(line)
            self.transcript.append((persona, line))
            if deliberating and persona == "FRACTURE":
                self._match_stalls.append(line)
            print(f"{persona}: {line}", flush=True)
            # cite only the facts PRISM actually referenced (mechanic named
            # in the line) — the sources behind what he just said
            citations = None
            if persona == "PRISM" and item.get("_facts"):
                low = line.lower()
                citations = [cite for name, _f, cite in item["_facts"]
                             if name in low or cite["label"].lower() in low]
            # PTS gate: the line is written, now wait for the viewer to
            # actually reach this turn. Held here rather than before
            # generation so the lag pays for the generation. [MATCH START]
            # has no turn and goes out at once; [RESULT] waits for the end of
            # the battle to be presented, or it spoils the finish while the
            # audience is still watching the last exchange.
            if self.pts is not None:
                if item["text"].startswith("[MATCH START]"):
                    # the camera gate: the engine starts ~20s before the
                    # frame and recorder exist, and an opening line spoken
                    # then is sliced off the video (take 54: a REAL crit
                    # callout read as imagined because its referent never
                    # aired)
                    held = await self.pts.wait_for_first()
                else:
                    held = await self.pts.wait_for(
                        _turn_of(item["text"]),
                        final=item["text"].startswith("[RESULT]"))
                if held > 0.05:
                    self._pace_stats["pts_held_s"].append(held)
            await self.publish(item["text"], persona, line, item["hud"],
                               citations)
            # Speak it, if a voice is attached. AFTER the PTS gate on purpose:
            # the whole point of the clock is that a line lands when the viewer
            # reaches the moment, and audio that ran early would undo it.
            # Rendering happens in a thread (RTF ~0.5, so about half the line's
            # length) and playback is fire-and-forget, so awaiting this costs
            # the render only. The real duration comes back and replaces the
            # reading-rate estimate for the pacing accounting below.
            if self.speech is not None:
                reg_beat = next((b for b in (item.get("beats") or [])
                                 if b.get("register")), None)
                spoken = await asyncio.to_thread(
                    self.speech.speak, persona, line,
                    reg_beat.get("register") if reg_beat else None)
                if spoken:
                    secs = spoken
            if secs is not None:
                spent += secs
                self._pace_stats["speech_s"].append(secs)
                # queue behind whatever is still playing, so the deadline is
                # when THIS line finishes, not when it was handed over
                self._speaking_until = max(self._speaking_until,
                                           time.monotonic()) + secs

        self._pace_stats["turnaround_ms"].append(
            1000 * (time.monotonic() - started))
        if self.speech_budget is not None and spent > self.speech_budget:
            self._pace_stats["overruns"] += 1

    async def worker(self):
        while True:
            await self._wake.wait()
            self._wake.clear()
            while (self._pending_framing or self._pending_turn
                   or self._pending_queue):
                if self._pending_framing:
                    item = self._pending_framing.popleft()
                elif self._pending_queue:
                    item = self._pending_queue.popleft()
                else:
                    item, self._pending_turn = self._pending_turn, None
                await self.speak(item)


async def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--upstream", default=DEFAULT_UPSTREAM,
                    help="OpenAI-compatible endpoint (the no-think proxy)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--grudges", default=str(DEFAULT_GRUDGES),
                    help="grudge-ledger JSON (FRACTURE's Book of Grudges); "
                         "absent = no grudges, graceful")
    ap.add_argument("--speech-budget", type=float, default=None,
                    metavar="SECONDS",
                    help="per-beat wall-clock speech allowance (the beat "
                         "floor). Gates the second voice of a handoff pair. "
                         "Inert until a speech layer supplies durations")
    ap.add_argument("--speech", action="store_true",
                    help="speak the lines through the duo TTS service "
                         "(caster-avatars/tts_server.py). OFF by default: "
                         "audio is opt-in and text stays the default output. "
                         "If the service is unreachable the broadcast runs "
                         "silently rather than losing lines")
    ap.add_argument("--speech-url", default=None,
                    help="duo TTS endpoint (default http://127.0.0.1:8133)")
    ap.add_argument("--speech-out", default=None, metavar="DIR",
                    help="also keep the rendered wavs here, for a recorded "
                         "take or a viseme pass")
    ap.add_argument("--speech-sink", default=None, metavar="SINK",
                    help="play into this sink instead of the default. A null sink keeps the recorder's capture while leaving the room silent")
    ap.add_argument("--no-play", action="store_true",
                    help="render speech but do not play it (useful when the "
                         "recorder is capturing a different audio sink)")
    ap.add_argument("--pts-url", default=None,
                    help="presentation-clock feed (e.g. ws://127.0.0.1:8132) "
                         "— hold each finished line until the VIEWER reaches "
                         "its turn. Off by default: publishing is unchanged.")
    ap.add_argument("--pts-max-hold", type=float, default=180.0,
                    help="never hold a beat longer than this; a closed "
                         "broadcast page must not mute the commentary")
    ap.add_argument("--expert", default=DEFAULT_EXPERT,
                    help="grounded-rag /retrieve base URL for PRISM's fact "
                         "injection ('off' disables)")
    args = ap.parse_args()

    pts = (PresentationClock(args.pts_url, max_hold=args.pts_max_hold)
           if args.pts_url else None)
    speech = None
    if args.speech:
        from crystal_broadcast.speech import DEFAULT_URL, Speech
        speech = Speech(url=args.speech_url or DEFAULT_URL,
                        play=not args.no_play, out_dir=args.speech_out,
                        sink=args.speech_sink)
        # say so at startup rather than discovering it line by line: a silent
        # broadcast that was MEANT to have audio is the confusing failure
        if speech.available():
            print(f"caster: speech ON via {speech.url}", flush=True)
        else:
            print(f"caster: --speech given but no service at {speech.url} — "
                  f"lines will publish as text only", flush=True)
    caster = Caster(args.upstream, args.model, grudge_path=args.grudges,
                    expert_url=None if args.expert == "off" else args.expert,
                    speech_budget=args.speech_budget,
                    duration_fn=speech.duration_fn if speech else None,
                    speech=speech, pts=pts)
    if pts is not None:
        pts.start()
        print(f"caster: PTS scheduling on — holding lines until the viewer "
              f"reaches their turn (feed {args.pts_url}, max hold "
              f"{args.pts_max_hold:.0f}s)", flush=True)
    if caster.grudges.ledger:
        print(f"caster: loaded {len(caster.grudges.ledger)} grudges "
              f"from {args.grudges}", flush=True)
    if caster.expert_url:
        caster._expert_up = caster._ping_expert()
        state = "reachable" if caster._expert_up else "UNREACHABLE"
        print(f"caster: PRISM fact injection via {caster.expert_url} "
              f"(expert {state}); preview cache-warm on", flush=True)
    async with websockets.serve(caster.handle, "127.0.0.1", args.port):
        print(f"caster: duo live on ws://127.0.0.1:{args.port} "
              f"(model {args.model} via {args.upstream})", flush=True)
        # log the in-progress match's RAG summary on shutdown too — a
        # timed-out game never emits [RESULT], and the demo harness SIGTERMs
        # the caster to tear down, so without this the numbers are lost
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        worker = asyncio.create_task(caster.worker())
        await stop.wait()
        if any(caster._fact_stats.values()):
            caster._log_fact_summary()
        # same reason: a timed-out match never emits [RESULT], and the pacing
        # numbers are the point of running it
        caster._log_pace_summary()
        worker.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
