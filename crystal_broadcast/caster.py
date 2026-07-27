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

    def repl(m):
        tok = m.group(0)
        if tok in known:
            return tok
        near = difflib.get_close_matches(tok, known, n=1, cutoff=0.8)
        return near[0] if near else tok

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

    def _retrieve_fact(self, name: str, warm: bool = False):
        """Pull a mechanic's real effect from the expert (/retrieve) ->
        (fact_text, citation) or None. citation = {'label','corpus'} for the
        on-screen source chip. Cached (mechanics are static); a down/absent
        expert degrades to None so PRISM reasons as before — never raises.
        `warm=True` marks a team-preview pre-fetch: it fills the cache but is
        kept out of the in-game counters, so 'cold_fetch' measures only the
        round-trips warming failed to pre-empt."""
        if name in self._fact_cache:
            if not warm:
                self._fact_stats["cache_hit"] += 1
            return self._fact_cache[name]
        result = None
        try:
            body = json.dumps(
                {"question": f"what does {name} do in Pokemon"}).encode()
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
                self._fact_cache[name] = result   # cache successes only
        except Exception:
            result = None
        # warm successes are tallied by _warm_cache; keep them out of the
        # in-game buckets so cold_fetch stays a clean 'warming missed this'
        if result and not warm:
            self._fact_stats["cold_fetch"] += 1
        elif not result and not warm:
            self._fact_stats["miss"] += 1
        return result

    def _gather_facts(self, beat_text: str, abilities=()) -> list:
        """(name, fact_text, citation) for each curated mechanic named in the
        beat PLUS the two active mons' abilities (already resolved upstream to
        single-known values), capped so prompt and latency stay bounded.
        Injecting the active abilities is the fix for PRISM reasoning from a
        NAME the beat didn't state — blaming Good as Gold for a Ghost's
        spinblock. Worker-thread only."""
        if not self.expert_url:
            return []
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
        facts = []
        for name in (beat_hits + ability_hits)[:3]:
            got = self._retrieve_fact(name)
            if got:
                facts.append((name, got[0], got[1]))
        if facts:
            self._fact_stats["injected"] += len(facts)
            self._fact_stats["beats_with_facts"] += 1
        return facts

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
                     "below. Use these facts to add meaning to what DID happen "
                     "— why a mon survived, why a status is actually a boon — "
                     "never to invent a reason a move failed. Context, not a "
                     "mandate: reach for them only when they explain the "
                     f"moment, not every line:\n{lines}\n\n")
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
        deliberating = any(b.get("beat") == "deep_think"
                           for b in item.get("beats") or [])
        # fetch grounded facts once per beat if PRISM will speak (off the
        # event loop; cached across turns so it's usually free)
        if "PRISM" in speakers and item.get("_facts") is None:
            hud = item.get("hud") or {}
            abilities = [hud.get("us_ability"), hud.get("them_ability")]
            item["_facts"] = await asyncio.to_thread(
                self._gather_facts, item["text"],
                [a for a in abilities if a])
        for idx, persona in enumerate(speakers):
            # Dual-beat pre-flight: a handoff pair is two utterances back to
            # back, so it can outlast the beat floor and push every later beat
            # late. The first voice always speaks (silence is worse than
            # overrun); the second is gated below, once its real length is
            # known. This cheap check only skips a generation that cannot
            # possibly fit. Both are inert without a duration source, so
            # text-only pacing is byte-for-byte unchanged.
            if (idx and self.speech_budget is not None
                    and spent >= self.speech_budget):
                self._pace_stats["preflight_drops"] += 1
                print(f"caster: pre-flight dropped {persona} — {spent:.2f}s "
                      f"already fills the {self.speech_budget:.1f}s budget",
                      flush=True)
                break
            try:
                raw = await asyncio.to_thread(self._generate_sync,
                                              persona, item)
            except Exception as e:
                print(f"caster: generation failed for {persona}: {e!r}",
                      flush=True)
                continue
            line = _sanitize(_SELF_LABEL.sub("", raw.strip()))
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
                    retry = _sanitize(_SELF_LABEL.sub("", raw.strip()))
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
                    retry = _sanitize(_SELF_LABEL.sub("", raw.strip()))
                    retry = _fix_species_spelling(retry, item)
                    if retry and not self._fabricated_miss(retry, item):
                        line = retry
                except Exception:
                    pass
            if line and self._fabricated_crit(line, item):
                try:
                    raw = await asyncio.to_thread(
                        self._generate_sync, persona, item,
                        "Do NOT call this a critical hit or crit — nothing "
                        "in the beat says a critical hit happened. State only "
                        "what the beat reports.")
                    retry = _sanitize(_SELF_LABEL.sub("", raw.strip()))
                    if retry and not self._fabricated_crit(retry, item):
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
                    retry = _sanitize(_SELF_LABEL.sub("", raw.strip()))
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
                    retry = _sanitize(_SELF_LABEL.sub("", raw.strip()))
                    if retry and not self._fabricated_immunity(retry, item):
                        line = retry
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
                    retry = _sanitize(_SELF_LABEL.sub("", raw.strip()))
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
                    retry = _sanitize(_SELF_LABEL.sub("", raw.strip()))
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
                    retry = _sanitize(_SELF_LABEL.sub("", raw.strip()))
                    if retry and not self._stall_repeats(retry):
                        line = retry
                except Exception:
                    pass
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
                held = await self.pts.wait_for(
                    _turn_of(item["text"]),
                    final=item["text"].startswith("[RESULT]"))
                if held > 0.05:
                    self._pace_stats["pts_held_s"].append(held)
            await self.publish(item["text"], persona, line, item["hud"],
                               citations)
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
    caster = Caster(args.upstream, args.model, grudge_path=args.grudges,
                    expert_url=None if args.expert == "off" else args.expert,
                    speech_budget=args.speech_budget, pts=pts)
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
