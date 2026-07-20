#!/usr/bin/env python3
"""Caster service: the duo's voice box, a headless drop-in for AIRI.

Speaks AIRI's WS protocol on ws://127.0.0.1:8131 so the existing plumbing
works unchanged on both sides of it:
  * gen9_player --airi --airi-url ws://127.0.0.1:8131/ws delivers beats
    through the stock AiriBridge (module:authenticate -> input:text);
    structured director beats + HUD ride in extra data fields real AIRI
    would ignore.
  * commentary_overlay / airi_bridge --watch subscribe here exactly as
    they did to AIRI and receive output:gen-ai:chat:complete envelopes
    (superjson-wrapped) per finished line, with data.persona attached.

For each beat the caster picks the speaking persona(s) from the director's
routing (handoff order for dual beats), builds a per-persona prompt (the
contract file + a bounded duo transcript + the beat + register direction),
and generates through the ollama no-think proxy. The duo transcript is
shared, so PRISM sees FRACTURE's line before correcting it — the
correction loop is an ordered pair of generations, not a prompt prayer.

Latency policy is skip-don't-queue: one pending slot per priority class;
a newer turn beat replaces an unspoken older one. MATCH START / RESULT
always speak.

Run:  .venv/bin/python showdown/caster.py [--port 8131]
      [--upstream http://127.0.0.1:11435] [--model ...]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import websockets

import re

from showdown.airi_bridge import _sanitize, _unwrap
from showdown.grudge_ledger import GrudgeLedger

# the model mimics the transcript format and prefixes its own line with a
# speaker label (sometimes stacked: "PRISM: PRISM: ..."); strip them all
_SELF_LABEL = re.compile(r"^\s*(?:(?:PRISM|FRACTURE)\s*:\s*)+", re.I)

PERSONA_DIR = Path(__file__).parent / "personas"
DEFAULT_PORT = 8131
DEFAULT_UPSTREAM = "http://127.0.0.1:11435"
DEFAULT_MODEL = "gemma4:26b-a4b-it-q4_K_M"
DEFAULT_GRUDGES = Path(__file__).parent / "grudges.json"

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
                 grudge_path: str | None = None):
        self.upstream = upstream
        self.model = model
        self.prompts = {p: (PERSONA_DIR / f).read_text()
                        for p, f in _PERSONA_FILE.items()}
        self.grudges = GrudgeLedger.load(grudge_path)
        self.transcript: deque = deque(maxlen=12)
        self.clients: set = set()
        # skip-don't-queue: newest unspoken turn beat wins; framing beats
        # (MATCH START / RESULT) queue separately and always speak
        self._pending_turn: dict | None = None
        self._pending_framing: deque = deque()
        self._wake = asyncio.Event()

    # --- intake (AIRI-protocol server) ---------------------------------
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
                    # accept anyone local, mirror AIRI's ack envelope so
                    # AiriBridge's handshake succeeds unchanged
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
                            "hud": data.get("hud")}
                    if (text.startswith("[MATCH START]")
                            or text.startswith("[RESULT]")):
                        self._pending_framing.append(item)
                    else:
                        self._pending_turn = item  # replace unspoken older
                    self._wake.set()
        finally:
            self.clients.discard(ws)

    # --- output (AIRI-shaped envelopes) ---------------------------------
    async def publish(self, beat_text: str, persona: str, line: str,
                      hud: dict | None):
        envelope = json.dumps({"json": {
            "type": "output:gen-ai:chat:complete",
            "data": {"text": beat_text, "persona": persona, "hud": hud,
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
        direction += (" One or two short spoken sentences, react now. "
                      "Output only the line itself.")
        if nudge:
            direction += f" {nudge}"
        user = ""
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

    async def speak(self, item: dict):
        if item["text"].startswith("[MATCH START]"):
            self.transcript.clear()
        for persona in _speakers(item["beats"], item["text"]):
            try:
                raw = await asyncio.to_thread(self._generate_sync,
                                              persona, item)
            except Exception as e:
                print(f"caster: generation failed for {persona}: {e!r}",
                      flush=True)
                continue
            line = _sanitize(_SELF_LABEL.sub("", raw.strip()))
            # facts-of-record guard: a fabricated crit is the common one
            # (a super-effective/heavy hit narrated as a "crit" that never
            # happened). If the line claims a crit the beat never stated,
            # regenerate once forbidding it.
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
            if not line:
                print(f"caster: {persona} line sanitized to empty, "
                      f"dropped: {raw[:90]!r}", flush=True)
                continue
            self.transcript.append((persona, line))
            print(f"{persona}: {line}", flush=True)
            await self.publish(item["text"], persona, line, item["hud"])

    async def worker(self):
        while True:
            await self._wake.wait()
            self._wake.clear()
            while self._pending_framing or self._pending_turn:
                if self._pending_framing:
                    item = self._pending_framing.popleft()
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
    args = ap.parse_args()

    caster = Caster(args.upstream, args.model, grudge_path=args.grudges)
    if caster.grudges.ledger:
        print(f"caster: loaded {len(caster.grudges.ledger)} grudges "
              f"from {args.grudges}", flush=True)
    async with websockets.serve(caster.handle, "127.0.0.1", args.port):
        print(f"caster: duo live on ws://127.0.0.1:{args.port} "
              f"(model {args.model} via {args.upstream})", flush=True)
        await caster.worker()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
