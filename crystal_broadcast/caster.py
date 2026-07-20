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

# the model mimics the transcript format and prefixes its own line with a
# speaker label (sometimes stacked: "PRISM: PRISM: ..."); strip them all
_SELF_LABEL = re.compile(r"^\s*(?:(?:PRISM|FRACTURE)\s*:\s*)+", re.I)

PERSONA_DIR = Path(__file__).parent / "personas"
DEFAULT_PORT = 8131
DEFAULT_UPSTREAM = "http://127.0.0.1:11435"
DEFAULT_MODEL = "gemma4:26b-a4b-it-q4_K_M"

# generation knobs per persona: FRACTURE runs hot and short, PRISM cool
# and a touch longer
_GEN = {
    "FRACTURE": {"temperature": 1.0, "max_tokens": 90},
    "PRISM": {"temperature": 0.6, "max_tokens": 140},
}
_PERSONA_FILE = {"PRISM": "prism.txt", "FRACTURE": "fracture.txt"}

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
    if persona == "either":
        return (["FRACTURE"] if top.get("priority") == "interrupt"
                else ["PRISM"])
    if persona == "none":
        return []
    return [{"gremlin": "FRACTURE", "analyst": "PRISM"}[persona]]


class Caster:
    def __init__(self, upstream: str, model: str):
        self.upstream = upstream
        self.model = model
        self.prompts = {p: (PERSONA_DIR / f).read_text()
                        for p, f in _PERSONA_FILE.items()}
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
    def _prompt(self, persona: str, item: dict) -> list[dict]:
        beats = item["beats"]
        register = next((b.get("register") for b in beats
                         if b.get("register")), None)
        transcript = "\n".join(f"{p}: {ln}" for p, ln in self.transcript)
        direction = f"You are {persona}."
        if register:
            direction += f" Register: {register}."
        direction += (" One or two short spoken sentences, react now. "
                      "Output only the line itself.")
        user = ""
        if transcript:
            user += f"Broadcast so far:\n{transcript}\n\n"
        user += f"New beat from the director:\n{item['text']}\n({direction})"
        return [{"role": "system", "content": self.prompts[persona]},
                {"role": "user", "content": user}]

    def _generate_sync(self, persona: str, item: dict) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": self._prompt(persona, item),
            "stream": False,
            **_GEN[persona],
        }).encode()
        req = urllib.request.Request(
            f"{self.upstream}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.load(resp)
        return out["choices"][0]["message"]["content"]

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
    args = ap.parse_args()

    caster = Caster(args.upstream, args.model)
    async with websockets.serve(caster.handle, "127.0.0.1", args.port):
        print(f"caster: duo live on ws://127.0.0.1:{args.port} "
              f"(model {args.model} via {args.upstream})", flush=True)
        await caster.worker()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
