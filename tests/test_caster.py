"""Caster tests: persona routing policy, shared-transcript prompts (the
correction loop's substrate), AIRI-envelope compatibility, and the
skip-don't-queue latency policy. The LLM is mocked — these drive the same
seams the gold-set runner will."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.caster import Caster, _speakers
from showdown.airi_bridge import _unwrap


def _beat(persona, priority="interrupt", handoff=None, register=None):
    return {"beat": "x", "persona": persona, "priority": priority,
            "handoff": handoff, "register": register, "prose": "", "data": {}}


def test_speaker_policy():
    assert _speakers([], "[MATCH START] New battle") == ["PRISM", "FRACTURE"]
    assert _speakers([], "[RESULT] WIN vs X") == ["FRACTURE", "PRISM"]
    assert _speakers([], "[BATTLE T4] quiet turn") == ["PRISM"]
    assert _speakers([_beat("gremlin")], "[BATTLE T5]") == ["FRACTURE"]
    assert _speakers([_beat("analyst")], "[BATTLE T5]") == ["PRISM"]
    assert _speakers([_beat("both", handoff=["gremlin", "analyst"])],
                     "[BATTLE T5]") == ["FRACTURE", "PRISM"]
    assert _speakers([_beat("either", priority="interrupt")],
                     "[BATTLE T5]") == ["FRACTURE"]
    assert _speakers([_beat("either", priority="normal")],
                     "[BATTLE T5]") == ["PRISM"]
    assert _speakers([_beat("none")], "[BATTLE T5]") == []
    # coinciding interrupts owned by different personas: both speak,
    # fast reaction leads regardless of beat order
    assert _speakers([_beat("gremlin"), _beat("analyst")],
                     "[BATTLE T5]") == ["FRACTURE", "PRISM"]
    assert _speakers([_beat("analyst"), _beat("gremlin")],
                     "[BATTLE T5]") == ["FRACTURE", "PRISM"]
    # a normal-priority second beat does not add a voice
    assert _speakers([_beat("gremlin"), _beat("analyst", priority="normal")],
                     "[BATTLE T5]") == ["FRACTURE"]


def test_correction_loop_transcript_sharing():
    """On a dual beat PRISM's prompt must contain FRACTURE's line — the
    correction loop is real only if the second speaker sees the first."""
    c = Caster("http://unused", "test-model", expert_url=None)
    calls = []

    def fake_gen(persona, item):
        calls.append((persona, c._prompt(persona, item)))
        return {"FRACTURE": "THAT WAS MY SWITCH. All me.",
                "PRISM": "It was the search's switch."}[persona]

    c._generate_sync = fake_gen
    item = {"text": "[BATTLE T7] Court Change swapped the hazards.",
            "beats": [_beat("both", handoff=["gremlin", "analyst"],
                            register="heist")],
            "hud": None}
    asyncio.run(c.speak(item))
    assert [p for p, _ in calls] == ["FRACTURE", "PRISM"]
    prism_user = calls[1][1][1]["content"]
    assert "FRACTURE: THAT WAS MY SWITCH. All me." in prism_user
    assert "Register: heist" in prism_user
    assert [p for p, _ in c.transcript] == ["FRACTURE", "PRISM"]


def test_match_start_resets_transcript():
    c = Caster("http://unused", "test-model", expert_url=None)
    c.transcript.append(("PRISM", "leftover from last game"))
    c._generate_sync = lambda persona, item: "fresh line"
    asyncio.run(c.speak({"text": "[MATCH START] New battle vs X.",
                         "beats": [], "hud": None}))
    assert all(ln != "leftover from last game" for _, ln in c.transcript)


def test_envelope_parses_like_airi():
    """The published envelope must round-trip through the overlay's
    subscription parsing (superjson unwrap + field extraction)."""
    c = Caster("http://unused", "test-model", expert_url=None)

    class FakeWS:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

    ws = FakeWS()
    c.clients.add(ws)
    asyncio.run(c.publish("[BATTLE T9] beat text", "FRACTURE",
                          "A CRIT. Rigged.", {"turn": 9, "value": 0.44}))
    msg = _unwrap(ws.sent[0])
    assert msg["type"] == "output:gen-ai:chat:complete"
    data = msg["data"]
    assert data["text"].startswith("[")          # overlay's beat gate
    assert data["message"]["content"] == "A CRIT. Rigged."
    assert data["persona"] == "FRACTURE"
    assert data["hud"]["value"] == 0.44


def test_sanitizer_guards_output():
    c = Caster("http://unused", "test-model", expert_url=None)
    c._generate_sync = lambda persona, item: "<thought>plan things</thought>"
    sent = []

    async def fake_publish(*a):
        sent.append(a)

    c.publish = fake_publish
    asyncio.run(c.speak({"text": "[BATTLE T3] x", "beats": [], "hud": None}))
    assert sent == []          # scaffolding never reaches the feed


def test_opener_guard_retries_once_with_nudge():
    c = Caster("http://unused", "test-model", expert_url=None)
    c.transcript.append(("PRISM", "The search is opting for Earthquake."))
    calls = []

    def fake_gen(persona, item, nudge=None, temp_boost=0.0):
        calls.append((nudge, temp_boost))
        if nudge is None:
            return "The search is opting for Make It Rain."   # same opener
        return "Make It Rain buys back the tempo we spent."

    c._generate_sync = fake_gen
    asyncio.run(c.speak({"text": "[BATTLE T5] x", "beats": [], "hud": None}))
    assert len(calls) == 2
    assert calls[1][0] is not None and calls[1][1] == 0.3
    assert c.transcript[-1] == ("PRISM",
                                "Make It Rain buys back the tempo we spent.")


def test_opener_guard_ignores_different_openers():
    c = Caster("http://unused", "test-model", expert_url=None)
    c.transcript.append(("PRISM", "The search is opting for Earthquake."))
    calls = []

    def fake_gen(persona, item, nudge=None, temp_boost=0.0):
        calls.append(nudge)
        return "Tempo is the whole story of this turn."

    c._generate_sync = fake_gen
    asyncio.run(c.speak({"text": "[BATTLE T6] x", "beats": [], "hud": None}))
    assert calls == [None]


def test_prism_angle_rotates_by_turn():
    c = Caster("http://unused", "test-model", expert_url=None)
    prompts = [c._prompt("PRISM", {"text": "[BATTLE T%d] x" % t,
                                   "beats": [], "hud": {"turn": t}})
               for t in (1, 2, 3)]
    angles = [p[1]["content"].split("Angle: ")[1].split(".")[0]
              for p in prompts]
    assert len(set(angles)) == 3
    # register beats take precedence over the angle rotation
    reg = c._prompt("FRACTURE", {"text": "[BATTLE T4] x",
                                 "beats": [_beat("gremlin",
                                                 register="despair")],
                                 "hud": {"turn": 4}})
    assert "Register: despair" in reg[1]["content"]
    assert "Angle:" not in reg[1]["content"]


def test_fabricated_crit_detection():
    c = Caster("http://unused", "test-model", expert_url=None)
    se_beat = {"text": "[BATTLE T14] Bitter Blade knocked out Gholdengo "
                       "with super effective. X vs Y."}
    crit_beat = {"text": "[BATTLE T5] Iron Head landed a critical hit."}
    # crit claimed, beat has no 'critical' -> fabrication
    assert c._fabricated_crit("A SUPER EFFECTIVE CRIT deleted it!", se_beat)
    # crit claimed, beat reports a critical -> fine
    assert not c._fabricated_crit("A CRIT! Rigged!", crit_beat)
    # no crit claim -> fine
    assert not c._fabricated_crit("That was super effective!", se_beat)


def test_fabricated_crit_triggers_one_regen():
    c = Caster("http://unused", "test-model", expert_url=None)
    calls = []

    def fake_gen(persona, item, nudge=None, temp_boost=0.0):
        calls.append(nudge)
        return ("It was a brutal CRIT!" if nudge is None
                else "It was a brutal super-effective hit!")

    c._generate_sync = fake_gen
    item = {"text": "[BATTLE T3] a super effective hit landed. X vs Y.",
            "beats": [], "hud": None}
    asyncio.run(c.speak(item))
    assert len(calls) == 2 and calls[1] is not None          # regenerated
    assert "crit" not in c.transcript[-1][1].lower()


def test_fabricated_synergy_detection():
    c = Caster("http://unused", "test-model", expert_url=None)
    # beat about a poison with NO ability flag -> naming Poison Heal is invented
    plain = {"text": "[BATTLE T57] badly poisoned Dragonite. X vs Pecharunt."}
    assert c._fabricated_synergy(
        "The poison is a boon for Pecharunt via Poison Heal.", plain)
    # beat carries the real synergy tail -> naming it is fine
    real = {"text": "[BATTLE T28] badly poisoned Gliscor — that just feeds "
                    "Gliscor's Poison Heal."}
    assert not c._fabricated_synergy(
        "That Toxic heals Gliscor through Poison Heal.", real)
    # no synergy claim at all -> fine
    assert not c._fabricated_synergy("Pecharunt is poisoned and chipping.",
                                     plain)


def test_fabricated_synergy_triggers_one_regen():
    c = Caster("http://unused", "test-model", expert_url=None)
    calls = []

    def fake_gen(persona, item, nudge=None, temp_boost=0.0):
        calls.append(nudge)
        return ("The poison feeds its Guts, honestly." if nudge is None
                else "The poison just chips it down each turn.")

    c._generate_sync = fake_gen
    item = {"text": "[BATTLE T9] badly poisoned Pecharunt. X vs Y.",
            "beats": [], "hud": None}
    asyncio.run(c.speak(item))
    assert len(calls) == 2 and calls[1] is not None
    assert "guts" not in c.transcript[-1][1].lower()


def test_prism_fact_injection_prism_only():
    c = Caster("http://unused", "test-model", expert_url="http://x")
    c._retrieve_fact = lambda name: (
        ("unaffected by other Pokemon's status moves",
         {"label": "Good as Gold", "corpus": "Bulbapedia"})
        if name == "good as gold" else None)
    facts = c._gather_facts("[BATTLE T27] Gholdengo's Good as Gold is up.")
    assert facts == [("good as gold",
                      "unaffected by other Pokemon's status moves",
                      {"label": "Good as Gold", "corpus": "Bulbapedia"})]
    item = {"text": "[BATTLE T27] Gholdengo's Good as Gold.",
            "beats": [], "hud": {"turn": 27}, "_facts": facts}
    prism = c._prompt("PRISM", item)[1]["content"]
    assert "GROUNDED FACTS" in prism and "status moves" in prism
    # FRACTURE never receives the facts (no-citations contract)
    frac = c._prompt("FRACTURE", item)[1]["content"]
    assert "GROUNDED FACTS" not in frac


def test_fact_injection_off_and_no_hit():
    # expert disabled -> no facts, no lookups
    c = Caster("http://unused", "test-model", expert_url=None)
    assert c._gather_facts("Good as Gold and Drain Punch everywhere") == []
    # expert on but no listed mechanic in the beat -> no lookup attempted
    c2 = Caster("http://unused", "test-model", expert_url="http://x")
    called = []
    c2._retrieve_fact = lambda n: called.append(n)
    assert c2._gather_facts("[BATTLE T2] a routine Earthquake exchange.") == []
    assert called == []


def test_skip_dont_queue():
    """A newer turn beat replaces an unspoken older one; framing beats
    (MATCH START / RESULT) all survive."""

    async def scenario():
        c = Caster("http://unused", "test-model", expert_url=None)
        spoken = []

        async def fake_speak(item):
            spoken.append(item["text"])

        c.speak = fake_speak

        class FakeWS:
            def __init__(self, frames):
                self.frames = frames

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.frames:
                    raise StopAsyncIteration
                return self.frames.pop(0)

            async def send(self, _):
                pass

        frames = [json.dumps({"type": "input:text",
                              "data": {"text": t, "beats": [], "hud": None}})
                  for t in ("[MATCH START] game on",
                            "[BATTLE T2] first",
                            "[BATTLE T5] second overwrites first",
                            "[RESULT] WIN vs X")]
        await c.handle(FakeWS(frames))
        worker = asyncio.get_event_loop().create_task(c.worker())
        await asyncio.sleep(0.05)
        worker.cancel()
        return spoken

    spoken = asyncio.run(scenario())
    assert spoken == ["[MATCH START] game on", "[RESULT] WIN vs X",
                      "[BATTLE T5] second overwrites first"]
    assert "[BATTLE T2] first" not in spoken


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        fn()
        print(f"ok {name}")
    print(f"\n{len(fns)} tests passed")
