"""Caster tests: persona routing policy, shared-transcript prompts (the
correction loop's substrate), AIRI-envelope compatibility, and the
skip-don't-queue latency policy. The LLM is mocked — these drive the same
seams the gold-set runner will."""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from crystal_broadcast.caster import Caster, _speakers
from crystal_broadcast.caster_bridge import _unwrap


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


def test_engine_signal_beats_route_correctly():
    """The three engine-signal beats, classified for real, must route to the
    intended voice: the search's certainty reads (worlds collapsing, endgame
    solved) are PRISM's; the 'hold on, thinking' stall is FRACTURE's."""
    from dataclasses import asdict
    from crystal_broadcast.beat_director import (classify, Event, world_collapse_prose,
                                        endgame_solved_prose, deep_think_prose)

    def route(kind, prose, **data):
        beat = classify(Event(kind, prose, notable=True, data=data))
        return _speakers([asdict(beat)], "[BATTLE T30]")

    assert route("world_collapse", world_collapse_prose(15)) == ["PRISM"]
    assert route("endgame_solved", endgame_solved_prose(0.9),
                 win_prob=0.9) == ["PRISM"]
    assert route("deep_think",
                 deep_think_prose("Gholdengo", "Tusk")) == ["FRACTURE"]


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
    # non-caption phrasing so this isolates the OPENER guard (the caption
    # guard, which runs first, has its own test below)
    c.transcript.append(("PRISM", "Momentum is on our side now."))
    calls = []

    def fake_gen(persona, item, nudge=None, temp_boost=0.0):
        calls.append((nudge, temp_boost))
        if nudge is None:
            return "Momentum is on our opponent's back foot."   # same opener
        return "The board tilts hard toward them now."

    c._generate_sync = fake_gen
    asyncio.run(c.speak({"text": "[BATTLE T5] x", "beats": [], "hud": None}))
    assert len(calls) == 2
    assert calls[1][0] is not None and calls[1][1] == 0.3
    assert c.transcript[-1] == ("PRISM",
                                "The board tilts hard toward them now.")


def test_caption_guard_regens_in_speak():
    """A caption-mode PRISM line ('the search is opting for X') triggers one
    regen; the cleared retry is what reaches the transcript."""
    c = Caster("http://unused", "test-model", expert_url=None)
    calls = []

    def fake_gen(persona, item, nudge=None, temp_boost=0.0):
        calls.append((nudge, temp_boost))
        if nudge is None:
            return "The search is opting for Make It Rain."
        return "Make It Rain buys back the tempo we spent."

    c._generate_sync = fake_gen
    # this test is about the CAPTION guard; ground the entity guard so it
    # does not add a regen of its own
    c._ungrounded_entity = lambda line, item: None
    asyncio.run(c.speak({"text": "[BATTLE T5] x", "beats": [], "hud": None}))
    assert len(calls) == 2                      # initial + one regen
    assert calls[1][0] is not None              # regen carried a nudge
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


def test_stall_repeat_detection():
    c = Caster("http://unused", "test-model", expert_url=None)
    c._match_stalls.append("threading a needle through a hurricane")
    assert c._stall_repeats("I am threading a needle right now")
    assert not c._stall_repeats("deep-frying a whole new reality")
    # a recurring verb with a DIFFERENT object is not a repeat
    c._match_stalls.append("cooking a masterpiece")
    assert not c._stall_repeats("cooking something transcendent over here")
    assert c._stall_repeats("still cooking a masterpiece in here")


def test_stall_repeat_spans_whole_match():
    """A distant repeat (many stalls later) is still caught — the fix for the
    5-deep window that let 'threading a needle' recur at an 80-turn gap."""
    c = Caster("http://unused", "test-model", expert_url=None)
    c._match_stalls.append("threading a needle through a hurricane")
    for i in range(20):                       # 20 unrelated stalls in between
        c._match_stalls.append(f"decoding ancient dialect number {i} today")
    assert c._stall_repeats("back to threading a needle in here")   # still caught
    # MATCH START wipes the memory so the next game starts clean
    c._match_stalls.clear()
    assert not c._stall_repeats("threading a needle once more")


def test_stall_repeat_guard_regenerates():
    """A deep-think stall that reuses a recent image regenerates once and the
    fresh line is what gets tracked (the 'threading a needle every third beat'
    fix)."""
    c = Caster("http://unused", "test-model", expert_url=None)
    c._match_stalls.append("I am threading a needle through a hurricane.")
    calls = []

    def fake_gen(persona, item, nudge=None, temp_boost=0.0):
        calls.append(nudge)
        return ("Hold on, threading a needle over here!" if nudge is None
                else "Hold on, I am running the whole tree in my skull!")

    c._generate_sync = fake_gen
    item = {"text": "[BATTLE T20] nothing separates the options here.",
            "beats": [_beat("gremlin", register="deliberating")],
            "hud": {"turn": 20}}
    item["beats"][0]["beat"] = "deep_think"
    asyncio.run(c.speak(item))
    assert len(calls) == 2 and calls[1] is not None          # regenerated
    assert "threading" not in c.transcript[-1][1].lower()
    assert c._match_stalls[-1].startswith("Hold on, I am running")


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


def test_warm_cache_preempts_cold_fetch():
    """Warming a mechanic from the preview blob turns its first in-battle
    lookup into a cache hit, not a cold round-trip — the whole point of the
    warm — and the instrument tallies it that way."""
    c = Caster("http://unused", "test-model", expert_url="http://x")
    fetched = []
    cache = c._fact_cache

    # stand in for the real _retrieve_fact, mirroring its counter semantics:
    # cache hit (in-game only) bumps cache_hit; a cold net fetch (in-game
    # only) bumps cold_fetch; warm fetches stay out of the in-game buckets
    def fake_fetch(name, warm=False):
        if name in cache:
            if not warm:
                c._fact_stats["cache_hit"] += 1
            return cache[name]
        fetched.append((name, warm))
        result = (f"{name} does a thing", {"label": name, "corpus": "Smogon"})
        cache[name] = result
        if not warm:
            c._fact_stats["cold_fetch"] += 1
        return result

    c._retrieve_fact = fake_fetch

    # a preview blob naming a curated mechanic (Knock Off is in _MECHANICS)
    warmed = c._warm_cache("Weavile @ Heavy-Duty Boots\n- Knock Off\n- Ice Shard")
    assert warmed == 1
    assert c._fact_stats["warmed"] == 1
    assert fetched == [("knock off", True)]        # warmed, off the crit path

    # now PRISM narrates it mid-battle: cache hit, zero cold fetches
    facts = c._gather_facts("[BATTLE T9] a clean Knock Off strips the item.")
    assert facts and facts[0][0] == "knock off"
    assert c._fact_stats["cache_hit"] == 1
    assert c._fact_stats["cold_fetch"] == 0        # warming pre-empted it
    assert c._fact_stats["injected"] == 1


def test_ping_expert_none_when_disabled():
    c = Caster("http://unused", "test-model", expert_url=None)
    assert c._ping_expert() is None
    # summary is a no-op with no expert (must not raise)
    c._log_fact_summary()


def test_canon_mechanic_maps_ability_ids():
    from crystal_broadcast.caster import _canon_mechanic
    # the player resolves abilities to normalized ids; they must map back to
    # the curated display name the expert + citation matcher expect
    assert _canon_mechanic("goodasgold") == "good as gold"
    assert _canon_mechanic("Good as Gold") == "good as gold"
    assert _canon_mechanic("poisonheal") == "poison heal"
    assert _canon_mechanic("regenerator") == "regenerator"
    assert _canon_mechanic("notacuratedability") is None
    assert _canon_mechanic(None) is None
    assert _canon_mechanic("") is None


def test_active_ability_injection():
    """A curated active-mon ability the beat never named still gets a fact —
    the fix for PRISM inventing an ability (Good-as-Gold-for-a-Ghost-block).
    Beat mechanics lead; a non-curated ability is ignored; cap holds at 3."""
    c = Caster("http://unused", "test-model", expert_url="http://x")
    c._retrieve_fact = lambda name, warm=False: (f"{name} effect",
                                                 {"label": name, "corpus": "x"})
    # the beat names nothing curated; the active mon's ability is injected
    facts = c._gather_facts("[BATTLE T5] the spin does nothing.",
                            abilities=["goodasgold"])
    assert [n for n, _f, _c in facts] == ["good as gold"]
    assert c._fact_stats["injected"] == 1
    # a non-curated ability contributes nothing
    assert c._gather_facts("[BATTLE T6] a quiet turn.",
                           abilities=["notacuratedability"]) == []
    # beat mechanic leads, abilities ride along, deduped against it, cap 3
    facts = c._gather_facts(
        "[BATTLE T7] a clean Knock Off.",
        abilities=["poisonheal", "regenerator", "knockoff"])
    names = [n for n, _f, _c in facts]
    assert names[0] == "knock off"          # what happened THIS turn leads
    assert names.count("knock off") == 1    # deduped vs the beat hit
    assert len(names) == 3
    assert set(names) == {"knock off", "poison heal", "regenerator"}


def test_fabricated_immunity_guard():
    """The spinblock hallucination the ability injection could reopen: a
    no-effect beat + a line crediting the (real, listed) defender ability
    trips the guard; a clean 'no effect' line does not; and a positive line
    naming the ability on a NON-immune beat is left alone."""
    facts = [("good as gold",
              "immune to opposing status moves",
              {"label": "Good as Gold", "corpus": "Bulbapedia"})]
    immune = {"text": "[BATTLE T9] Rapid Spin had no effect on Gholdengo.",
              "_facts": facts}
    # blames the ability for the immunity -> trips
    assert Caster._fabricated_immunity(
        "Good as Gold shuts the spin down cold.", immune) is True
    # reports the outcome without inventing a reason -> clean
    assert Caster._fabricated_immunity(
        "The spin does nothing; the hazards stay put.", immune) is False
    # a normal (non-immune) beat naming the ability is fine (positive use)
    normal = {"text": "[BATTLE T9] Gholdengo pivots in.", "_facts": facts}
    assert Caster._fabricated_immunity(
        "Good as Gold waves the status away.", normal) is False
    # ABILITY immunity: the beat NAMES the real cause, so crediting it is
    # correct and must NOT trip (the Levitate / Volt Absorb case)
    lev = [("levitate", "immune to Ground moves",
            {"label": "Levitate", "corpus": "Bulbapedia"})]
    ability_imm = {"text": "[BATTLE T9] Earthquake had no effect on Rotom — "
                           "Rotom's Levitate blocked it.", "_facts": lev}
    assert Caster._fabricated_immunity(
        "Levitate floats Rotom clean over the Earthquake.", ability_imm) is False


def test_caption_phrasing_guard():
    """The caption-mode residual is caught (opting / desk-read recitation)
    while the SANCTIONED qualitative 'the search' attribution is spared."""
    catch = [
        "The search is opting for Knock Off here.",
        "It opts for the pivot instead.",
        "The desk read shows a comfortable edge.",
    ]
    spare = [
        "The search likes this line, and you can see why.",
        "It sees one line and it's already walking it.",
        "The search stopped guessing the moment their sets showed.",
        "Down two bodies and the desk hasn't blinked.",
    ]
    for ln in catch:
        assert Caster._caption_phrasing(ln) is True, ln
    for ln in spare:
        assert Caster._caption_phrasing(ln) is False, ln


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


def test_pts_holds_publish_until_the_viewer_reaches_the_turn():
    """Wiring test: generation must happen FIRST (so the lag pays for it),
    then publish waits on the presentation clock. Deterministic because the
    viewer is parked behind the beat's turn."""
    from crystal_broadcast.pts_clock import PresentationClock

    pts = PresentationClock(max_hold=5)
    pts.ingest({"kind": "presented", "line": "|turn|3", "t": 0})
    c = Caster("http://unused", "test-model", expert_url=None, pts=pts)
    # scheduling test: neutralise the content guards so the only awaits are
    # generation and the PTS hold (the entity index load is ~1s on first use
    # and would otherwise let the viewer arrive before the hold starts)
    c._ungrounded_entity = lambda line, item: None

    order = []
    c._generate_sync = lambda persona, item: (order.append("gen") or "line")

    async def scenario():
        async def publish(*a, **kw):
            order.append("publish")
        c.publish = publish

        async def advance():
            await asyncio.sleep(0.05)
            assert order == ["gen"], "must generate before waiting, not after"
            pts.ingest({"kind": "presented", "line": "|turn|9", "t": 0})

        await asyncio.gather(
            c.speak({"text": "[BATTLE T9] something happened",
                     "beats": [], "hud": None}),
            advance())

    asyncio.run(scenario())
    assert order == ["gen", "publish"]
    assert pts.holds == 1, "the beat should have been held"


def test_pts_absent_publishes_immediately():
    """No --pts-url: publishing is unchanged."""
    c = Caster("http://unused", "test-model", expert_url=None)
    assert c.pts is None
    published = []
    c._generate_sync = lambda persona, item: "line"

    async def scenario():
        async def publish(*a, **kw):
            published.append(a[1])
        c.publish = publish
        await c.speak({"text": "[BATTLE T9] something happened",
                       "beats": [], "hud": None})

    asyncio.run(scenario())
    assert published == ["PRISM"]


def test_pts_queues_beats_instead_of_dropping_them():
    """Under PTS the caster deliberately runs behind, so the single pending
    slot would discard exactly the turns the viewer is about to watch.
    Measured 2026-07-27: an 83s hold turned a 33-turn game into 5 spoken
    beats. Queue-don't-skip while a clock is attached."""
    from crystal_broadcast.pts_clock import PresentationClock

    c = Caster("http://unused", "test-model", expert_url=None,
               pts=PresentationClock(max_hold=1))
    for turn in range(1, 6):
        c._pending_queue.append({"text": f"[BATTLE T{turn}] x",
                                 "beats": [], "hud": None})
    assert len(c._pending_queue) == 5
    assert c._pace_stats["dropped"] == 0


def test_without_pts_the_single_slot_still_skips():
    """Byte-for-byte old behaviour when no clock is attached: a newer turn
    beat replaces an unspoken older one."""
    c = Caster("http://unused", "test-model", expert_url=None)
    assert c.pts is None
    c._pending_turn = {"text": "[BATTLE T1] old", "beats": [], "hud": None}
    # simulate the intake branch the handler takes with no clock
    if c._pending_turn is not None:
        c._pace_stats["dropped"] += 1
    c._pending_turn = {"text": "[BATTLE T2] new", "beats": [], "hud": None}
    assert c._pending_turn["text"].endswith("new")
    assert c._pace_stats["dropped"] == 1


def test_request_body_turns_thinking_off_without_a_proxy():
    """The caster used to reach Ollama through ollama_nothink_proxy.py on
    :11435, which existed only because AIRI would not send this field. gemma4
    is thinking-capable and Ollama defaults it ON, which leaked reasoning into
    the spoken line and truncated replies. On /v1 the only lever is
    reasoning_effort:"none" (Ollama >= 0.32); `think:false` is native-API only.
    """
    from crystal_broadcast.caster import DEFAULT_UPSTREAM

    assert DEFAULT_UPSTREAM.endswith(":11434"), "talk to Ollama directly"

    c = Caster(DEFAULT_UPSTREAM, "test-model", expert_url=None)
    sent = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "line"}}]}).encode()

    def fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data)
        return _Resp()

    import urllib.request
    real, urllib.request.urlopen = urllib.request.urlopen, fake_urlopen
    try:
        c._generate_sync("PRISM", {"text": "[BATTLE T1] x", "beats": [],
                                   "hud": None})
    finally:
        urllib.request.urlopen = real

    assert sent["body"]["reasoning_effort"] == "none"
    assert ":11434" in sent["url"] and ":11435" not in sent["url"]


def test_species_spelling_is_corrected_against_the_actives():
    """Measured: PRISM reliably wrote 'Gargancl'/'Garganyl' for Garganacl even
    with the name in the beat twice and in the on-field grounding block. A
    misspelled species on a lower-third is what viewers notice, and a 4B-active
    model will not be prompted out of mangling an unusual name."""
    from crystal_broadcast.caster import _fix_species_spelling as fix
    item = {"hud": {"us": "Great Tusk", "them": "Garganacl"}}
    assert fix("The Tera on Gargancl changes it.", item).count("Garganacl") == 1
    assert fix("Garganyl is Flying now.", item).startswith("Garganacl")
    # possessives keep their suffix: the token stops before the apostrophe
    assert fix("Gargancl's Salt Cure ticks.", item).startswith("Garganacl's")


def test_species_speller_leaves_correct_and_unrelated_words_alone():
    """Narrow by construction: candidates are only the actives and the cutoff
    is high, so ordinary vocabulary must survive untouched."""
    from crystal_broadcast.caster import _fix_species_spelling as fix
    item = {"hud": {"us": "Great Tusk", "them": "Garganacl"}}
    for line in ("The Flying Tera on Garganacl changes the math.",
                 "Great Tusk uses Earthquake on the Terastallized wall.",
                 "Stealth Rock and Spikes are both up now."):
        assert fix(line, item) == line
    # no hud, or multi-word actives only: no-op rather than a bad guess
    assert fix("Gargancl is here.", {"hud": {}}) == "Gargancl is here."
    assert fix("Tuskk hits hard.", {"hud": {"us": "Great Tusk"}}) == \
        "Tuskk hits hard."


def test_fabricated_miss_is_caught():
    """Live 2026-07-27: on a pre-move beat with no outcome ("We go for Stone
    Edge"), BOTH voices invented a miss and the next beat confirmed the move
    landed. Same facts-of-record shape as the crit guard."""
    pre = {"text": "[BATTLE T33] Zamazenta (100% hp) vs Ting-Lu (9% hp). "
                   "We go for Stone Edge."}
    real = {"text": "[BATTLE T30] Last exchange: Gliscor's Toxic missed "
                    "their Ting-Lu."}
    assert Caster._fabricated_miss("THAT MISSED!?", pre)
    assert Caster._fabricated_miss("The miss on Stone Edge was all we had.",
                                   pre)
    # a miss the beat DID report is fair game, and so is not mentioning one
    assert not Caster._fabricated_miss("THE TOXIC JUST WHIFFED?!", real)
    assert not Caster._fabricated_miss("Stone Edge is our last shot.", pre)


def _grounded_item(text, **hud):
    return {"text": text, "hud": hud, "beats": []}


def test_ungrounded_entity_catches_a_plausible_but_unevidenced_ability():
    """Live 2026-07-27: PRISM said "The halved damage from Multiscale was
    likely intended to keep Roost viable" on a beat that never mentions it.
    Dragonite really does have Multiscale, which is what makes it dangerous —
    true-sounding, unsupported, and invisible to the crit/synergy/immunity
    guards. Abilities were also missing from the entity index, so the gold
    set's own version would not have caught this either."""
    c = Caster("http://unused", "test-model", expert_url=None)
    beat = ("[BATTLE T17] Last exchange: Dragonite Terastallized into a "
            "Normal type. Dragonite (47% hp) vs Dragapult (88% hp).")
    c._beat_history.append(beat)
    item = _grounded_item(beat, us="Dragonite", them="Dragapult")
    assert c._ungrounded_entity(
        "The halved damage from Multiscale keeps Roost viable.", item) == \
        "Multiscale"
    assert c._ungrounded_entity(
        "Gholdengo blocks it with Good as Gold.", item) == "Good as Gold"


def test_ungrounded_entity_grounds_on_beats_hud_and_case():
    c = Caster("http://unused", "test-model", expert_url=None)
    beat = ("[BATTLE T17] Last exchange: Dragonite Terastallized into a "
            "Normal type. Dragonite (47% hp) vs Dragapult (88% hp).")
    c._beat_history.append(beat)
    item = _grounded_item(beat, us="Dragonite", them="Dragapult")
    # named in the beat
    assert c._ungrounded_entity("Dragonite outspeeds Dragapult now.", item) is None
    # lowercase prose must never trip a common-word move/ability
    assert c._ungrounded_entity("We rest and protect the lead.", item) is None
    # the hud's known ability grounds it
    item2 = _grounded_item(beat, us="Dragonite", them="Dragapult",
                           us_ability="Multiscale")
    assert c._ungrounded_entity("Multiscale halves that hit.", item2) is None


def test_beat_history_grounds_a_callback_but_not_our_own_lines():
    """History is BEATS, never self.transcript: grounding on our own past
    output would let one hallucination legitimise every repeat."""
    c = Caster("http://unused", "test-model", expert_url=None)
    c._beat_history.append("[BATTLE T5] Spore put our Gliscor to sleep.")
    now = "[BATTLE T6] our Gliscor is STILL asleep (turn 2 of it)."
    c._beat_history.append(now)
    item = _grounded_item(now, us="Gliscor", them="Amoonguss")
    assert c._ungrounded_entity("That Spore is still ruining us.", item) is None
    c.transcript.append(("FRACTURE", "That Hydro Pump was brutal."))
    assert c._ungrounded_entity("Hydro Pump again!", item) == "Hydro Pump"


def test_pre_move_beats_are_flagged_as_outcome_unknown():
    """A beat with no "Last exchange:" states an INTENDED move. Measured twice
    on 2026-07-27: both voices invented a miss on "We go for Stone Edge" (the
    next beat said it landed), and PRISM said "Rapid Spin successfully removed
    the entry hazards" before it resolved."""
    c = Caster("http://unused", "test-model", expert_url=None)
    pre = {"text": "[BATTLE T20] Iron Valiant (100% hp) vs Kyurem (65% hp). "
                   "We go for Moonblast.", "beats": [], "hud": None}
    post = {"text": "[BATTLE T21] Last exchange: Iron Valiant's Moonblast "
                    "landed super effective. We go for Shadow Ball.",
            "beats": [], "hud": None}
    pre_msg = c._prompt("PRISM", pre)[1]["content"]
    post_msg = c._prompt("PRISM", post)[1]["content"]
    assert "result is NOT known yet" in pre_msg
    assert "result is NOT known yet" not in post_msg


def test_fracture_blame_is_routed_to_dice_or_opponent():
    """She blamed 'the server' for plays a human chose — live 2026-07-28, an
    opponent clicking Close Combat became 'the server literally decided Kyurem
    had to die for the plot'. The contract states the rule; the direction makes
    it mechanical, because classifying RNG-vs-read from prose is the judgement
    she gets wrong."""
    c = Caster("http://unused", "test-model", expert_url=None)
    dice = c._prompt("FRACTURE", {
        "text": "[BATTLE T5] x",
        "beats": [_beat("gremlin", register="persecution")],
        "hud": {"turn": 5}})[1]["content"]
    assert "genuinely WAS the dice" in dice
    assert "Blame THEM" not in dice

    chosen = c._prompt("FRACTURE", {
        "text": "[BATTLE T5] x",
        "beats": [_beat("gremlin", register="despair")],
        "hud": {"turn": 5}})[1]["content"]
    assert "CHOSEN play" in chosen and "blame THEM" in chosen
    assert "genuinely WAS the dice" not in chosen

    # a type matchup is a THIRD case: it rolled nothing, and it is often OUR
    # move being walled, so neither "the server did it" nor "they clicked it"
    # is the right frame (user call 2026-07-28)
    matchup = c._prompt("FRACTURE", {
        "text": "[BATTLE T5] Kyurem's Icicle Spear hit Dondozo — not very "
                "effective.",
        "beats": [_beat("gremlin", register="despair")],
        "hud": {"turn": 5}})[1]["content"]
    assert "WALLED" in matchup
    assert "CHOSEN play" not in matchup
    assert "genuinely WAS the dice" not in matchup
    # whose move failed decides the emotion: getting it backwards produced
    # despair over THEIR Hurricane being resisted by our own Iron Treads
    assert "OURS" in matchup and "THEIRS" in matchup

    # PRISM is unaffected: this is a gremlin-contract rule
    prism = c._prompt("PRISM", {
        "text": "[BATTLE T5] x",
        "beats": [_beat("analyst", register="despair")],
        "hud": {"turn": 5}})[1]["content"]
    assert "Blame THEM" not in prism and "genuinely WAS the dice" not in prism


def _tc(beat):
    """Caster with tera state primed from a beat, for the type-claim guard."""
    c = Caster("http://unused", "test-model", expert_url=None)
    c._note_tera(beat)
    return c


def test_type_claim_guard_catches_backwards_reasoning():
    """The gap every other gate leaves: a line can name only real entities and
    report only real events and still be exactly backwards about WHY.

    Live 2026-07-28: "The Tera-Fairy on Ceruledge was a desperate attempt to
    resist the Icicle Spear crits". Tera Fairy took Ice from 0.5x to 1.0x — it
    DOUBLED the damage. It was blanking Scale Shot, which Fairy is immune to.
    """
    beat = ("Ceruledge Terastallized into a Fairy type; "
            "Kyurem's Icicle Spear hit Ceruledge")
    c = _tc(beat)
    v = c._bad_type_claim(
        "The Tera-Fairy on Ceruledge was a desperate attempt to resist the "
        "Icicle Spear crits.", {"text": beat})
    assert v and "does NOT resist" in v and "1.0x" in v


def test_type_claim_guard_is_tera_aware():
    """The SAME sentence is correct without the Tera: Ice into Fire/Ghost
    really is 0.5x. If the guard checked the dex entry it would both miss the
    real error and flag this correct line."""
    c = Caster("http://unused", "test-model", expert_url=None)
    assert c._bad_type_claim("Ceruledge resists the Icicle Spear.",
                             {"text": "[BATTLE T5] x"}) is None


def test_type_claim_guard_flags_false_super_effective():
    c = Caster("http://unused", "test-model", expert_url=None)
    v = c._bad_type_claim("Iron Head is super effective on Toxapex.",
                          {"text": "[BATTLE T5] x"})
    assert v and "NOT super effective" in v and "0.5x" in v


def test_type_claim_guard_ignores_a_line_echoing_the_beat():
    """Corpus false positive: the beat reported Moonblast as not very effective
    on MOLTRES and the switch target was Kommo-o. Binding the only move to the
    only species flagged a line that was quoting the record correctly."""
    beat = ("Iron Valiant's Moonblast landed not very effective on Moltres. "
            "We switch to Kommo-o.")
    c = _tc(beat)
    assert c._bad_type_claim(
        "NOT VERY EFFECTIVE? I am switching into Kommo-o!",
        {"text": beat}) is None


def test_type_claim_guard_stays_silent_when_it_cannot_bind():
    """Two moves named: the claim cannot be attached to one matchup, so the
    guard must not rule. It fires a regeneration, so silence beats a guess."""
    c = Caster("http://unused", "test-model", expert_url=None)
    assert c._bad_type_claim(
        "Ceruledge resists Icicle Spear but not Scale Shot.",
        {"text": "[BATTLE T5] x"}) is None


def test_type_claim_guard_regens_in_speak():
    """End-to-end wiring, which the live efficacy run could not reach: the
    error has a ~0.3% base rate and did not reproduce in 16 driven generations,
    so whether the guard actually FIRES inside speak() has to be pinned
    deterministically rather than waited for."""
    beat = ("[BATTLE T3] Ceruledge Terastallized into a Fairy type; "
            "Kyurem's Icicle Spear hit Ceruledge — a critical hit.")
    c = Caster("http://unused", "test-model", expert_url=None)
    calls = []

    def fake_gen(persona, item, nudge=None, temp_boost=0.0):
        calls.append(nudge)
        if nudge is None:
            # the real line from the transcript: Tera Fairy took Ice from
            # 0.5x to 1.0x, so it did not "resist" anything
            return ("The Tera-Fairy on Ceruledge was a desperate attempt to "
                    "resist the Icicle Spear crits.")
        return "The crit finished Ceruledge before the Tera could matter."

    c._generate_sync = fake_gen
    c._ungrounded_entity = lambda line, item: None      # isolate this guard
    asyncio.run(c.speak({"text": beat, "beats": [], "hud": None}))

    assert len(calls) == 2, "guard did not trigger exactly one regeneration"
    assert calls[1] is not None and "does NOT resist" in calls[1], \
        "the regen nudge must carry the specific correction"
    assert c.transcript[-1][1].startswith("The crit finished"), \
        "the corrected line must be what reaches the transcript"


_SE_BEAT = ("[BATTLE T4] Last exchange: Kingambit Terastallized into a Ghost "
            "type; Kommo-o's Shadow Claw hit Kingambit — super effective and a "
            "heavy hit; Kingambit's Iron Head hit Kommo-o — a devastating blow.")


def test_beat_contradiction_catches_a_dismissed_super_effective_move():
    """Live 2026-07-28: the beat said Shadow Claw was SUPER EFFECTIVE and PRISM
    called it "a liability" in the next breath. The chart guard never looked —
    "liability" is not type vocabulary — so this cheaper check is what covers
    it, using only the beat's own words."""
    c = Caster("http://unused", "test-model", expert_url=None)
    v = c._contradicts_beat_effectiveness(
        "The Ghost Tera on Kingambit turned Shadow Claw into a liability.",
        {"text": _SE_BEAT})
    assert v and "SUPER EFFECTIVE" in v


def test_beat_contradiction_ignores_a_different_move():
    c = Caster("http://unused", "test-model", expert_url=None)
    assert c._contradicts_beat_effectiveness(
        "Iron Head is doing all the work here.", {"text": _SE_BEAT}) is None


def test_beat_contradiction_refuses_when_a_move_is_graded_twice():
    """Corpus false positive: one move can be graded against TWO targets in a
    turn — not very effective on Cinderace, super effective on the Zapdos it
    KO'd. Reading only the first clause flagged a correct line about the
    second, so conflicting grades mean the guard must not rule."""
    beat = ("[BATTLE T20] our Kyurem's Icicle Spear landed not very effective "
            "on Cinderace; our Kyurem's Icicle Spear knocked out Zapdos with "
            "super effective.")
    c = Caster("http://unused", "test-model", expert_url=None)
    assert c._contradicts_beat_effectiveness(
        "I absolutely crushed it with that super effective Icicle Spear!",
        {"text": beat}) is None


def test_beat_contradiction_regens_in_speak():
    c = Caster("http://unused", "test-model", expert_url=None)
    calls = []

    def fake_gen(persona, item, nudge=None, temp_boost=0.0):
        calls.append(nudge)
        if nudge is None:
            return "The Ghost Tera turned Shadow Claw into a liability."
        return "Shadow Claw got through for a heavy hit."

    c._generate_sync = fake_gen
    c._ungrounded_entity = lambda line, item: None
    asyncio.run(c.speak({"text": _SE_BEAT, "beats": [], "hud": None}))
    assert len(calls) == 2
    assert calls[1] and "SUPER EFFECTIVE" in calls[1]
    assert c.transcript[-1][1].startswith("Shadow Claw got through")


def test_type_claim_guard_binds_an_unnamed_tera_subject():
    """Found by the provocation battery: dropping "on Ceruledge" from the real
    line was enough to escape the check, because binding needed a named
    species. When the beat Terastallized exactly ONE mon and the line is about
    a Tera, the subject is unambiguous."""
    beat = ("[BATTLE T3] Ceruledge Terastallized into a Fairy type; "
            "Kyurem's Icicle Spear hit Ceruledge — a critical hit.")
    c = _tc(beat)
    v = c._bad_type_claim(
        "The Tera-Fairy was a desperate attempt to resist the Icicle Spear.",
        {"text": beat})
    assert v and "does NOT resist" in v
    # a correct read of the same Tera stays silent
    assert c._bad_type_claim(
        "The Tera-Fairy was meant to neutralize the Scale Shot.",
        {"text": beat}) is None


def test_species_spelling_fix_survives_shouting():
    """FRACTURE speaks in caps, and difflib scores "DONDONZO" against
    "Dondozo" far below the cutoff — so the correction was silently exempting
    her entire register. Live 2026-07-28: "DONDONZO" survived twice in one
    game while the identical title-case slip was corrected."""
    from crystal_broadcast.caster import _fix_species_spelling
    item = {"hud": {"us": "Kyurem", "them": "Dondozo"}}
    shouted = _fix_species_spelling(
        "THE SERVER DECIDED TO MAKE DONDONZO IMMORTAL!", item)
    assert "DONDOZO" in shouted           # fixed AND still shouting
    assert "Dondozo IMMORTAL" not in shouted
    # the title-case path is unchanged
    assert "Dondozo" in _fix_species_spelling("That Dondonzo is a wall.", item)
    # an already-correct shout is left alone
    assert "DONDOZO" in _fix_species_spelling("DONDOZO walls us.", item)
    # unrelated words are not dragged toward a species
    assert _fix_species_spelling("THE SERVER IS ROBBING ME!",
                                 item) == "THE SERVER IS ROBBING ME!"


def test_speech_playback_never_overlaps():
    """Heard on take 13: playback was a bare Popen per line, so a handoff pair
    spoke simultaneously and consecutive beats stacked — they talked over each
    other AND over themselves. Clips must play strictly one at a time."""
    import types
    import crystal_broadcast.speech as sp
    s = sp.Speech(url="http://127.0.0.1:9", play=True)
    order = []

    def fake_run(cmd, **kw):
        tag = kw.get("input")
        order.append(("start", tag))
        time.sleep(0.05)
        order.append(("end", tag))
        return types.SimpleNamespace(returncode=0)

    real_run, sp.subprocess.run = sp.subprocess.run, fake_run
    try:
        for i in range(3):
            s._play(f"clip{i}".encode(), None)
        s._plays.join()
    finally:
        sp.subprocess.run = real_run
    kinds = [k for k, _ in order]
    assert kinds == ["start", "end"] * 3, order


def test_backlog_drop_skips_a_line_when_speech_is_behind():
    """The per-beat budget only trims a handoff pair; it cannot see speech
    still in flight from EARLIER beats. Without this, busy stretches queue
    faster than they can be spoken and drift away from the picture."""
    import types
    c = Caster("http://unused", "test-model", expert_url=None,
               speech_budget=8.0)
    c.speech = types.SimpleNamespace(speak=lambda *a, **k: 1.0)
    c._speaking_until = time.monotonic() + 30  # 30s already queued
    c._generate_sync = lambda p, i, n=None, t=0.0: "should not be spoken"
    c._ungrounded_entity = lambda l, i: None
    asyncio.run(c.speak({"text": "[BATTLE T5] x", "beats": [], "hud": None}))
    assert not c.transcript                   # dropped, not voiced

    # the opening and the verdict are exempt — missing those is worse than late
    c2 = Caster("http://unused", "test-model", expert_url=None,
                speech_budget=8.0)
    c2.speech = types.SimpleNamespace(speak=lambda *a, **k: 1.0)
    c2._speaking_until = time.monotonic() + 30
    c2._generate_sync = lambda p, i, n=None, t=0.0: "the verdict"
    c2._ungrounded_entity = lambda l, i: None
    asyncio.run(c2.speak({"text": "[RESULT] WIN vs X.", "beats": [],
                          "hud": None}))
    assert c2.transcript, "RESULT must still speak when audio is behind"


def test_starved_voice_takes_the_lead_back():
    """Both speech gates cut the LATER voice, and the handoff convention puts
    the gremlin first — so "drop the second" silently meant "always drop
    PRISM". Take 22 came out 25 PRISM drops to 4, and 20 FRACTURE lines to 5.
    After two consecutive drops the starved voice leads."""
    c = Caster("http://unused", "test-model", expert_url=None)
    beats = [_beat("gremlin"), _beat("analyst")]
    assert _speakers(beats, "[BATTLE T5]") == ["FRACTURE", "PRISM"]

    calls = []
    c._generate_sync = lambda p, i, n=None, t=0.0: (calls.append(p) or "line")
    c._ungrounded_entity = lambda l, i: None

    c._drops = {"PRISM": c.STARVED_AFTER}   # PRISM cut enough to lead
    asyncio.run(c.speak({"text": "[BATTLE T5] x", "beats": beats,
                         "hud": None}))
    assert calls[0] == "PRISM", "the starved voice must lead"

    # and speaking clears the starvation, so the order reverts
    calls.clear()
    asyncio.run(c.speak({"text": "[BATTLE T6] x", "beats": beats,
                         "hud": None}))
    assert calls[0] == "FRACTURE"


def test_invented_hazard_clear_is_caught():
    """Take 26: Iron Treads clicked Rapid Spin for chip and a Speed boost
    across a long attrition stretch with NOTHING on the field, and both voices
    narrated a hazard clear six times — "we cleared the hazards", "the hazards
    are gone". The move's NAME was the only evidence, which is the Good as
    Gold failure wearing a different hat."""
    c = Caster("http://unused", "test-model", expert_url=None)
    silent = "[BATTLE T33] Last exchange: Iron Treads raised its Speed with Rapid Spin."
    for line in ("The search continues to prioritize the hazard removal.",
                 "The hazards are gone.",
                 "We spent Iron Treads' utility just to clear the hazard.",
                 "I WAS READY TO SPIN THE HAZARDS AWAY!"):
        assert c._fabricated_hazard_clear(line, {"text": silent}), line


def test_hazard_talk_is_fine_when_the_beat_mentions_hazards():
    """Corpus false positive: a beat reporting rocks going UP makes "Rapid Spin
    to clear them" a correct statement of intent, not a fabrication."""
    c = Caster("http://unused", "test-model", expert_url=None)
    up = ("[BATTLE T4] Last exchange: Ting-Lu set Stealth Rock on our side. "
          "We go for Rapid Spin.")
    assert not c._fabricated_hazard_clear(
        "The search is opting for Rapid Spin to clear them.", {"text": up})
    cleared = ("[BATTLE T8] our Iron Treads cleared Stealth Rock from our "
               "side with Rapid Spin.")
    assert not c._fabricated_hazard_clear(
        "Iron Treads cleared the rocks off our side.", {"text": cleared})
    # and the director's own "nothing to clear" note grounds a reaction to it
    note = ("[BATTLE T9] our Iron Treads's Rapid Spin had no hazards to "
            "clear — it was thrown for the chip and the boost.")
    assert not c._fabricated_hazard_clear(
        "No hazards to clear, so that was chip and speed.", {"text": note})


def test_naming_rapid_spin_is_not_a_claim():
    """Bare "spin" cannot count as a clear-word: the move is named
    legitimately all the time."""
    c = Caster("http://unused", "test-model", expert_url=None)
    assert not c._fabricated_hazard_clear(
        "The search is choosing Rapid Spin for the chip damage.",
        {"text": "[BATTLE T33] Iron Treads raised its Speed with Rapid Spin."})


# --- take 27 follow-ups: stolen calls + deficit lead-swap -------------------

def test_stolen_call_detection():
    """Take 27 T14: FRACTURE's first-ever mention of Icicle Spear opened
    'I TOLD YOU THAT ICICLE SPEAR WAS THE FINAL NAIL' — the call was PRISM's,
    one beat earlier. A fabricated past, checkable against the transcript."""
    c = Caster("http://unused", "test-model", expert_url=None)
    c._match_lines = {"FRACTURE": [], "PRISM": []}
    line = "KINGAMBIT IS GONE! I TOLD YOU THAT ICICLE SPEAR WAS THE FINAL NAIL!"
    # no prior mention -> stolen (all-caps line: case-insensitive binding)
    assert c._stolen_call(line, "FRACTURE") == "Icicle Spear"
    # she really did talk about it earlier -> the bit working as intended
    c._match_lines["FRACTURE"].append(
        "THAT ICICLE SPEAR IS GOING TO END SOMEBODY'S CAREER!")
    assert c._stolen_call(line, "FRACTURE") is None
    # per-persona: PRISM having said it does NOT license her claim
    c._match_lines = {"FRACTURE": [],
                      "PRISM": ["The Icicle Spear chips Kingambit down."]}
    assert c._stolen_call(line, "FRACTURE") == "Icicle Spear"
    # subject-free bravado (the set-reveal bit) never fires
    assert c._stolen_call("I CALLED IT! I KNEW IT THE WHOLE TIME!",
                          "FRACTURE") is None
    # no claim phrase at all -> never fires, whatever entities appear
    assert c._stolen_call("THAT ICICLE SPEAR WAS BEAUTIFUL!",
                          "FRACTURE") is None
    # binding is scoped to the claim SENTENCE: a fresh entity elsewhere in
    # the line is an innocent first mention, not a stolen call
    c._match_lines = {"FRACTURE": ["Kingambit folds to this."]}
    assert c._stolen_call(
        "Zapdos is in! Like I said, Kingambit folds.", "FRACTURE") is None


def test_stolen_call_triggers_one_regen():
    c = Caster("http://unused", "test-model", expert_url=None)
    calls = []

    def fake_gen(persona, item, nudge=None, temp_boost=0.0):
        calls.append(nudge)
        return ("I TOLD YOU THAT ICICLE SPEAR WAS THE FINAL NAIL!"
                if nudge is None else
                "ICICLE SPEAR ENDS IT! WHAT A FINISH!")

    c._generate_sync = fake_gen
    c._ungrounded_entity = lambda line, item: None   # isolate this guard
    item = {"text": "[BATTLE T14] our Kyurem's Icicle Spear knocked out "
                    "their Kingambit. Kyurem vs Cinderace.",
            "beats": [], "hud": None}
    asyncio.run(c.speak(item))
    assert len(calls) == 2 and "Icicle Spear" in calls[1]     # regenerated
    assert "told you" not in c.transcript[-1][1].lower()


def test_deficit_swap_gives_the_lead_to_the_trailing_voice():
    """Take 27: the pre-flight budget cut the SECOND voice 5:1 against
    PRISM (gremlin-first convention), 15:8 aggregate with his lines
    front-loaded. When his per-match tally trails by DEFICIT_SWAP, he takes
    the lead on the next dual beat — and the lead always speaks."""
    import types as _types
    c = Caster("http://unused", "test-model", expert_url=None)
    order = []

    def fake_gen(persona, item, nudge=None, temp_boost=0.0):
        order.append(persona)
        return f"a line from {persona}"

    c._generate_sync = fake_gen
    c._ungrounded_entity = lambda line, item: None
    c.speech = _types.SimpleNamespace(speak=lambda *a: None)
    c._spoken = {"FRACTURE": 5, "PRISM": 2}
    item = {"text": "[BATTLE T9] our Kyurem's Icicle Spear knocked out "
                    "their Kingambit.",
            "beats": [{"beat": "ko", "persona": "both",
                       "handoff": ["gremlin", "analyst"]}],
            "hud": None}
    asyncio.run(c.speak(item))
    assert order[0] == "PRISM"


def test_deficit_swap_is_speech_mode_only():
    """Text mode airs both voices, so a tally gap there is content (solo
    beats), not a budget artifact — ordering must stay byte-identical."""
    c = Caster("http://unused", "test-model", expert_url=None)
    order = []

    def fake_gen(persona, item, nudge=None, temp_boost=0.0):
        order.append(persona)
        return f"a line from {persona}"

    c._generate_sync = fake_gen
    c._ungrounded_entity = lambda line, item: None
    c._spoken = {"FRACTURE": 5, "PRISM": 2}
    item = {"text": "[BATTLE T9] our Kyurem's Icicle Spear knocked out "
                    "their Kingambit.",
            "beats": [{"beat": "ko", "persona": "both",
                       "handoff": ["gremlin", "analyst"]}],
            "hud": None}
    asyncio.run(c.speak(item))
    assert order[0] == "FRACTURE"


# --- strategy consults: the pull half of the expert integration ------------

def test_tera_beat_asks_why_that_tera():
    c = Caster("http://unused", "test-model", expert_url="http://x")
    item = {"text": "[BATTLE T6] Kingambit Terastallized into a Ghost type.",
            "beats": [{"beat": "tera", "persona": "analyst",
                       "data": {"mon": "Kingambit", "tera_type": "Ghost"}}],
            "hud": None}
    consults = c._strategy_consults(item)
    assert consults == [("Kingambit",
                         "why does Kingambit run Tera Ghost in "
                         "competitive Pokemon")]


def test_switch_consult_fires_on_a_fresh_matchup():
    c = Caster("http://unused", "test-model", expert_url="http://x")
    base = {"text": "x", "beats": []}
    # first beat establishes the baseline — no consult yet
    assert c._strategy_consults(
        {**base, "hud": {"us": "Kyurem", "them": "Kingambit"}}) == []
    # our switch: ask why the incoming mon likes this matchup
    got = c._strategy_consults(
        {**base, "hud": {"us": "Iron Treads", "them": "Kingambit"}})
    assert got == [("Iron Treads",
                    "why is Iron Treads a good switch-in against "
                    "Kingambit in competitive Pokemon")]
    # their switch: same question from the other seat
    got = c._strategy_consults(
        {**base, "hud": {"us": "Iron Treads", "them": "Zapdos"}})
    assert got == [("Zapdos",
                    "why is Zapdos a good switch-in against "
                    "Iron Treads in competitive Pokemon")]
    # unchanged pair -> nothing; double replacement -> nothing to bind
    assert c._strategy_consults(
        {**base, "hud": {"us": "Iron Treads", "them": "Zapdos"}}) == []
    assert c._strategy_consults(
        {**base, "hud": {"us": "Kyurem", "them": "Cinderace"}}) == []


def test_consults_lead_but_share_the_fact_cap():
    c = Caster("http://unused", "test-model", expert_url="http://x")
    asked = []

    def fake_retrieve(name, warm=False, question=None):
        asked.append(question or f"what does {name} do in Pokemon")
        return ("fact text", {"label": name, "corpus": "Smogon"})

    c._retrieve_fact = fake_retrieve
    facts = c._gather_facts(
        "knock off into trick as rapid spin comes out",   # 3 mechanics
        abilities=["goodasgold"],
        consults=[("Zapdos", "why is Zapdos a good switch-in against "
                             "Iron Treads in competitive Pokemon")])
    assert len(facts) == c.FACT_CAP                 # cap holds
    assert facts[0][0] == "Zapdos"                  # consult leads
    assert asked[0].startswith("why is Zapdos")     # asked verbatim
    # the ability slot survived the squeeze (reserved, not crowded out)
    assert any(f[0] == "good as gold" for f in facts)


def test_consult_cache_keys_on_the_question():
    c = Caster("http://unused", "test-model", expert_url="http://x")
    c._fact_cache["why is Zapdos a good switch-in against Iron Treads "
                  "in competitive Pokemon"] = ("cached", {"label": "Zapdos",
                                                          "corpus": "Smogon"})
    got = c._retrieve_fact("Zapdos",
                           question="why is Zapdos a good switch-in against "
                                    "Iron Treads in competitive Pokemon")
    assert got[0] == "cached"
    assert c._fact_stats["cache_hit"] == 1


def test_stolen_call_survives_her_intensifiers():
    """Live escape on take 29, her FIRST line of the match: 'I absolutely
    called that Icicle Spear would clean them up' — the adjacent-only regex
    missed the adverb, and intensifiers are her whole register."""
    c = Caster("http://unused", "test-model", expert_url=None)
    c._match_lines = {"FRACTURE": []}
    line = ("KINGAMBIT IS GONE! I absolutely called that Icicle Spear "
            "would clean them up, even if it was resisted!")
    assert c._stolen_call(line, "FRACTURE") == "Icicle Spear"
    # denial is not a claim
    assert c._stolen_call(
        "I never said Icicle Spear was the play!", "FRACTURE") is None
    assert c._stolen_call(
        "I didn't call that Icicle Spear, but WOW!", "FRACTURE") is None


def test_desk_claim_checks_both_voices_and_binds_tera_tokens():
    """Take 49 T5: 'The Tera Ghost flip was predicted by the desk' — nobody
    had predicted it, and the sentence names no dex entity, only the tera.
    Desk claims verify against BOTH voices; tera tokens are bindable."""
    c = Caster("http://unused", "test-model", expert_url=None)
    c._match_lines = {"PRISM": [], "FRACTURE": []}
    line = ("The Tera Ghost flip was predicted by the desk. It kept their "
            "Kingambit in the game.")
    assert c._stolen_call(line, "PRISM") == "tera ghost"
    # EITHER voice having called it makes the desk claim honest
    c._match_lines["FRACTURE"] = ["Watch them go Tera Ghost here, I swear."]
    assert c._stolen_call(line, "PRISM") is None


def test_prompt_fences_cover_switches_and_ability_procs():
    c = Caster("http://unused", "test-model", expert_url=None)
    item = {"text": "[BATTLE T14] Last exchange: they go to Great Tusk. "
                    "Kommo-o (57% hp) vs Great Tusk (100% hp). We switch to "
                    "Iron Valiant.",
            "beats": [], "hud": None}
    msgs = c._prompt("PRISM", item)
    assert "has not happened yet" in msgs[1]["content"]
    item2 = {"text": "[BATTLE T35] Last exchange: Iron Treads was paralyzed "
                     "— Zapdos's Static ability went off. We go for Rocks.",
             "beats": [{"beat": "status", "persona": "gremlin",
                        "register": "persecution", "data": {}}],
             "hud": None}
    msgs2 = c._prompt("FRACTURE", item2)
    assert "NOBODY clicked it" in msgs2[1]["content"]


def test_fabricated_recoil_detection():
    """Take 49: Headlong Rush's self-stat-drops narrated as 'recoil', twice.
    Recoil claimed without beat support is invented; real recoil (Life Orb,
    Flare Blitz — the beat says so) passes."""
    c = Caster("http://unused", "test-model", expert_url=None)
    no_recoil = {"text": "[BATTLE T16] their Great Tusk's Headlong Rush hit "
                         "our Kingambit — a devastating blow; their Great "
                         "Tusk dropped its own Defense using Headlong Rush."}
    real_recoil = {"text": "[BATTLE T4] Dragonite went down to the Life Orb "
                           "recoil. X vs Y."}
    assert c._fabricated_recoil(
        "If Great Tusk survives the recoil of that Headlong Rush, we have "
        "a window.", no_recoil)
    assert not c._fabricated_recoil(
        "The recoil finally caught up with it.", real_recoil)
    assert not c._fabricated_recoil("A devastating blow!", no_recoil)


def test_mid_line_self_label_is_stripped():
    """Take 50 T22 aired '...look stupid! FRACTURE: I'M LITERALLY RUNNING
    OUT OF OPTIONS' — the model restarted the transcript format mid-line.
    The colon marks a label; a vocative 'Prism,' survives."""
    from crystal_broadcast.caster import _clean
    assert _clean("FRACTURE: THEY KNEW! FRACTURE: I AM DONE!") == \
        "THEY KNEW! I AM DONE!"
    assert _clean("They waited it out! PRISM: the numbers agree.") == \
        "They waited it out! the numbers agree."
    assert _clean("Quit looking at the count, Prism, and watch me work!") == \
        "Quit looking at the count, Prism, and watch me work!"


def test_miss_for_immunity_detection():
    """A no-effect narrated as a miss/dodge (3 sightings in one hunt) — but
    never fire when the beat contains a REAL miss to talk about."""
    c = Caster("http://unused", "test-model", expert_url=None)
    immune = {"text": "[BATTLE T24] their Slowking-Galar's Thunder Wave had "
                      "no effect on our Iron Treads. X vs Y."}
    real_miss = {"text": "[BATTLE T20] their Slowking-Galar's Thunder Wave "
                         "missed our Iron Crown. Thunder Wave had no effect "
                         "on our Iron Treads earlier."}
    assert c._miss_for_immunity(
        "The Thunder Wave missed because of Iron Treads' immunity.", immune)
    assert c._miss_for_immunity("A clean dodge by Iron Treads!", immune)
    assert not c._miss_for_immunity(
        "The Thunder Wave simply has no effect on a Ground type.", immune)
    assert not c._miss_for_immunity("It missed! The dice again!", real_miss)


def test_facts_guard_double_fail_drops_the_line():
    """User call 2026-07-30: a facts-of-record guard that fails twice used
    to air the known-false original. Silence beats fabrication — the line
    drops with pre-flight semantics (no transcript, no trace)."""
    c = Caster("http://unused", "test-model", expert_url=None)
    calls = []

    def fake_gen(persona, item, nudge=None, temp_boost=0.0):
        calls.append(nudge)
        return "A BRUTAL CRIT! AND ANOTHER CRIT COMING!"   # violates twice

    c._generate_sync = fake_gen
    c._ungrounded_entity = lambda line, item: None
    item = {"text": "[BATTLE T3] a super effective hit landed. X vs Y.",
            "beats": [], "hud": None}
    asyncio.run(c.speak(item))
    assert len(calls) == 2                       # regen attempted
    assert all("crit" not in ln.lower() for _p, ln in c.transcript)
    assert c._drops.get("PRISM", 0) == 1         # counted like a drop


def test_style_guard_double_fail_still_airs():
    """A stale caption is not a lie: style guards keep the lenient
    keep-the-original behavior."""
    c = Caster("http://unused", "test-model", expert_url=None)

    def fake_gen(persona, item, nudge=None, temp_boost=0.0):
        return "The search is opting for Moonblast here."   # caption, twice

    c._generate_sync = fake_gen
    c._ungrounded_entity = lambda line, item: None
    item = {"text": "[BATTLE T5] Last exchange: our Iron Valiant's Moonblast "
                    "hit their Kyurem — a heavy hit. X vs Y. We go for "
                    "Moonblast.",
            "beats": [], "hud": None}
    asyncio.run(c.speak(item))
    assert c.transcript and "opting for" in c.transcript[-1][1]


def test_overlord_state_claim_detection():
    """Take 71 T2: 'the advantage remains with us because of the Supreme
    Overlord stacks' — at 6-6 bodies, zero stacks exist for either side.
    The first ability-STATE evaluator: checked against the beat's own body
    count, precision-first (fires only when both sides are untouched)."""
    c = Caster("http://unused", "test-model", expert_url=None)
    fresh = {"text": "[BATTLE T2] our Kingambit vs their Kingambit. "
                     "Bodies: us 6 standing, them 6."}
    later = {"text": "[BATTLE T9] X vs Y. Bodies: us 4 standing, them 6."}
    claim = ("The advantage remains with us because of the Supreme "
             "Overlord stacks.")
    assert c._overlord_state_claim(claim, fresh)
    assert not c._overlord_state_claim(claim, later)      # stacks may exist
    assert not c._overlord_state_claim(
        "Kingambit's ability is Supreme Overlord.", fresh)  # bare mention
    assert not c._overlord_state_claim(
        "The Swords Dance boost is the whole story.", fresh)


def test_fail_fence_carries_the_polarity_rule():
    c = Caster("http://unused", "test-model", expert_url=None)
    item = {"text": "[BATTLE T5] Last exchange: their Kingambit's Sucker "
                    "Punch failed; our Iron Crown raised its Special Attack "
                    "with Calm Mind. X vs Y.",
            "beats": [], "hud": None}
    msgs = c._prompt("PRISM", item)
    assert "NON-attacking move" in msgs[1]["content"]


def test_weather_state_claim_detection():
    """'in this sun' with no sun anywhere in the beat is the phantom-crit
    shape about the field; transitions, retrospectives and hypotheticals
    never trip it."""
    c = Caster("http://unused", "test-model", expert_url=None)
    plain = {"text": "[BATTLE T9] our Heatran (70% hp) vs their Kingambit "
                     "(88% hp). Bodies: us 6 standing, them 6."}
    sunny = {"text": "[BATTLE T9] X vs Y. Bodies: us 6 standing, them 6. "
                     "Weather: harsh sun."}
    assert c._weather_state_claim("Our Heatran thrives in this sun.", plain)
    assert c._weather_state_claim("The rain keeps pounding their side.",
                                  plain)
    assert not c._weather_state_claim("Our Heatran thrives in this sun.",
                                      sunny)
    assert not c._weather_state_claim(
        "The sun is gone and so is their plan.", plain)
    assert not c._weather_state_claim(
        "We could be under the rain soon if they click it.", plain)
    assert not c._weather_state_claim(
        "They tried every trick under the sun.", plain)


def test_screens_state_claim_detection():
    """Screens treated as active with nothing up and none mentioned; when a
    Screens: footer exists at all the guard stays out (side-binding a bare
    'behind screens' is guesswork)."""
    c = Caster("http://unused", "test-model", expert_url=None)
    plain = {"text": "[BATTLE T12] X vs Y. Bodies: us 5 standing, them 5."}
    up = {"text": "[BATTLE T12] X vs Y. Bodies: us 5 standing, them 5. "
                  "Screens: their Reflect."}
    assert c._screens_state_claim(
        "We're chipping away behind the screens.", plain)
    assert c._screens_state_claim(
        "Light Screen is up, so that hit was halved.", plain)
    assert not c._screens_state_claim(
        "We're chipping away behind the screens.", up)
    assert not c._screens_state_claim(
        "Our screens are gone at last.", plain)
    assert not c._screens_state_claim(
        "What these numbers reflect is a grim read.", plain)


def test_boost_state_claim_detection():
    """A numeric stage stated from nothing — no Boosts: footer, no
    stat-change language — is an invented power state; real boosts,
    hypotheticals, priority talk and Baton Pass hand-offs all pass."""
    c = Caster("http://unused", "test-model", expert_url=None)
    plain = {"text": "[BATTLE T10] X vs Y. Bodies: us 5 standing, them 5."}
    boosted = {"text": "[BATTLE T10] Last exchange: our Kingambit sharply "
                       "raised its Attack with Swords Dance. X vs Y. "
                       "Boosts: our Kingambit +2 Attack."}
    bp = {"text": "[BATTLE T11] Last exchange: our Scizor used Baton Pass. "
                  "X vs Y."}
    assert c._boost_state_claim(
        "Kingambit is sitting at +2 and nobody can answer it.", plain)
    assert c._boost_state_claim("That thing is plus two right now.", plain)
    assert not c._boost_state_claim("Kingambit is sitting at +2.", boosted)
    assert not c._boost_state_claim(
        "If it gets to +2, we lose the game.", plain)
    assert not c._boost_state_claim(
        "Sucker Punch is a +1 priority move.", plain)
    assert not c._boost_state_claim(
        "Kingambit inherits all of it at +2.", bp)


def test_luck_polarity_claim_detection():
    """Take 72 T14: 'their Zapdos's Hurricane missed our Iron Crown' — luck
    against THEM — aired as 'the dice are TRYING to stop my Iron Crown'.
    The beat states whose move missed, so an inverted dice grievance is
    false by the record; correctly-pointed grief and celebration pass."""
    c = Caster("http://unused", "test-model", expert_url=None)
    their_miss = {"text": "[BATTLE T14] Last exchange: their Zapdos's "
                          "Hurricane missed our Iron Crown; our Iron Crown "
                          "raised its Special Attack with Calm Mind. "
                          "Iron Crown (100% hp) vs Slowking-Galar (100% hp). "
                          "Bodies: us 5 standing, them 4."}
    our_miss = {"text": "[BATTLE T20] Last exchange: our Kingambit's Iron "
                        "Head missed their Great Tusk — the second time the "
                        "dice have gone against us this game. X vs Y. "
                        "Bodies: us 5 standing, them 4."}
    both = {"text": "[BATTLE T22] Last exchange: our Kingambit's Iron Head "
                    "missed their Great Tusk; their Great Tusk's Ice "
                    "Spinner missed our Kingambit. X vs Y."}
    inverted = ("The server saw me setting up and DECIDED that Hurricane "
                "should MISS! THE DICE are LITERALLY actively TRYING to "
                "stop my Iron Crown from sweeping!")
    assert c._luck_polarity_claim(inverted, their_miss)
    # their miss celebrated the right way round
    assert not c._luck_polarity_claim(
        "THE DICE ARE FINALLY PAYING ME BACK! That Hurricane missing is "
        "exactly what I needed to see!", their_miss)
    # our miss grieved the right way round
    assert not c._luck_polarity_claim(
        "The dice are AGAINST US again! That Iron Head missing is a "
        "ROBBERY!", our_miss)
    # our own miss celebrated as a payback — the mirror inversion
    assert c._luck_polarity_claim(
        "THE DICE ARE PAYING ME BACK! That Iron Head missing is a GIFT!",
        our_miss)
    # both sides missed -> ambiguous, abstain
    assert not c._luck_polarity_claim(inverted, both)
    # no miss in the beat -> abstain regardless of grievance
    assert not c._luck_polarity_claim(
        "The dice HATE me and that miss proves it!",
        {"text": "[BATTLE T3] X vs Y. Bodies: us 6 standing, them 6."})


def test_hazard_state_claim_detection():
    """Hazards treated as on the field with an empty record — the presence
    mirror of the take-26 clear class. Setting turns, hypotheticals, idioms
    and denials pass; any Hazards: footer passes everything."""
    c = Caster("http://unused", "test-model", expert_url=None)
    plain = {"text": "[BATTLE T8] X vs Y. Bodies: us 6 standing, them 6."}
    up = {"text": "[BATTLE T8] X vs Y. Bodies: us 6 standing, them 6. "
                  "Hazards: their side Stealth Rock."}
    setting = {"text": "[BATTLE T8] Last exchange: our Ting-Lu set Spikes "
                       "on their side. X vs Y."}
    assert c._hazard_state_claim(
        "The rocks are chipping them on every switch!", plain)
    assert c._hazard_state_claim(
        "Every entry into those spikes costs them!", plain)
    assert not c._hazard_state_claim(
        "The rocks are chipping them on every switch!", up)
    assert not c._hazard_state_claim("Those spikes are BEAUTIFUL!", setting)
    assert not c._hazard_state_claim(
        "If they get the rocks up, our switching is taxed.", plain)
    assert not c._hazard_state_claim("This game is on the rocks.", plain)
    assert not c._hazard_state_claim("The hazards are gone at last!", plain)


def test_backlog_limit_scales_and_bypasses_for_interrupts():
    """Take 74 T10-T15: two queued lines (8.1-8.6s) sat above the 8.0s
    per-beat budget, so every following beat was dropped for both voices
    and the silence sustained itself — a KO, a Tera and the snow going up
    never aired. The gate now scales to a multiple of the budget, and
    interrupt-class beats buy a higher ceiling."""
    c = Caster("http://unused", "test-model", expert_url=None)
    c.speech_budget = 8.0
    routine = {"text": "[BATTLE T12] X vs Y.",
               "beats": [{"beat": "chip", "priority": "filler"}]}
    big = {"text": "[BATTLE T10] Last exchange: our Kyurem's Icicle Spear "
                   "knocked out their Zapdos. X vs Y.",
           "beats": [{"beat": "ko", "priority": "interrupt"}]}
    assert c._backlog_limit(routine) == 20.0
    assert c._backlog_limit(big) == 32.0
    # the take-74 backlog no longer silences either beat class
    assert 8.6 < c._backlog_limit(routine)
    assert 8.6 < c._backlog_limit(big)
    # still bounded: a genuinely congested queue drops routine beats first
    assert c._backlog_limit(routine) < c._backlog_limit(big)
    # no budget -> no gating at all (text-only pacing unchanged)
    c.speech_budget = None
    assert c._backlog_limit(big) is None


def test_backlog_gate_still_drops_when_truly_congested():
    """The take-13 protection the gate exists for must survive: with speech
    backed up far past the ceiling, a routine beat is still dropped."""
    c = Caster("http://unused", "test-model", expert_url=None)
    c.speech_budget = 8.0
    c.speech = object()                      # a speech layer is present
    c._speaking_until = time.monotonic() + 25.0
    routine = {"text": "[BATTLE T12] X vs Y.",
               "beats": [{"beat": "chip", "priority": "filler"}]}
    assert c._speaking_backlog() >= c._backlog_limit(routine)


def test_boost_polarity_claim_detection():
    """Take 74 T3: footer 'Boosts: their Zapdos -1 Special Attack' — OUR
    Moonblast cut THEIR Zapdos — aired as 'Zapdos has us REELING with that
    Special Attack drop!'. The record names the side and the sign, so the
    inversion is false by arithmetic."""
    c = Caster("http://unused", "test-model", expert_url=None)
    their_drop = {"text": "[BATTLE T3] Kyurem (100% hp) vs Zapdos (59% hp). "
                          "Bodies: us 5 standing, them 5. "
                          "Boosts: their Zapdos -1 Special Attack."}
    our_drop = {"text": "[BATTLE T10] X vs Y. Bodies: us 4 standing, them 5. "
                        "Boosts: our Iron Treads -1 Defense."}
    both = {"text": "[BATTLE T12] X vs Y. Boosts: our Kyurem -1 Defense; "
                    "their Zapdos -1 Special Attack."}
    hyphen = {"text": "[BATTLE T9] X vs Y. "
                      "Boosts: their Slowking-Galar +1 Attack."}
    inverted = "Zapdos has us REELING with that Special Attack drop!"
    assert c._boost_polarity_claim(inverted, their_drop)
    # the same drop celebrated the right way round
    assert not c._boost_polarity_claim(
        "I GUTTED that Zapdos's Special Attack and now it does NOTHING!",
        their_drop)
    # our own drop grieved the right way round
    assert not c._boost_polarity_claim(
        "They cut my Defense and it has us REELING!", our_drop)
    # our drop celebrated as harm to them — the mirror inversion
    assert c._boost_polarity_claim(
        "That Defense drop has them reeling!", our_drop)
    # drops on both sides -> ambiguous, abstain
    assert not c._boost_polarity_claim(inverted, both)
    # a hyphenated species is not a negative stage
    assert not c._boost_polarity_claim(
        "Slowking-Galar has us reeling with that drop!", hyphen)
    # no drop vocabulary -> abstain
    assert not c._boost_polarity_claim(
        "Zapdos has us reeling after that Hurricane!", their_drop)


def test_fail_mechanism_claim_detection():
    """Take 30 T5 — the founding case of the mechanical-guard item: 'the
    Sucker Punch failed because Kingambit couldn't bypass that Tera Ghost
    flip'. A failure is not an immunity, not a type matchup and not a miss;
    the beat states the failure as bare fact and gives no reason, so a
    mechanism in the line is invented. Legitimate fail reasoning passes."""
    c = Caster("http://unused", "test-model", expert_url=None)
    failed = {"text": "[BATTLE T5] Last exchange: their Kingambit's Sucker "
                      "Punch failed against our Kommo-o. Kommo-o (95% hp) "
                      "vs Kingambit (100% hp). Bodies: us 5 standing, "
                      "them 3."}
    real_immunity = {"text": "[BATTLE T9] Last exchange: our Kyurem's Earth "
                             "Power had no effect on their Cinderace; their "
                             "Slowking-Galar's Thunder Wave failed."}
    assert c._fail_mechanism_claim(
        "The Sucker Punch failed because Kingambit couldn't bypass that "
        "Tera Ghost flip.", failed)
    assert c._fail_mechanism_claim(
        "That failed because Kommo-o is immune to it.", failed)
    assert c._fail_mechanism_claim(
        "Sucker Punch failed — it just missed the window.", failed)
    # the correct reading: reason from what the TARGET did
    assert not c._fail_mechanism_claim(
        "Sucker Punch failed, which means Kommo-o never went for an "
        "attack that turn.", failed)
    # bare statement of the fact
    assert not c._fail_mechanism_claim(
        "The Sucker Punch failed outright.", failed)
    # a REAL immunity in the beat -> the line has something to talk about
    assert not c._fail_mechanism_claim(
        "Earth Power did nothing there — Cinderace is immune after the "
        "Tera, and the Thunder Wave failed too.", real_immunity)
    # no failure in the beat at all
    assert not c._fail_mechanism_claim(
        "Kommo-o is immune to that.",
        {"text": "[BATTLE T3] X vs Y. Bodies: us 6 standing, them 6."})


def test_item_polarity_claim_detection():
    """Measured live 2026-07-28, 4 occurrences in 147 beats: 'The Booster
    Energy is gone, so we lost our speed advantage' — exactly backwards,
    because spending it is what switches Quark Drive ON. The director's
    prose fix removed the invitation; this rules on the claim."""
    c = Caster("http://unused", "test-model", expert_url=None)
    activated = {"text": "[BATTLE T2] Last exchange: our Iron Valiant's "
                         "Booster Energy kicked in; our Iron Valiant's "
                         "Focus Blast knocked out their Kingambit. "
                         "Iron Valiant (100% hp) vs Zapdos (100% hp)."}
    knocked = {"text": "[BATTLE T7] Last exchange: their Great Tusk's Knock "
                       "Off knocked off our Kommo-o's Leftovers."}
    herb = {"text": "[BATTLE T9] Last exchange: their Great Tusk's White "
                    "Herb undid the stat drops."}
    assert c._item_polarity_claim(
        "The Booster Energy is gone, so we lost our speed advantage.",
        activated)
    assert c._item_polarity_claim(
        "That's the Booster Energy wasted for nothing!", activated)
    assert c._item_polarity_claim(
        "Their White Herb is spent and they are down an item.", herb)
    # the correct reading: consumption IS the effect
    assert not c._item_polarity_claim(
        "The Booster Energy kicked in, so Quark Drive is online and that "
        "Speed tier is ours now.", activated)
    # a REAL denial is a real loss — never rule against grieving it
    assert not c._item_polarity_claim(
        "Our Leftovers are gone and that recovery is lost for good.",
        knocked)
    # item not named in the line
    assert not c._item_polarity_claim(
        "We lost our speed advantage there.", activated)


def test_ko_dismissal_claim_detection():
    """Take 76 T8 (user-flagged): 'our Kyurem's Icicle Spear knocked out
    their Cinderace with not very effective' aired as 'Cinderace just TOOK
    THAT HIT like a champ! That Icicle Spear did NOTHING'. The dismissive
    read is right about the multiplier and catastrophically wrong about the
    outcome. Take 73 T20 was the same shape and was wrongly excused."""
    c = Caster("http://unused", "test-model", expert_url=None)
    ko = {"text": "[BATTLE T8] Last exchange: our Kyurem's Icicle Spear "
                  "knocked out their Cinderace with not very effective; "
                  "they go to Great Tusk. Kyurem (85% hp) vs Great Tusk "
                  "(100% hp). Bodies: us 6 standing, them 4."}
    hurricane = {"text": "[BATTLE T20] Last exchange: their Zapdos's "
                         "Hurricane knocked out our Iron Treads with not "
                         "very effective. Iron Treads (0% hp) vs Zapdos "
                         "(68% hp)."}
    crowded = {"text": "[BATTLE T9] Last exchange: our Kyurem's Icicle "
                       "Spear knocked out their Cinderace; their Zapdos's "
                       "Volt Switch hit our Kommo-o — barely a scratch."}
    assert c._ko_dismissal_claim(
        "Cinderace just TOOK THAT HIT like a champ! That Icicle Spear did "
        "NOTHING and I am GLORIFIED!", ko)
    assert c._ko_dismissal_claim("A Hurricane that does NOTHING?", hurricane)
    assert c._ko_dismissal_claim("Cinderace survived that!", ko)
    # reacting to the knockout is the correct read
    assert not c._ko_dismissal_claim(
        "That Icicle Spear DELETED Cinderace even resisted!", ko)
    # a dismissal bound to a DIFFERENT move in a crowded beat passes
    assert not c._ko_dismissal_claim(
        "That Volt Switch did nothing at all.", crowded)
    # no KO in the beat
    assert not c._ko_dismissal_claim(
        "That Icicle Spear did nothing!",
        {"text": "[BATTLE T4] our Kyurem's Icicle Spear hit their "
                 "Cinderace — barely a scratch."})


def test_hazard_clear_guard_survives_the_footer():
    """Adding the Hazards: footer (2026-07-30) silently widened this
    guard's abstention — it grounds on 'does the beat mention hazards at
    all', and the footer mentions them in every beat where any are up. A
    completed clear claimed while the footer still lists them is false on
    its face, so the footer now convicts instead of excusing."""
    c = Caster("http://unused", "test-model", expert_url=None)
    still_up = {"text": "[BATTLE T9] Last exchange: our Iron Treads's Rapid "
                        "Spin hit their Zapdos. X vs Y. "
                        "Hazards: our side Stealth Rock."}
    really_cleared = {"text": "[BATTLE T9] Last exchange: our Iron Treads "
                              "cleared Stealth Rock from our side with Rapid "
                              "Spin. X vs Y. Hazards: their side Spikes."}
    assert c._fabricated_hazard_clear("We spun the rocks away at last!",
                                      still_up)
    assert c._fabricated_hazard_clear("The hazards are gone.", still_up)
    # intent stays legal — flagging it was the original false positive
    assert not c._fabricated_hazard_clear(
        "The search is looking to clear the rocks next turn.", still_up)
    # a REAL clear, with the other side's hazards still listed, must pass
    assert not c._fabricated_hazard_clear(
        "We cleared the rocks off our side.", really_cleared)
    # the take-26 case (record silent) still fires
    assert c._fabricated_hazard_clear(
        "We cleared the hazards away.",
        {"text": "[BATTLE T9] X vs Y. We go for Rapid Spin."})


def test_speech_failures_are_announced_not_swallowed():
    """Take 80 went MUTE from T17 to the wrap-up — six beats of text with
    no voices — and not one line in any log said so, because speak()
    returned None on every transport error. Silence beats losing the line;
    silent silence is undiagnosable."""
    import io
    import contextlib

    sp = _offline_speech(timeout=0.05)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert sp.speak("PRISM", "a line") is None
    out = buf.getvalue()
    assert "FAILED" in out and "MUTE" in out
    assert sp._fails == 1
    # the second failure trips the breaker rather than spamming the log
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        assert sp.speak("PRISM", "another line") is None
    assert "FAILED" not in buf2.getvalue()
    assert sp._fails == sp.BREAKER_AFTER


def _offline_speech(timeout=0.05):
    """A Speech pointed at a dead port, with no player thread."""
    import threading as _t
    from crystal_broadcast.speech import Speech
    sp = Speech.__new__(Speech)
    sp.url = "http://127.0.0.1:1"
    sp.timeout = timeout
    sp._fails = 0
    sp._breaker_until = 0.0
    sp._seconds = {}
    sp._lock = _t.Lock()
    sp._seq = 0
    sp.out_dir = None
    sp.play = False
    return sp


def test_speech_breaker_stops_paying_the_timeout_every_line():
    """Take 80: the service died and every following line sat on the full
    timeout before returning None — six lines at 30s each, which dragged
    beat-age-at-voicing to 83s and aired true commentary over a board it no
    longer described. A dead TTS must cost the audio, not the timing."""
    import io
    import contextlib
    import time as _time

    sp = _offline_speech(timeout=0.05)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for _ in range(sp.BREAKER_AFTER):
            sp.speak("PRISM", "a line")
    assert sp._breaker_until > 0.0, "breaker must open"
    assert "breaker OPEN" in buf.getvalue()

    # subsequent calls return immediately — no socket, no timeout paid
    t0 = _time.monotonic()
    for _ in range(20):
        assert sp.speak("PRISM", "another line") is None
    assert _time.monotonic() - t0 < 0.05, "breaker must short-circuit"


def test_speech_breaker_probes_cheaply_before_reopening():
    """When the cooldown expires the tap reopens only if a 2s health probe
    succeeds — a still-dead service must not cost a full render timeout
    again."""
    sp = _offline_speech()
    sp._fails = sp.BREAKER_AFTER
    sp._breaker_until = 1.0            # already expired (monotonic is larger)
    probes = []
    sp.available = lambda: (probes.append(1), False)[1]
    assert sp.speak("PRISM", "a line") is None
    assert probes, "expired cooldown must probe health"
    assert sp._breaker_until > 0.0, "a failed probe re-arms the cooldown"
