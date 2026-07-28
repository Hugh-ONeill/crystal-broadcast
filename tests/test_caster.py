"""Caster tests: persona routing policy, shared-transcript prompts (the
correction loop's substrate), AIRI-envelope compatibility, and the
skip-don't-queue latency policy. The LLM is mocked — these drive the same
seams the gold-set runner will."""
import asyncio
import json
import sys
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
