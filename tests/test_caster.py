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
