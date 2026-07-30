"""Beat director tests — driven exactly the way the gold-set eval runner
will drive it: fabricated protocol batches through ProtocolScanner, ctx
through Director.decide, assertions on the emitted Beats. No live battle,
no AIRI, no wall clock."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from crystal_broadcast.beat_director import (Director, ProtocolScanner, TurnContext,
                                    classify, Event, world_collapse_prose,
                                    endgame_solved_prose, deep_think_prose,
                                    archetype_prose)


def _ctx(turn=5, value=0.5, elapsed=30.0, **kw):
    defaults = dict(me_name="Gliscor", me_hp=80, opp_name="Kingambit",
                    opp_hp=90)
    defaults.update(kw)
    return TurnContext(turn=turn, value=value, elapsed=elapsed, **defaults)


def _stats(name):
    return {"Hatterene": (90, 136), "Kingambit": (135, 60)}.get(name)


# --- scanner: protocol -> typed events ------------------------------------

def test_scanner_ko_attribution_and_side():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "move", "p2a: Kingambit", "Sucker Punch", "p1a: Darkrai"],
        ["", "-damage", "p1a: Darkrai", "0 fnt"],
        ["", "faint", "p1a: Darkrai"],
    ], role="p1")
    kos = [e for e in evs if e.type == "ko"]
    assert len(kos) == 1
    assert kos[0].side == "us"            # OUR mon went down
    assert "knocked out our Darkrai" in kos[0].prose
    assert kos[0].data["move"] == "Sucker Punch"


def test_scanner_court_change_and_hazard_clear():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "-sideend", "p1: wiz", "Spikes", "[from] move: Rapid Spin"],
        ["", "-swapsideconditions"],
    ], role="p1")
    types = [e.type for e in evs]
    assert "hazard_cleared" in types and "hazard_flip" in types
    cleared = next(e for e in evs if e.type == "hazard_cleared")
    # the possessive belongs to the SIDE, not the condition: "our Spikes" reads
    # as the Spikes we set, which is how both personas came to invert a clear
    assert cleared.side == "us" and "from our side" in cleared.prose


def test_scanner_hazard_clear_names_the_spinner():
    """A clear must say WHO did it. Passive prose ('our Stealth Rock was
    cleared away by Rapid Spin') let the casters read our own successful spin
    as the opponent robbing us, and claim we had set hazards the opponent set."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Iron Treads", "Iron Treads, L80", "100/100"],
        ["", "move", "p1a: Iron Treads", "Rapid Spin", "p2a: Gliscor"],
        ["", "-sideend", "p1: wiz", "Stealth Rock", "[from] move: Rapid Spin"],
    ], role="p1")
    cleared = next(e for e in evs if e.type == "hazard_cleared")
    assert "Iron Treads" in cleared.prose          # the actor is named
    assert "from our side" in cleared.prose        # location, not ownership
    assert "our Stealth Rock" not in cleared.prose
    assert cleared.data["user"] and "Iron Treads" in cleared.data["user"]


def test_scanner_yawn_cause_captured():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "-status", "p1a: Gliscor", "slp", "[from] move: Yawn"],
    ], role="p1")
    assert evs[0].type == "status_applied"
    assert evs[0].data["cause"] == "Yawn"


def test_mirror_ko_disambiguates_ownership():
    """In a species mirror the KO prose must say WHOSE fell — bare
    'Kingambit knocked out Kingambit' let the caster flip ownership
    (measured live: FRACTURE called our Kingambit's death a self-KO)."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Kingambit", "Kingambit, M", "100/100"],
        ["", "switch", "p2a: Kingambit", "Kingambit, M", "100/100"],
        ["", "move", "p1a: Kingambit", "Low Kick", "p2a: Kingambit"],
        ["", "-supereffective", "p2a: Kingambit"],
        ["", "-damage", "p2a: Kingambit", "0 fnt"],
        ["", "faint", "p2a: Kingambit"],
    ], role="p2")                      # WE are p2
    ko = next(e for e in evs if e.type == "ko")
    assert "their Kingambit's Low Kick knocked out our Kingambit" in ko.prose
    assert ko.side == "us"             # our mon fell
    # data stays bare species for machine use
    assert ko.data["mover"] == "Kingambit" and ko.data["target"] == "Kingambit"


def test_mirror_match_ko_when_opponent_active_differs():
    """The mirror is a MATCH property, not just an active one: our Clefable
    dies to their Toxapex while both teams carry a Clefable, and the KO prose
    must still say OUR Clefable (measured live: FRACTURE called our fainted
    Clefable 'their Cleric' with their Toxapex in). Roster comes from the
    |poke| preview (their Clefable never even switched in here)."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "poke", "p1", "Clefable, M", "item"],     # their team HAS Clefable
        ["", "poke", "p1", "Toxapex, F", "item"],
        ["", "poke", "p2", "Clefable, M", "item"],     # our team HAS Clefable
        ["", "switch", "p1a: Toxapex", "Toxapex, F", "100/100"],
        ["", "switch", "p2a: Clefable", "Clefable, M", "50/394"],
        ["", "move", "p1a: Toxapex", "Poison Jab", "p2a: Clefable"],
        ["", "-damage", "p2a: Clefable", "0 fnt"],
        ["", "faint", "p2a: Clefable"],
    ], role="p2")
    ko = next(e for e in evs if e.type == "ko")
    assert "knocked out our Clefable" in ko.prose      # OURS fell
    assert "their Toxapex" in ko.prose                 # sides are always marked now


def test_mirror_residual_faint_disambiguates():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Gliscor", "Gliscor, M", "100/100"],
        ["", "switch", "p2a: Gliscor", "Gliscor, M", "100/100"],
        ["", "faint", "p2a: Gliscor"],           # residual, no move this turn
    ], role="p2")
    ko = next(e for e in evs if e.type == "ko")
    assert ko.prose == "our Gliscor went down"


def test_move_prose_marks_the_side_even_without_a_mirror():
    """Superseded 2026-07-29: this used to pin non-mirror prose as byte-
    identical, so a move was only side-marked when the same species sat on
    both rosters. Take 22 showed what that costs — every misattribution in
    the match was an unmarked move, and the beats FRACTURE got right were
    exactly the ones a mirror happened to mark ("Iron Valiant's Focus Blast
    knocked out Kingambit" -> "THEY CLICKED FOCUS BLAST", ours). Whose move
    it is is the single most invertible fact in a beat, so it is now always
    stated."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Great Tusk", "Great Tusk", "100/100"],
        ["", "switch", "p2a: Gholdengo", "Gholdengo", "100/100"],
        ["", "move", "p1a: Great Tusk", "Earthquake", "p2a: Gholdengo"],
        ["", "-damage", "p2a: Gholdengo", "0 fnt"],
        ["", "faint", "p2a: Gholdengo"],
    ], role="p2")
    ko = next(e for e in evs if e.type == "ko")
    assert ko.prose == ("their Great Tusk's Earthquake knocked out "
                        "our Gholdengo")


def test_sleep_talk_move_labeled():
    """A move called by Sleep Talk is labeled, so 'Crunch' on an asleep mon
    reads as the Sleep Talk call it is (user-caught: looked like a stray
    direct move)."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Gliscor", "Gliscor", "100/100"],
        ["", "switch", "p2a: Dondozo", "Dondozo", "100/100"],
        ["", "move", "p2a: Dondozo", "Sleep Talk", "p2a: Dondozo"],
        ["", "move", "p2a: Dondozo", "Crunch", "p1a: Gliscor",
         "[from] move: Sleep Talk"],
        ["", "-damage", "p1a: Gliscor", "40/100"],
    ], role="p2")
    hit = next(e for e in evs if e.type == "move_hit")
    assert "Dondozo's Crunch (via Sleep Talk)" in hit.prose
    # the Sleep Talk vehicle itself (no damage) emits no separate beat
    assert not any("Sleep Talk" in e.prose and "via" not in e.prose
                   for e in evs)


def test_mirror_volatile_qualified():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Dondozo", "Dondozo", "100/100"],
        ["", "switch", "p2a: Dondozo", "Dondozo", "100/100"],
        ["", "-start", "p2a: Dondozo", "Substitute"],
    ], role="p2")
    v = next(e for e in evs if e.type == "volatile_start")
    assert "our Dondozo put up a Substitute" in v.prose


def test_mirror_matchup_line_qualified():
    """The 'X vs Y' matchup line in the composed beat must tag our/their in a
    mirror — an unqualified 'Corviknight (5%) vs Corviknight (81%)' flipped
    HP ownership (measured live: 'their Corviknight crippled' when it was
    ours at 5%)."""
    d = Director()
    dec = d.decide(_ctx(turn=104, value=0.5, elapsed=30.0,
                        me_name="Corviknight", me_hp=5,
                        opp_name="Corviknight", opp_hp=81))
    assert "our Corviknight (5% hp) vs their Corviknight (81% hp)" in dec.text
    # non-mirror is byte-unchanged
    dec2 = d.decide(_ctx(turn=105, value=0.5, elapsed=30.0,
                         me_name="Gliscor", me_hp=50,
                         opp_name="Kingambit", opp_hp=90))
    assert "Gliscor (50% hp) vs Kingambit (90% hp)" in dec2.text
    assert "our Gliscor" not in dec2.text


def test_mirror_status_prose_qualified():
    """A status in a species mirror names whose mon took it."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Gliscor", "Gliscor, M", "100/100"],
        ["", "switch", "p2a: Gliscor", "Gliscor, M", "100/100"],
        ["", "-status", "p2a: Gliscor", "tox", "[from] item: Toxic Orb"],
    ], role="p2")
    st = next(e for e in evs if e.type == "status_applied")
    assert "our Gliscor" in st.prose            # OUR Gliscor was poisoned
    assert st.data["mon"] == "Gliscor"          # data stays bare species


def test_rest_sleep_suppresses_escalating_grievance():
    """A deliberate Rest sleep must not trigger the gremlin's 'STILL asleep'
    grievance; an enemy-inflicted sleep still does (user-caught: FRACTURE
    grieving over a Rested mon turn after turn)."""
    d = Director()
    d.observe([Event("status_applied", "our Dondozo fell asleep", side="us",
                     notable=True,
                     data={"mon": "Dondozo", "status": "slp",
                           "cause": "Rest"})])
    d.decide(_ctx(turn=5, me_name="Dondozo", me_status="slp", elapsed=30.0))
    d.observe([Event("ko", "X went down", side="them", notable=True)])
    d2 = d.decide(_ctx(turn=6, me_name="Dondozo", me_status="slp",
                       elapsed=30.0))
    assert "STILL asleep" not in (d2.text or "")
    # contrast: an enemy Spore sleep DOES escalate
    e = Director()
    e.observe([Event("status_applied", "put Snorlax to sleep", side="us",
                     notable=True,
                     data={"mon": "Snorlax", "status": "slp",
                           "cause": "Spore"})])
    e.decide(_ctx(turn=5, me_name="Snorlax", me_status="slp", elapsed=30.0))
    e.observe([Event("ko", "Y went down", side="them", notable=True)])
    e2 = e.decide(_ctx(turn=6, me_name="Snorlax", me_status="slp",
                       elapsed=30.0))
    assert "Snorlax is STILL asleep (turn 2 of it)" in e2.text


# --- classification: events -> beats with persona/register ----------------

def test_burn_allegiance_registers():
    ours = Event("status_applied", "burned Kingambit", side="us",
                 data={"mon": "Kingambit", "status": "brn"})
    theirs = Event("status_applied", "burned Kingambit", side="them",
                   data={"mon": "Kingambit", "status": "brn"})
    b_ours = classify(ours, _stats)
    b_theirs = classify(theirs, _stats)
    assert b_ours.persona == "gremlin" and b_ours.register == "despair"
    assert b_theirs.register == "celebration"


def test_burn_on_special_attacker_is_analyst_critique():
    ev = Event("status_applied", "burned Hatterene", side="them",
               data={"mon": "Hatterene", "status": "brn"})
    b = classify(ev, _stats)
    assert b.persona == "analyst" and b.register == "wasted-burn"
    assert b.priority == "normal"


def test_yawn_sleep_is_negotiated_not_shock():
    ev = Event("status_applied", "put Gliscor to sleep", side="us",
               data={"mon": "Gliscor", "status": "slp", "cause": "Yawn"})
    b = classify(ev, _stats)
    assert b.persona == "analyst" and b.register == "negotiated"
    direct = Event("status_applied", "put Gliscor to sleep", side="us",
                   data={"mon": "Gliscor", "status": "slp", "cause": None})
    assert classify(direct, _stats).persona == "gremlin"


def test_court_change_dual_beat_handoff():
    ev = Event("hazard_flip", "Court Change swapped the hazards", notable=True)
    b = classify(ev, _stats)
    assert b.persona == "both" and b.handoff == ["gremlin", "analyst"]
    assert b.priority == "interrupt"


def test_hazard_clear_grievance_only_for_our_stack():
    ours = Event("hazard_cleared", "our Spikes was cleared away by Rapid Spin",
                 side="us", data={"condition": "Spikes", "by": "Rapid Spin"})
    theirs = Event("hazard_cleared", "their Spikes was cleared away by Defog",
                   side="them", data={"condition": "Spikes", "by": "Defog"})
    assert classify(ours, _stats).register == "sunk-cost-outrage"
    assert classify(ours, _stats).persona == "gremlin"
    assert classify(theirs, _stats).register == "housekeeping"
    assert classify(theirs, _stats).persona == "analyst"


def test_crit_allegiance():
    against = Event("move_hit", "X landed a critical hit", side="us",
                    data={"crit": True})
    ours = Event("move_hit", "X landed a critical hit", side="them",
                 data={"crit": True})
    assert classify(against, _stats).register == "persecution"
    assert classify(ours, _stats).register == "delight"


# --- director: gating, text, state ----------------------------------------

def test_silence_on_quiet_turn():
    d = Director(min_interval=20.0, min_swing=0.10)
    d.decide(_ctx(turn=1, value=0.5, elapsed=30.0))  # establishes prev
    dec = d.decide(_ctx(turn=2, value=0.51, elapsed=6.0))
    assert dec.silence and dec.text is None


def test_floor_blocks_even_notable():
    d = Director()
    d.observe([Event("ko", "Kingambit went down", side="them", notable=True)])
    dec = d.decide(_ctx(elapsed=2.0))
    assert dec.silence


def test_beat_text_format_stable():
    d = Director()
    d.observe([Event("ko", "Iron Valiant's Shadow Ball knocked out Gholdengo "
                     "with super effective", side="us", notable=True)])
    dec = d.decide(_ctx(turn=6, value=0.62, elapsed=30.0,
                        me_name="Darkrai", me_hp=100,
                        opp_name="Iron Valiant", opp_hp=59,
                        ours_fainted=frozenset({"Gholdengo"}),
                        choice_text="We go for Dark Pulse."))
    assert dec.text.startswith("[BATTLE T6] Last exchange: ")
    assert "Darkrai (100% hp) vs Iron Valiant (59% hp)." in dec.text
    assert "We go for Dark Pulse." in dec.text
    assert "Desk read: we hold a real edge." in dec.text
    assert "Bodies: us 5 standing, them 6." in dec.text
    # KO already narrated in the exchange -> no flat "We lost" duplicate
    assert "We lost" not in dec.text
    ko_beats = [b for b in dec.beats if b.beat == "ko"]
    assert len(ko_beats) == 1 and ko_beats[0].persona == "gremlin"


def test_unnarrated_faint_gets_flat_mention():
    d = Director()
    dec = d.decide(_ctx(turn=9, value=0.4, elapsed=30.0,
                        theirs_fainted=frozenset({"Clefable"})))
    assert "Their Clefable is down." in dec.text


def test_desk_read_spoken_only_on_band_change():
    d = Director()
    d1 = d.decide(_ctx(turn=1, value=0.62, elapsed=30.0))
    assert "Desk read:" in d1.text
    # same band, tiny drift, notable event forces the beat through the gate
    d.observe([Event("tera", "Kingambit Terastallized into a Dark type",
                     side="them", notable=True)])
    d2 = d.decide(_ctx(turn=2, value=0.63, elapsed=30.0))
    assert d2.text is not None and "Desk read:" not in d2.text
    # band change speaks again
    d.observe([Event("ko", "X went down", side="them", notable=True)])
    d3 = d.decide(_ctx(turn=3, value=0.75, elapsed=30.0))
    assert "Desk read: we're clearly ahead" in d3.text


def test_contradiction_once_per_onset_and_beat():
    d = Director()
    base = dict(elapsed=30.0,
                ours_fainted=frozenset({"A", "B", "C"}))
    d1 = d.decide(_ctx(turn=10, value=0.65, **base))
    assert "sharply disagree" in d1.text
    assert any(b.beat == "desk_contradiction" for b in d1.beats)
    d.observe([Event("ko", "Y went down", side="them", notable=True)])
    d2 = d.decide(_ctx(turn=11, value=0.66, **base,
                       theirs_fainted=frozenset()))
    assert d2.text is not None and "sharply disagree" not in d2.text


def test_swing_measured_against_last_sent():
    d = Director(min_swing=0.10)
    d.decide(_ctx(turn=1, value=0.60, elapsed=30.0))
    # gated decisions must not move the reference point
    for turn, v in ((2, 0.57), (3, 0.54)):
        dec = d.decide(_ctx(turn=turn, value=v, elapsed=6.0))
        assert dec.silence
    # slow bleed has now crossed the threshold vs the last SENT value
    dec = d.decide(_ctx(turn=4, value=0.49, elapsed=6.1))
    assert not dec.silence
    assert any(b.beat == "desk_swing" for b in dec.beats)


def test_affliction_escalation_counter():
    d = Director()
    d.observe([Event("status_applied", "put Gliscor to sleep", side="us",
                     notable=True,
                     data={"mon": "Gliscor", "status": "slp"})])
    d1 = d.decide(_ctx(turn=5, me_name="Gliscor", me_status="slp",
                       elapsed=30.0))
    assert "STILL" not in (d1.text or "")
    d.observe([Event("ko", "X went down", side="them", notable=True)])
    d2 = d.decide(_ctx(turn=6, me_name="Gliscor", me_status="slp",
                       elapsed=30.0))
    assert "our Gliscor is STILL asleep (turn 2 of it)" in d2.text
    # cured -> counter resets; no callback
    d.observe([Event("ko", "Y went down", side="them", notable=True)])
    d3 = d.decide(_ctx(turn=7, me_name="Gliscor", me_status=None,
                       elapsed=30.0))
    assert "STILL" not in (d3.text or "")


def test_crowded_turn_keeps_high_priority_prose():
    """A Tera (interrupt) on a 5+-event turn must survive the 4-line
    exchange window — the blind last-4 dropped it (replay-pinning catch)."""
    d = Director()
    d.observe([
        Event("tera", "Great Tusk Terastallized into a Steel type",
              side="them", notable=True, data={"tera_type": "Steel"}),
        Event("move_hit", "Kyurem's Freeze-Dry landed not very effective",
              side="us", data={}),
        Event("move_hit", "Great Tusk's Rapid Spin landed a critical hit",
              side="us", notable=True, data={"crit": True}),
        Event("volatile_end", "Kyurem's Substitute broke", side="us",
              notable=True, data={}),
        Event("boost", "Great Tusk raised its Speed", side="them",
              notable=True, data={}),
    ])
    dec = d.decide(_ctx(turn=10, value=0.5, elapsed=30.0))
    assert "Terastallized into a Steel type" in dec.text
    # chronological order preserved among the kept lines
    assert dec.text.index("Terastallized") < dec.text.index("critical hit")


def test_status_synergy_is_a_boon_not_grief():
    # our Poison Heal Gliscor getting Toxic'd is the plan working, not a
    # wound — analyst boon, never gremlin despair (user-caught live)
    def af(name, side):
        return {"gliscor": {"poisonheal"}}.get(name.lower(), set())
    ev = Event("status_applied", "badly poisoned Gliscor", side="us",
               data={"mon": "Gliscor", "status": "tox"})
    b = classify(ev, None, af)
    assert b.persona == "analyst" and b.register == "status-boon"
    assert "Poison Heal" in b.prose


def test_status_synergy_burn_on_guts():
    def af(name, side):
        return {"ursaluna": {"guts"}}.get(name.lower(), set())
    # burn on THEIR Guts attacker backfires — we helped them
    ev = Event("status_applied", "burned Ursaluna", side="them",
               data={"mon": "Ursaluna", "status": "brn"})
    b = classify(ev, None, af)
    assert b.register == "status-backfire" and "Guts" in b.prose


def test_status_synergy_hedges_when_ability_uncertain():
    def af(name, side):
        # a species that CAN but might not run the synergy ability
        return {"breloom": {"poisonheal", "technician", "effectspore"}}.get(
            name.lower(), set())
    ev = Event("status_applied", "badly poisoned Breloom", side="us",
               data={"mon": "Breloom", "status": "tox"})
    b = classify(ev, None, af)
    assert b.register == "status-boon-hedge"


def test_status_without_synergy_stays_despair():
    def af(name, side):
        return {"corviknight": {"pressure"}}.get(name.lower(), set())
    ev = Event("status_applied", "burned Corviknight", side="us",
               data={"mon": "Corviknight", "status": "brn"})
    assert classify(ev, None, af).register == "despair"
    # and with no ability_fn at all, behaviour is unchanged
    assert classify(ev, None, None).register == "despair"


# --- engine-signal beats (search telemetry, injected like belief_delta) ---

def test_world_collapse_is_analyst_meta_beat():
    ev = Event("world_collapse", world_collapse_prose(15), notable=True)
    b = classify(ev, _stats)
    assert b.beat == "world_collapse" and b.persona == "analyst"
    assert b.priority == "normal" and b.register == "worlds-collapsed"
    assert "15" in b.prose and "sets" in b.prose


def test_endgame_solved_routes_through_endgame_beat():
    ev = Event("endgame_solved", endgame_solved_prose(0.9), notable=True,
               data={"win_prob": 0.9})
    b = classify(ev, _stats)
    # a solver takeover is the analyst's flagship "provably over" interrupt
    assert b.beat == "endgame" and b.persona == "analyst"
    assert b.priority == "interrupt" and b.register == "solved"
    assert "solver" in b.prose and b.data["win_prob"] == 0.9


def test_deep_think_is_gremlin_interrupt():
    ev = Event("deep_think", deep_think_prose("Gholdengo", "Great Tusk"),
               notable=True)
    b = classify(ev, _stats)
    assert b.beat == "deep_think" and b.persona == "gremlin"
    assert b.priority == "interrupt" and b.register == "deliberating"
    # names the active matchup so the reacting voice has a grounded subject
    assert "Gholdengo" in b.prose and "Great Tusk" in b.prose


def test_endgame_solved_prose_states_verdict_honestly():
    assert "winning" in endgame_solved_prose(0.95)
    assert "lost" in endgame_solved_prose(0.05)
    assert "razor-thin" in endgame_solved_prose(0.5)


def test_interject_composes_out_of_band_beat():
    d = Director()
    text, beat = d.interject("deep_think", 33,
                             deep_think_prose("Kingambit", "Dragonite"))
    # keeps the '[BATTLE Tn]' feed shape the overlay parser expects
    assert text.startswith("[BATTLE T33] ")
    assert beat.beat == "deep_think" and beat.persona == "gremlin"
    assert "Kingambit" in text
    # an unknown kind classifies to nothing -> None, never a crash
    assert d.interject("not_a_beat", 5, "x") is None


def test_engine_beat_rides_quiet_turn_decision():
    """world_collapse / endgame_solved are observed (folded) into the next
    decision like any notable event — they must force a beat through the
    gate on an otherwise-quiet turn and land in the composed recap text."""
    d = Director()
    d.decide(_ctx(turn=27, value=0.6, elapsed=30.0))  # establish prev
    d.observe([Event("endgame_solved", endgame_solved_prose(0.9),
                     notable=True, data={"win_prob": 0.9})])
    dec = d.decide(_ctx(turn=28, value=0.61, elapsed=6.0))  # sub-interval
    assert not dec.silence
    assert "solver" in dec.text
    assert any(b.beat == "endgame" for b in dec.beats)


def test_match_framing_texts():
    d = Director()
    start = d.match_start("FPAiri", ["Gliscor", "Darkrai"],
                          ["Kingambit"], lead="Gliscor")
    assert start.startswith("[MATCH START] New battle vs FPAiri.")
    assert "We lead Gliscor." in start
    text, beat = d.match_end("WIN", 3, 0, "FPAiri")
    assert text.startswith("[RESULT] WIN vs FPAiri.")
    assert beat.beat == "recap" and beat.handoff == ["gremlin", "analyst"]


def test_archetype_prose_known_and_unknown():
    """Extensible registry: a known label yields a call-out; an unknown or
    absent label stays silent (never mis-frames a matchup)."""
    frame = archetype_prose("stall")
    assert frame and "stall mirror" in frame
    assert archetype_prose("sun") is None      # not in the registry yet
    assert archetype_prose(None) is None
    assert archetype_prose("") is None


def test_match_start_folds_archetype_and_is_byte_stable_without():
    """The archetype call-out rides the MATCH START framing when detected;
    with no archetype the text is byte-identical to before (no regression)."""
    d = Director()
    base = d.match_start("Foe", ["Gliscor", "Clodsire"], ["Dondozo"],
                         lead="Gliscor")
    withA = d.match_start("Foe", ["Gliscor", "Clodsire"], ["Dondozo"],
                          lead="Gliscor", archetype="stall")
    assert "stall mirror" in withA
    assert base == d.match_start("Foe", ["Gliscor", "Clodsire"], ["Dondozo"],
                                 lead="Gliscor", archetype=None)
    # an unknown archetype must not alter the baseline either
    assert base == d.match_start("Foe", ["Gliscor", "Clodsire"], ["Dondozo"],
                                 lead="Gliscor", archetype="rain")


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        fn()
        print(f"ok {name}")
    print(f"\n{len(fns)} tests passed")


def test_ko_victim_survives_a_same_batch_switch_in():
    """Regression (live 2026-07-27): flush() is DEFERRED to the next move, so
    a faint followed by the replacement switching into the same slot used to
    repoint the position token before the prose was built. The beat went out
    as 'Garganacl's Ice Punch knocked out our Gholdengo' when it was Darkrai
    that died and Gholdengo was the mon that replaced it."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Darkrai", "Darkrai", "4/100"],
        ["", "move", "p2a: Garganacl", "Ice Punch", "p1a: Darkrai"],
        ["", "-damage", "p1a: Darkrai", "0 fnt"],
        ["", "faint", "p1a: Darkrai"],
        # the replacement takes the SAME slot before anything flushes
        ["", "switch", "p1a: Gholdengo", "Gholdengo, tera:Steel", "69/100"],
    ], role="p1")
    kos = [e for e in evs if e.type == "ko"]
    assert len(kos) == 1
    assert "Darkrai" in kos[0].prose
    assert "Gholdengo" not in kos[0].prose
    assert kos[0].data["target"] == "Darkrai"


def test_trick_is_one_beat_and_names_its_user():
    """A swap emits two -item lines. Two beats meant two responses from the
    duo for a single play, and the passive wording ('X was handed a Choice
    Scarf by Trick') let FRACTURE narrate OUR play as the opponent's."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "move", "p1a: Gholdengo", "Trick", "p2a: Garganacl"],
        ["", "-activate", "p1a: Gholdengo", "move: Trick",
         "[of] p2a: Garganacl"],
        ["", "-item", "p2a: Garganacl", "Choice Scarf", "[from] move: Trick"],
        ["", "-item", "p1a: Gholdengo", "Leftovers", "[from] move: Trick"],
    ], role="p1")
    tricks = [e for e in evs if e.type == "item_tricked"]
    assert len(tricks) == 1, "one play must not spawn two beats"
    ev = tricks[0]
    assert ev.side == "us"                       # OUR play, not theirs
    assert "Gholdengo used Trick" in ev.prose
    assert "Garganacl" in ev.prose
    assert "Choice Scarf" in ev.prose and "Leftovers" in ev.prose
    assert ev.data.get("user") == "Gholdengo"
    assert classify(ev) is not None


def test_trick_with_one_sided_item_still_names_its_user():
    """Target held nothing: still one beat, still says who did it."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "move", "p1a: Gholdengo", "Trick", "p2a: Garganacl"],
        ["", "-item", "p2a: Garganacl", "Choice Scarf", "[from] move: Trick"],
    ], role="p1")
    tricks = [e for e in evs if e.type == "item_tricked"]
    assert len(tricks) == 1
    assert "Gholdengo" in tricks[0].prose
    assert "Choice Scarf" in tricks[0].prose


def test_ability_triggered_status_is_passive_and_credits_the_holder():
    """Regression (live 2026-07-27): the active template put the ability in
    the subject slot ('Flame Body burned our Zamazenta'), and FRACTURE read
    that as the opponent acting: 'Moltres just used Flame Body to torch my
    Zamazenta'. Flame Body is a passive that fired off OUR OWN contact move.

    Round 2 (take 27 era, user-reported): even the passive 'by Moltres's
    Flame Body' left an agent slot, and FRACTURE upgraded a Static proc to
    'Zapdos clicked Static'. The prose now says ABILITY out loud with a proc
    verb — a device that went off has no click to claim."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "move", "p1a: Zamazenta", "Close Combat", "p2a: Moltres"],
        ["", "-damage", "p2a: Moltres", "55/100"],
        ["", "-status", "p1a: Zamazenta", "brn", "[from] ability: Flame Body",
         "[of] p2a: Moltres"],
    ], role="p1")
    st = [e for e in evs if e.type == "status_applied"]
    assert len(st) == 1
    prose = st[0].prose
    assert "was burned" in prose               # passive, not "Flame Body burned"
    assert not prose.startswith("Flame Body")
    assert "Moltres's Flame Body ability went off" in prose  # a proc, not a play
    assert st[0].data["cause"] == "Flame Body"  # data unchanged for the caster


def test_move_and_item_status_causes_keep_the_active_voice():
    """Only ABILITIES change voice. A move's user really did act, and an orb
    is self-inflicted, so neither can be mistaken for the opponent acting."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "-status", "p2a: Moltres", "tox", "[from] move: Toxic"],
        ["", "-status", "p1a: Gliscor", "tox", "[from] item: Toxic Orb"],
    ], role="p1")
    prose = [e.prose for e in evs if e.type == "status_applied"]
    assert any(p.startswith("Toxic ") for p in prose)
    assert any(p.startswith("Toxic Orb ") for p in prose)


def test_move_hit_names_its_target():
    """Regression (live 2026-07-27): the hit prose stated a mover and an
    effect but no victim, so FRACTURE turned 'Great Tusk's Ice Spinner landed
    super effective and a devastating blow' into 'I land a massive Ice
    Spinner' — claiming a hit she had TAKEN. The ko branch always named its
    target; hits now do too."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "move", "p2a: Great Tusk", "Ice Spinner", "p1a: Gliscor"],
        ["", "-supereffective", "p1a: Gliscor"],
        ["", "-damage", "p1a: Gliscor", "39/100"],
        ["", "move", "p1a: Gliscor", "Toxic", "p2a: Great Tusk"],
    ], role="p1")
    hits = [e for e in evs if e.type == "move_hit"]
    assert len(hits) == 1
    assert "Gliscor" in hits[0].prose
    assert hits[0].side == "us"          # we TOOK it
    # the target must sit NEXT TO the verb. Trailing it ("...landed super
    # effective and a devastating blow on Gliscor") still let a caster invert
    # who hit whom, live on 2026-07-28.
    assert hits[0].prose.startswith("their Great Tusk's Ice Spinner hit our Gliscor")


def test_recovery_is_a_beat_and_names_its_source():
    """Regression (live 2026-07-28): -heal was bookkeeping only, so a Toxapex
    clicking Recover produced NO event. The record showed our hit landing and
    the target back at 99% with nothing in between, and PRISM read six straight
    turns of 'that puts Toxapex into range for a KO' while it sat at full."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "move", "p1a: Kingambit", "Iron Head", "p2a: Toxapex"],
        ["", "-damage", "p2a: Toxapex", "50/100"],
        ["", "move", "p2a: Toxapex", "Recover", "p2a: Toxapex"],
        ["", "-heal", "p2a: Toxapex", "99/100"],
    ], role="p1")
    heals = [e for e in evs if e.type == "heal"]
    assert len(heals) == 1
    assert "99%" in heals[0].prose
    assert "Recover" in heals[0].prose          # the source, not a bare heal
    assert heals[0].side == "them"
    assert heals[0].data["cause"] == "Recover"


def test_passive_heal_trickle_stays_silent():
    """Leftovers/Poison Heal tick ~6% a turn. Emitting those would bury the
    beat feed in noise the desk would then feel obliged to narrate."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "-damage", "p2a: Toxapex", "50/100"],
        ["", "-heal", "p2a: Toxapex", "56/100", "[from] item: Leftovers"],
    ], role="p1")
    assert not [e for e in evs if e.type == "heal"]


def test_self_targeted_move_does_not_say_on_itself():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "move", "p1a: Gliscor", "Swords Dance", "p1a: Gliscor"],
        ["", "-boost", "p1a: Gliscor", "atk", "2"],
    ], role="p1")
    for e in evs:
        assert " on Gliscor" not in e.prose


def _quiet_ctx(turn, elapsed):
    """A turn with nothing notable: no faints, no swing, no events."""
    return _ctx(turn=turn, value=0.5, elapsed=elapsed)


def test_turn_gate_survives_a_fast_engine_where_the_time_floor_does_not():
    """Regression (measured 2026-07-27): unpaced, a 40-turn game resolves in
    ~3 minutes, so ctx.elapsed sits under the 5s floor almost every turn and
    the whole broadcast got FIVE beats. Under PTS the viewer watches on the
    CLIENT's clock, so density must track turns."""
    fast = 2.0   # seconds per turn with the pace hold off

    timed = Director()                       # wall-clock gating (default)
    timed._prev_value = 0.5
    spoke_timed = sum(1 for t in range(2, 22)
                      if not timed.decide(_quiet_ctx(t, fast)).silence)

    turned = Director(min_turn_gap=1, quiet_turn_gap=3)
    turned._prev_value = 0.5
    spoke_turned = sum(1 for t in range(2, 22)
                       if not turned.decide(_quiet_ctx(t, fast)).silence)

    assert spoke_timed == 0, "the 5s floor silences a fast engine entirely"
    assert spoke_turned >= 5, "turn gating keeps talking to the viewer"


def test_turn_gate_still_refuses_to_talk_twice_in_one_turn():
    d = Director(min_turn_gap=2, quiet_turn_gap=2)
    d._prev_value = 0.5
    assert not d.decide(_quiet_ctx(5, 1.0)).silence      # first beat
    assert d.decide(_quiet_ctx(6, 1.0)).silence          # gap 1 < 2
    assert not d.decide(_quiet_ctx(7, 1.0)).silence      # gap 2 == 2


def test_turn_gating_is_off_by_default():
    """min_turn_gap=0 keeps the wall-clock behaviour byte-for-byte."""
    d = Director()
    assert d.min_turn_gap == 0
    d._prev_value = 0.5
    assert d.decide(_quiet_ctx(5, 1.0)).silence          # under the 5s floor
    assert not d.decide(_quiet_ctx(6, 25.0)).silence     # past min_interval


# --- agency sweep (2026-07-27): every remaining beat whose subject was a
# mechanic, or whose actor was missing, found by auditing the prose templates
# after the KO / Trick / ability-status fixes. Same class: the beat was true,
# the attribution was not.

def test_self_inflicted_stat_drop_is_not_read_as_the_opponent_doing_it():
    """'X's Defense was cut' is the SAME sentence whether X dropped it with
    its own Close Combat or the opponent's Intimidate did. Opposite readings,
    and the caster picks one."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "move", "p2a: Great Tusk", "Headlong Rush", "p1a: Ting-Lu"],
        ["", "-damage", "p1a: Ting-Lu", "0 fnt"],
        ["", "-unboost", "p2a: Great Tusk", "def", "1"],
    ], role="p1")
    ub = [e for e in evs if e.type == "unboost"][0]
    assert "its own Defense" in ub.prose and "Headlong Rush" in ub.prose
    assert ub.data["cause"] == "Headlong Rush"


def test_ability_stat_drop_credits_the_holder():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p2a: Landorus-Therian", "Landorus-Therian", "100/100"],
        ["", "-unboost", "p1a: Gliscor", "atk", "1",
         "[from] ability: Intimidate", "[of] p2a: Landorus-Therian"],
    ], role="p1")
    ub = [e for e in evs if e.type == "unboost"][0]
    assert "Landorus-Therian's Intimidate" in ub.prose
    assert ub.side == "us"                     # OUR mon took it


def test_opponent_move_stat_drop_names_the_move_and_the_mover():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "move", "p2a: Ogerpon", "Icy Wind", "p1a: Dragonite"],
        ["", "-damage", "p1a: Dragonite", "70/100"],
        ["", "-unboost", "p1a: Dragonite", "spe", "1"],
    ], role="p1")
    ub = [e for e in evs if e.type == "unboost"][0]
    assert "Ogerpon's Icy Wind" in ub.prose and "Dragonite" in ub.prose


def test_knock_off_names_who_did_it():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "move", "p1a: Ting-Lu", "Knock Off", "p2a: Blissey"],
        ["", "-enditem", "p2a: Blissey", "Leftovers", "[from] move: Knock Off"],
    ], role="p1")
    ko = [e for e in evs if e.type == "item_knocked_off"][0]
    assert "Ting-Lu's Knock Off" in ko.prose and "Blissey" in ko.prose
    assert "Leftovers" in ko.prose
    assert ko.data["user"] == "Ting-Lu"


def test_theft_names_the_thief_on_both_protocol_halves():
    """The -item half already named the thief actively; the -enditem half was
    passive, so the same steal read as an event with no perpetrator."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "move", "p2a: Weavile", "Thief", "p1a: Gliscor"],
        ["", "-enditem", "p1a: Gliscor", "Toxic Orb", "[from] move: Thief"],
    ], role="p1")
    st = [e for e in evs if e.type == "item_stolen"][0]
    assert "Weavile swiped" in st.prose
    assert st.data["user"] == "Weavile"


def test_a_miss_names_who_dodged():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "move", "p2a: Kingambit", "Sucker Punch", "p1a: Darkrai"],
        ["", "-miss", "p2a: Kingambit", "p1a: Darkrai"],
        ["", "move", "p1a: Darkrai", "Dark Pulse", "p2a: Kingambit"],
    ], role="p1")
    mm = [e for e in evs if e.type == "move_missed"][0]
    assert "missed our Darkrai" in mm.prose


def test_untagged_drop_with_no_known_move_stays_neutral():
    """No cause available: keep the passive wording rather than invent an
    actor — never guess agency. The SUBJECT is still side-marked (take 30:
    four unmarked Calm Mind beats became 'THEY CLICKED CALM MIND ON THE
    IRON CROWN', ours); whose stat it is was never in doubt, only who did it."""
    sc = ProtocolScanner()
    evs = sc.scan([["", "-unboost", "p1a: Gliscor", "spe", "1"]], role="p1")
    ub = [e for e in evs if e.type == "unboost"][0]
    assert ub.prose == "our Gliscor's Speed was cut"
    assert ub.data["cause"] is None


def test_field_effects_name_who_set_them():
    """'Rain set in' / 'Spikes went up on their side' read as weather, in the
    idiomatic sense: things that merely happen. Which side owns a screen, a
    hazard layer or the weather is usually the whole tactical point."""
    sc = ProtocolScanner()
    haz = sc.scan([
        ["", "move", "p1a: Gliscor", "Spikes", "p2a: Great Tusk"],
        ["", "-sidestart", "p2: FPAiri", "Spikes"],
    ], role="p1")
    h = [e for e in haz if e.type == "hazard_set"][0]
    assert "Gliscor set Spikes" in h.prose and "their side" in h.prose

    sc = ProtocolScanner()
    rain = sc.scan([
        ["", "switch", "p2a: Pelipper", "Pelipper", "100/100"],
        ["", "-weather", "RainDance", "[from] ability: Drizzle",
         "[of] p2a: Pelipper"],
    ], role="p1")
    w = [e for e in rain if e.type == "weather_set"][0]
    assert "Pelipper's Drizzle" in w.prose

    sc = ProtocolScanner()
    tr = sc.scan([
        ["", "move", "p2a: Iron Valiant", "Trick Room", "p2a: Iron Valiant"],
        ["", "-fieldstart", "move: Trick Room"],
    ], role="p1")
    f = [e for e in tr if e.type == "field_start"][0]
    assert "Iron Valiant brought up Trick Room" in f.prose


def test_field_effect_with_no_known_cause_keeps_the_old_wording():
    """Never guess a setter: with nothing to attribute to, stay impersonal."""
    sc = ProtocolScanner()
    evs = sc.scan([["", "-weather", "Sandstorm"]], role="p1")
    w = [e for e in evs if e.type == "weather_set"][0]
    assert w.prose == "sand set in" or w.prose.endswith("set in")
    assert w.data["user"] is None


def test_residual_ko_names_what_actually_killed_it():
    """Regression (live 2026-07-27): a faint that is NOT a move's finishing
    blow read as a bare "X went down" directly after X's OWN attack, so
    FRACTURE announced "EARTHQUAKE TOOK THE BODY" when Ting-Lu had died to
    its own poison — crediting a mon's attack with killing that mon."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "move", "p2a: Ting-Lu", "Earthquake", "p1a: Zamazenta"],
        ["", "-damage", "p1a: Zamazenta", "66/100"],
        ["", "-damage", "p2a: Ting-Lu", "0 fnt", "[from] psn"],
        ["", "faint", "p2a: Ting-Lu"],
    ], role="p1")
    ko = [e for e in evs if e.type == "ko" and e.data.get("residual")][0]
    assert "went down to the poison" in ko.prose
    assert ko.data["cause"] == "psn"
    assert "Earthquake" not in ko.prose


def test_residual_ko_stays_bare_when_nothing_names_a_cause():
    """Never guess: no [from] tag, no invented cause."""
    sc = ProtocolScanner()
    evs = sc.scan([["", "-damage", "p2a: Blissey", "0 fnt"],
                   ["", "faint", "p2a: Blissey"]], role="p1")
    ko = [e for e in evs if e.type == "ko"][0]
    assert ko.prose == "Blissey went down"
    assert ko.data["cause"] is None


def test_cosmetic_stat_drop_does_not_compel_a_turn():
    """Live 2026-07-28: beat "Clefable's Moonblast cut Kingambit's Special
    Attack" -> FRACTURE: "THAT MOONBLAST WAS A DISASTER! My Kingambit is
    sitting here with his Special Attack gutted". Kingambit is 135 Atk / 60
    SpA; the drop is cosmetic. The inflation follows from the turn being
    COMPELLED, so an unused-stat change must not mark it notable.
    """
    d = Director(stats_fn=_stats)
    d.observe([Event("unboost", "Clefable's Moonblast cut Kingambit's "
                     "Special Attack", side="us", notable=True,
                     data={"mon": "Kingambit", "stat": "spa", "amount": 1})])
    assert d._notable is False

    # the same drop on the stat it actually uses still compels
    d2 = Director(stats_fn=_stats)
    d2.observe([Event("unboost", "Kingambit's Attack was cut", side="us",
                      notable=True,
                      data={"mon": "Kingambit", "stat": "atk", "amount": 1})])
    assert d2._notable is True

    # and a special attacker's SpA drop is not cosmetic either
    d3 = Director(stats_fn=_stats)
    d3.observe([Event("unboost", "Hatterene's Special Attack was cut",
                      side="us", notable=True,
                      data={"mon": "Hatterene", "stat": "spa", "amount": 1})])
    assert d3._notable is True


def test_cosmetic_stat_change_never_suppresses_a_mixed_attacker():
    """The threshold exists to protect mixed attackers: Kyurem 130/130, Iron
    Valiant 130/120, Kommo-o 110/100 all genuinely care about both stats."""
    def mixed(name):
        return {"Kyurem": (130, 130), "Iron Valiant": (130, 120)}.get(name)
    d = Director(stats_fn=mixed)
    for mon in ("Kyurem", "Iron Valiant"):
        for stat in ("atk", "spa"):
            ev = Event("unboost", f"{mon} drop", side="us", notable=True,
                       data={"mon": mon, "stat": stat, "amount": 1})
            assert d._cosmetic_stat_change(ev) is False, (mon, stat)


def test_real_stats_beat_base_stats_for_commitment():
    """User's point (2026-07-28): EVs and nature decide what a mon is BUILT to
    do, and base stats can say "mixed" about a spread that is nothing of the
    kind. Iron Valiant's base 130/120 reads mixed; the 252+ Atk build it
    actually runs does not care about a Special Attack drop.

    Showdown sends our own side's computed stats, so this is a lookup for us
    and necessarily still a species-level guess for the opponent.
    """
    ev = Event("unboost", "Iron Valiant's Special Attack was cut", side="us",
               notable=True,
               data={"mon": "Iron Valiant", "stat": "spa", "amount": 1})

    def base_only(name, side=None):
        return {"Iron Valiant": (130, 120)}.get(name)

    def real_for_us(name, side=None):
        if side == "us" and name == "Iron Valiant":
            return (359, 197)          # 252+ Atk, SpA uninvested
        return {"Iron Valiant": (130, 120)}.get(name)

    assert Director(stats_fn=base_only)._cosmetic_stat_change(ev) is False
    assert Director(stats_fn=real_for_us)._cosmetic_stat_change(ev) is True


def test_one_arg_stats_fn_still_supported():
    """The gold-set eval and older callers pass stats_fn(name); the side-aware
    form must stay optional or the eval breaks."""
    d = Director(stats_fn=lambda name: {"Kingambit": (135, 60)}.get(name))
    ev = Event("unboost", "cut", side="us", notable=True,
               data={"mon": "Kingambit", "stat": "spa", "amount": 1})
    assert d._cosmetic_stat_change(ev) is True


def test_item_consumption_says_what_it_did():
    """Consuming an item is usually the item WORKING, but "used up its Booster
    Energy" is loss-coded before the caster reads it. Measured live
    2026-07-28: 4 occurrences in 147 beats, every one framed as a loss, the
    clearest being "The Booster Energy is gone, so we lost our speed
    advantage" — backwards, since spending it is what switches Quark Drive on.
    """
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Iron Valiant", "Iron Valiant", "100/100"],
        ["", "-enditem", "p1a: Iron Valiant", "Booster Energy"],
    ], role="p1")
    used = next(e for e in evs if e.type == "item_used")
    assert "kicked in" in used.prose
    assert "used up" not in used.prose

    # an unlisted consumable still must not be loss-coded
    sc2 = ProtocolScanner()
    evs2 = sc2.scan([
        ["", "switch", "p1a: Iron Valiant", "Iron Valiant", "100/100"],
        ["", "-enditem", "p1a: Iron Valiant", "Some New Gadget"],
    ], role="p1")
    assert "activated" in next(e for e in evs2
                               if e.type == "item_used").prose


def test_genuinely_bad_item_losses_keep_their_framing():
    """The default is activation because the BAD cases are narrow and already
    have their own prose — this pins that the change did not sweep them up."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Great Tusk", "Great Tusk", "100/100"],
        ["", "-enditem", "p1a: Great Tusk", "Air Balloon"],
    ], role="p1")
    assert "popped" in next(e for e in evs
                            if e.type == "balloon_popped").prose

    sc2 = ProtocolScanner()
    evs2 = sc2.scan([
        ["", "switch", "p2a: Gliscor", "Gliscor", "100/100"],
        ["", "move", "p2a: Gliscor", "Knock Off", "p1a: Kingambit"],
        ["", "-enditem", "p1a: Kingambit", "Leftovers",
         "[from] move: Knock Off"],
    ], role="p1")
    off = next(e for e in evs2 if e.type == "item_knocked_off")
    assert "Knock Off" in off.prose and "Leftovers" in off.prose


def test_item_events_say_whose_item_it_was():
    """Whose item fired IS the fact — the boost or the save belongs to a side.
    `qual` only disambiguates a species mirror, so "Iron Crown's Booster Energy
    kicked in" left the owner to be guessed. Live 2026-07-28, FRACTURE guessed
    wrong about OUR Iron Crown ("THEY BROUGHT THE BOOST ENERGY! SKARMORY JUST
    SNATCHED THE MOMENTUM") while the prompt's on-field block named ours
    correctly — so stating it in the direction was not enough.
    """
    def scan(tok, role="p1", item="Booster Energy"):
        sc = ProtocolScanner()
        evs = sc.scan([
            ["", "switch", tok, "Iron Crown", "100/100"],
            ["", "-enditem", tok, item],
        ], role=role)
        return next(e for e in evs
                    if e.type in ("item_used", "sash_saved",
                                  "balloon_popped")).prose

    assert scan("p1a: Iron Crown").startswith("our Iron Crown")
    assert scan("p2a: Iron Crown").startswith("their Iron Crown")
    assert scan("p1a: Iron Crown", item="Focus Sash").startswith("our ")
    assert scan("p2a: Iron Crown", item="Air Balloon").startswith("their ")


def _moves(mapping):
    return lambda mon, side=None: mapping.get(mon, [])


def test_revealed_moves_beat_base_stats_for_the_opponent():
    """Their EV spread is never revealed, so base stats are the only fallback
    and they UNDER-suppress: Iron Valiant's 130/120 reads "mixed" even when
    the thing in front of us has only ever thrown physical attacks. What it
    has actually clicked is evidence, and it sharpens every turn."""
    ev = Event("unboost", "Iron Valiant's Special Attack was cut", side="them",
               notable=True,
               data={"mon": "Iron Valiant", "stat": "spa", "amount": 1})
    stats = lambda name, side=None: {"Iron Valiant": (130, 120)}.get(name)

    # base stats alone: mixed, so not suppressed
    assert Director(stats_fn=stats)._cosmetic_stat_change(ev) is False
    # two physical attacks revealed, no special: SpA is not in use
    d = Director(stats_fn=stats,
                 moves_fn=_moves({"Iron Valiant": ["physical", "physical"]}))
    assert d._cosmetic_stat_change(ev) is True


def test_revealed_moves_need_real_evidence():
    """One attack proves nothing — most mons open with one — and a mixed read
    must fall through. Over-suppression is the dangerous direction: a silenced
    moment leaves nothing in the transcript to notice."""
    ev = Event("unboost", "cut", side="them", notable=True,
               data={"mon": "Iron Valiant", "stat": "spa", "amount": 1})
    stats = lambda name, side=None: {"Iron Valiant": (130, 120)}.get(name)
    for revealed in ([], ["physical"], ["physical", "special"],
                     ["special", "physical", "physical"]):
        d = Director(stats_fn=stats,
                     moves_fn=_moves({"Iron Valiant": revealed}))
        assert d._cosmetic_stat_change(ev) is False, revealed


def test_revealed_moves_only_ever_add_suppression():
    """A physical-only read must not UN-suppress a mon the stat ratio already
    called committed."""
    ev = Event("unboost", "cut", side="them", notable=True,
               data={"mon": "Kingambit", "stat": "spa", "amount": 1})
    d = Director(stats_fn=lambda n, s=None: {"Kingambit": (135, 60)}.get(n),
                 moves_fn=_moves({"Kingambit": ["physical", "special"]}))
    assert d._cosmetic_stat_change(ev) is True


def test_spin_with_nothing_to_clear_says_so():
    """The record was silent about hazards during a long Rapid Spin stretch,
    so both casters invented one — six hazard-clear claims in take 26. A clear
    that never happens emits no event, so the beat has to state the absence."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Iron Treads", "Iron Treads", "100/100"],
        ["", "switch", "p2a: Zapdos", "Zapdos", "100/100"],
        ["", "move", "p1a: Iron Treads", "Rapid Spin", "p2a: Zapdos"],
        ["", "-damage", "p2a: Zapdos", "91/100"],
    ], role="p1")
    note = [e for e in evs if e.type == "spin_no_hazards"]
    assert note and "no hazards to clear" in note[0].prose
    assert not note[0].notable          # colour, never forces a turn


def test_spin_that_really_clears_stays_quiet_about_absence():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Iron Treads", "Iron Treads", "100/100"],
        ["", "switch", "p2a: Zapdos", "Zapdos", "100/100"],
        ["", "-sidestart", "p1: wiz", "Stealth Rock"],
        ["", "move", "p1a: Iron Treads", "Rapid Spin", "p2a: Zapdos"],
        ["", "-sideend", "p1: wiz", "Stealth Rock", "[from] move: Rapid Spin"],
    ], role="p1")
    assert not [e for e in evs if e.type == "spin_no_hazards"]
    assert [e for e in evs if e.type == "hazard_cleared"]


def test_spin_only_clears_the_users_own_side():
    """Their rocks are up, ours are clean — Rapid Spin clears the USER's side,
    so it still had nothing to clear."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Iron Treads", "Iron Treads", "100/100"],
        ["", "switch", "p2a: Zapdos", "Zapdos", "100/100"],
        ["", "-sidestart", "p2: opp", "Stealth Rock"],
        ["", "move", "p1a: Iron Treads", "Rapid Spin", "p2a: Zapdos"],
    ], role="p1")
    assert [e for e in evs if e.type == "spin_no_hazards"]


# --- actor attribution sweep (2026-07-29, after take 27's Substitute bug) ---
# Passive, causeless prose is the caster's invitation to invent the actor —
# the class behind the hazard-clear, Trick, Knock Off and Substitute bugs.
# These pin every construction the sweep converted to name its actor, plus
# the fallbacks that must STAY passive when no cause is on record.

def test_sub_break_names_the_breaker():
    """Take 27 T6: 'Kommo-o's Substitute broke' (no actor) became PRISM
    crediting our not-yet-thrown Shadow Claw with breaking our OWN sub."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Kommo-o", "Kommo-o", "100/100"],
        ["", "switch", "p2a: Kingambit", "Kingambit", "100/100"],
        ["", "move", "p2a: Kingambit", "Iron Head", "p1a: Kommo-o"],
        ["", "-end", "p1a: Kommo-o", "Substitute"],
    ], role="p1")
    ends = [e for e in evs if e.type == "volatile_end"]
    assert ends
    assert ends[0].prose == ("their Kingambit's Iron Head broke "
                             "Kommo-o's Substitute")
    assert ends[0].data["breaker"] == "Kingambit"


def test_sub_break_without_a_cause_stays_passive():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Kommo-o", "Kommo-o", "100/100"],
        ["", "-end", "p1a: Kommo-o", "Substitute"],
    ], role="p1")
    ends = [e for e in evs if e.type == "volatile_end"]
    assert ends and ends[0].prose == "Kommo-o's Substitute broke"


def test_volatile_starts_name_the_inflictor():
    cases = [
        ("Encore", "Encore", "their Grimmsnarl locked Kyurem in with Encore"),
        ("Taunt", "Taunt", "their Grimmsnarl shut Kyurem down with Taunt"),
        ("Leech Seed", "move: Leech Seed",
         "their Grimmsnarl planted Leech Seed into Kyurem"),
        ("Yawn", "move: Yawn",
         "their Grimmsnarl's Yawn is making Kyurem drowsy"),
        ("Attract", "Attract",
         "their Grimmsnarl left Kyurem infatuated with Attract"),
    ]
    for move, cond, want in cases:
        sc = ProtocolScanner()
        evs = sc.scan([
            ["", "switch", "p1a: Kyurem", "Kyurem", "100/100"],
            ["", "switch", "p2a: Grimmsnarl", "Grimmsnarl", "100/100"],
            ["", "move", "p2a: Grimmsnarl", move, "p1a: Kyurem"],
            ["", "-start", "p1a: Kyurem", cond],
        ], role="p1")
        starts = [e for e in evs if e.type == "volatile_start"]
        assert starts and starts[0].prose == want, (move, starts)
        assert starts[0].data["user"] == "Grimmsnarl"


def test_disable_names_the_disabled_move():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Kyurem", "Kyurem", "100/100"],
        ["", "switch", "p2a: Gengar", "Gengar", "100/100"],
        ["", "move", "p2a: Gengar", "Disable", "p1a: Kyurem"],
        ["", "-start", "p1a: Kyurem", "Disable", "Freeze-Dry"],
    ], role="p1")
    starts = [e for e in evs if e.type == "volatile_start"]
    assert starts
    assert starts[0].prose == "their Gengar disabled Kyurem's Freeze-Dry"


def test_confusion_from_fatigue_is_self_inflicted():
    """Outrage fatigue: passive 'became confused' lets a caster hand the
    opponent credit for something the mon did to itself."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Kyurem", "Kyurem", "100/100"],
        ["", "-start", "p1a: Kyurem", "confusion", "[fatigue]"],
    ], role="p1")
    starts = [e for e in evs if e.type == "volatile_start"]
    assert starts
    assert starts[0].prose == "Kyurem wore itself out into confusion"
    assert starts[0].data["cause"] == "fatigue"


def test_confusion_from_a_known_confuser_names_it():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Kyurem", "Kyurem", "100/100"],
        ["", "switch", "p2a: Gengar", "Gengar", "100/100"],
        ["", "move", "p2a: Gengar", "Confuse Ray", "p1a: Kyurem"],
        ["", "-start", "p1a: Kyurem", "confusion"],
    ], role="p1")
    starts = [e for e in evs if e.type == "volatile_start"]
    assert starts
    assert starts[0].prose == "their Gengar's Confuse Ray left Kyurem confused"


def test_ability_caused_volatile_stays_passive():
    """Cute Charm: the -start lands on the ATTACKER right after its own
    contact move, so lm is the victim's own move and must NOT bind — the
    Flame Body lesson says ability procs keep the passive voice."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Kyurem", "Kyurem", "100/100"],
        ["", "switch", "p2a: Enamorus", "Enamorus", "100/100"],
        ["", "move", "p1a: Kyurem", "Icicle Spear", "p2a: Enamorus"],
        ["", "-start", "p1a: Kyurem", "Attract",
         "[from] ability: Cute Charm", "[of] p2a: Enamorus"],
    ], role="p1")
    starts = [e for e in evs if e.type == "volatile_start"]
    assert starts and starts[0].prose == "Kyurem became infatuated"


def test_haze_names_its_user():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Kyurem", "Kyurem", "100/100"],
        ["", "switch", "p2a: Toxapex", "Toxapex", "100/100"],
        ["", "move", "p2a: Toxapex", "Haze"],
        ["", "-clearallboost"],
    ], role="p1")
    clears = [e for e in evs if e.type == "boosts_cleared"]
    assert clears
    assert clears[0].prose == ("their Toxapex's Haze wiped away "
                               "every stat change")


def test_clear_smog_names_user_and_victim():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Kyurem", "Kyurem", "100/100"],
        ["", "switch", "p2a: Toxapex", "Toxapex", "100/100"],
        ["", "move", "p2a: Toxapex", "Clear Smog", "p1a: Kyurem"],
        ["", "-damage", "p1a: Kyurem", "88/100"],
        ["", "-clearboost", "p1a: Kyurem"],
    ], role="p1")
    clears = [e for e in evs if e.type == "boosts_cleared"]
    assert clears
    assert clears[0].prose == ("their Toxapex's Clear Smog cleared "
                               "Kyurem's boosts")


def test_tailwind_names_the_setter():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Zapdos", "Zapdos", "100/100"],
        ["", "move", "p1a: Zapdos", "Tailwind"],
        ["", "-sidestart", "p1: wiz", "move: Tailwind"],
    ], role="p1")
    tw = [e for e in evs if e.type == "tailwind_up"]
    assert tw
    assert tw[0].prose == "our Zapdos set Tailwind for our side"


# --- take 28 audit fixes: fails, status causes, residual windows, switches --

def test_failed_move_is_narrated():
    """Five silent fails in take 28's endgame; PRISM invented 'The Sucker
    Punch connects' into the gap. A fail is a fact — say it."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Kingambit", "Kingambit", "100/100"],
        ["", "switch", "p2a: Cinderace", "Cinderace", "100/100"],
        ["", "move", "p1a: Kingambit", "Sucker Punch", "p2a: Cinderace"],
        ["", "-fail", "p1a: Kingambit"],
    ], role="p1")
    fails = [e for e in evs if e.type == "move_failed"]
    assert fails
    assert fails[0].prose == ("our Kingambit's Sucker Punch failed "
                              "against their Cinderace")


def test_residual_burn_damage_not_credited_to_a_failed_move():
    """Take 28 T27/T30: Will-O-Wisp FAILED (already burned), then end-of-turn
    burn chip landed in its damage window and aired as 'Will-O-Wisp hit our
    Kingambit — barely a scratch'. [from]-tagged damage is never the move's."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Kingambit", "Kingambit", "100/100"],
        ["", "switch", "p2a: Cinderace", "Cinderace", "100/100"],
        ["", "move", "p2a: Cinderace", "Will-O-Wisp", "p1a: Kingambit"],
        ["", "-fail", "p1a: Kingambit", "brn"],
        ["", "-heal", "p1a: Kingambit", "24/100 brn",
         "[from] item: Leftovers"],
        ["", "-damage", "p1a: Kingambit", "18/100 brn", "[from] brn"],
    ], role="p1")
    fails = [e for e in evs if e.type == "move_failed"]
    assert fails and fails[0].prose == ("their Cinderace's Will-O-Wisp "
                                        "failed against our Kingambit")
    assert not [e for e in evs if e.type == "move_hit"]


def test_clean_status_move_names_actor_and_move():
    """Take 28 T24: 'burned our Kingambit' — no actor, no move (a direct
    status carries no [from] tag and a clean status move emits no event).
    FRACTURE invented 'Flamethrower'. The last move is the cause."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Zamazenta", "Zamazenta", "100/100"],
        ["", "switch", "p2a: Cinderace", "Cinderace", "100/100"],
        ["", "move", "p2a: Cinderace", "Will-O-Wisp", "p1a: Zamazenta"],
        ["", "-status", "p1a: Zamazenta", "brn"],
    ], role="p1")
    st = [e for e in evs if e.type == "status_applied"]
    assert st
    assert st[0].prose == ("their Cinderace's Will-O-Wisp burned Zamazenta")


def test_their_switch_is_narrated_ours_is_not():
    """Take 28 T11: their Great Tusk appeared board-only and FRACTURE said
    'THEY BROUGHT IN IRON TREADS' — ours. Their replacements get an event;
    our chosen switches stay with the decision prose; leads stay quiet."""
    sc = ProtocolScanner()
    evs = sc.scan([
        # leads — no events
        ["", "switch", "p1a: Kyurem", "Kyurem", "100/100"],
        ["", "switch", "p2a: Kingambit", "Kingambit", "100/100"],
        # their replacement — narrated
        ["", "switch", "p2a: Great Tusk", "Great Tusk", "78/100"],
        # our switch — NOT narrated here (decision prose covers it)
        ["", "switch", "p1a: Iron Treads", "Iron Treads", "100/100"],
    ], role="p1")
    sw = [e for e in evs if e.type == "opp_switch"]
    assert len(sw) == 1
    assert sw[0].prose == "they go to Great Tusk"
    assert sw[0].data["prev"] == "Kingambit"


def test_drag_is_narrated_for_both_sides():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Kyurem", "Kyurem", "100/100"],
        ["", "switch", "p2a: Zapdos", "Zapdos", "100/100"],
        ["", "drag", "p1a: Kommo-o", "Kommo-o", "100/100"],
        ["", "drag", "p2a: Kyurem", "Kyurem", "100/100"],
    ], role="p1")
    forced = [e for e in evs if e.type == "forced_switch"]
    assert [e.prose for e in forced] == [
        "we were dragged out — Kommo-o is in",
        "they were dragged out — their Kyurem is in",
    ]


# --- dice ledger: rage that builds (user-requested) -------------------------

def _miss(sc, mover, move, target):
    return sc.scan([
        ["", "move", mover, move, target],
        ["", "-miss", mover, target],
        ["", "move", target, "Recover", target],   # flush the miss
    ], role="p1")


def test_dice_ledger_counts_misses_and_escalates_prose():
    from crystal_broadcast.beat_director import classify
    sc = ProtocolScanner()
    sc.scan([
        ["", "switch", "p1a: Kommo-o", "Kommo-o", "100/100"],
        ["", "switch", "p2a: Toxapex", "Toxapex", "100/100"],
    ], role="p1")
    evs1 = _miss(sc, "p1a: Kommo-o", "Focus Blast", "p2a: Toxapex")
    m1 = [e for e in evs1 if e.type == "move_missed"][0]
    assert "dice" not in m1.prose                # first miss: no counter yet
    assert m1.data["luck_count"] == 1
    _miss(sc, "p1a: Kommo-o", "Focus Blast", "p2a: Toxapex")
    evs3 = _miss(sc, "p1a: Kommo-o", "Focus Blast", "p2a: Toxapex")
    m3 = [e for e in evs3 if e.type == "move_missed"][0]
    assert m3.prose.endswith(
        "— the third time the dice have gone against us this game")
    # by the third the register escalates: accumulation, not instance
    assert classify(m3).register == "escalating-grievance"


def test_their_dice_streak_becomes_rejoicing():
    from crystal_broadcast.beat_director import classify
    sc = ProtocolScanner()
    sc.scan([
        ["", "switch", "p1a: Kommo-o", "Kommo-o", "100/100"],
        ["", "switch", "p2a: Toxapex", "Toxapex", "100/100"],
    ], role="p1")
    for _ in range(2):
        _miss(sc, "p2a: Toxapex", "Toxic", "p1a: Kommo-o")
    evs = _miss(sc, "p2a: Toxapex", "Toxic", "p1a: Kommo-o")
    m = [e for e in evs if e.type == "move_missed"][0]
    assert m.prose.endswith(
        "— the third time the dice have gone against them this game")
    assert classify(m).register == "rejoicing"


def test_cant_move_shares_the_ledger_but_recharge_does_not():
    sc = ProtocolScanner()
    sc.scan([
        ["", "switch", "p1a: Kommo-o", "Kommo-o", "100/100"],
        ["", "switch", "p2a: Toxapex", "Toxapex", "100/100"],
    ], role="p1")
    _miss(sc, "p1a: Kommo-o", "Focus Blast", "p2a: Toxapex")
    evs = sc.scan([["", "cant", "p1a: Kommo-o", "par"]], role="p1")
    c = [e for e in evs if e.type == "cant_move"][0]
    assert c.prose.endswith(
        "— the second time the dice have gone against us this game")
    assert c.data["luck_count"] == 2
    # recharge is a mechanic the mover chose, never a dice event
    evs = sc.scan([["", "cant", "p1a: Kommo-o", "recharge"]], role="p1")
    r = [e for e in evs if e.type == "cant_move"][0]
    assert "dice" not in r.prose and r.data["luck_count"] is None


def test_boost_subject_is_always_side_marked():
    """Take 30 T5-T8: 'Iron Crown raised its Special Attack with Calm Mind'
    (unmarked, non-mirror) four beats running became 'THEY CLICKED CALM MIND
    ON THE IRON CROWN' — ours. Setup is the most side-critical fact after
    moves; mark it like moves."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Iron Crown", "Iron Crown", "100/100"],
        ["", "switch", "p2a: Kingambit", "Kingambit", "100/100"],
        ["", "move", "p1a: Iron Crown", "Calm Mind", "p1a: Iron Crown"],
        ["", "-boost", "p1a: Iron Crown", "spa", "1"],
        ["", "-boost", "p1a: Iron Crown", "spd", "1"],
        ["", "move", "p2a: Kingambit", "Swords Dance", "p2a: Kingambit"],
        ["", "-boost", "p2a: Kingambit", "atk", "2"],
    ], role="p1")
    boosts = [e for e in evs if e.type == "boost"]
    assert boosts[0].prose == ("our Iron Crown raised its Special Attack "
                               "with Calm Mind")
    assert boosts[2].prose == ("their Kingambit sharply raised its Attack "
                               "with Swords Dance")


# --- take 48 record bugs: stale residual + orphaned KO ----------------------

def test_new_occupant_does_not_inherit_slot_residual():
    """Take 48 T30, protocol-verified: the ONLY burn all game was Iron
    Valiant's, yet 'Kommo-o went down to the burn' aired — Valiant's residual
    sat under p1a and Kommo-o's unattributed faint popped it eleven turns
    later. The record itself fabricated a cause."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Iron Valiant", "Iron Valiant", "100/100"],
        ["", "switch", "p2a: Cinderace", "Cinderace", "100/100"],
        ["", "move", "p2a: Cinderace", "Will-O-Wisp", "p1a: Iron Valiant"],
        ["", "-status", "p1a: Iron Valiant", "brn"],
        ["", "-damage", "p1a: Iron Valiant", "88/100 brn", "[from] brn"],
        ["", "faint", "p1a: Iron Valiant"],
        ["", "switch", "p1a: Kommo-o", "Kommo-o", "100/100"],
        ["", "move", "p2a: Cinderace", "Pyro Ball", "p1a: Kommo-o"],
        ["", "-damage", "p1a: Kommo-o", "10/100"],
        ["", "-damage", "p1a: Kommo-o", "0 fnt"],
        ["", "faint", "p1a: Kommo-o"],
    ], role="p1")
    kos = [e for e in evs if e.type == "ko" and e.data.get("target") == "Kommo-o"]
    assert kos
    assert "burn" not in kos[0].prose


def test_ko_binds_when_self_effects_flush_the_move():
    """Headlong Rush: move, damage, TWO self-unboosts (which flush), faint.
    The KO used to fall unattributed into the residual path ('Iron Treads
    went down', take 28 T14) — the parked last move still knows its target."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Kommo-o", "Kommo-o", "100/100"],
        ["", "switch", "p2a: Great Tusk", "Great Tusk", "100/100"],
        ["", "move", "p2a: Great Tusk", "Headlong Rush", "p1a: Kommo-o"],
        ["", "-damage", "p1a: Kommo-o", "0 fnt"],
        ["", "-unboost", "p2a: Great Tusk", "def", "1"],
        ["", "-unboost", "p2a: Great Tusk", "spd", "1"],
        ["", "faint", "p1a: Kommo-o"],
    ], role="p1")
    kos = [e for e in evs if e.type == "ko"]
    assert kos
    assert kos[0].prose == ("their Great Tusk's Headlong Rush knocked out "
                            "Kommo-o")
    assert kos[0].data["move"] == "Headlong Rush"


# --- field-state footer (weather / screens / boosts) -----------------------

def test_field_state_footer_from_events():
    """The Bodies: contract extended: weather, screens and current-active
    boosts reconstructed from observed events surface as a footer the
    caster's state guards can check claims against."""
    d = Director()
    d.observe([
        Event("weather_set",
              "their Ninetales's Drought ability set harsh sun up",
              notable=True,
              data={"weather": "harsh sun", "user": "Ninetales"}),
        Event("screens_set", "their Grimmsnarl put Reflect up on their side",
              side="them", data={"condition": "Reflect",
                                 "user": "Grimmsnarl"}),
        Event("boost",
              "our Kingambit sharply raised its Attack with Swords Dance",
              side="us", notable=True,
              data={"mon": "Kingambit", "stat": "atk", "amount": 2,
                    "cause": "Swords Dance"}),
    ])
    dec = d.decide(_ctx(turn=7, me_name="Kingambit", me_hp=88,
                        opp_name="Grimmsnarl", opp_hp=100))
    assert "Weather: harsh sun." in dec.text
    assert "Screens: their Reflect." in dec.text
    assert "Boosts: our Kingambit +2 Attack." in dec.text


def test_field_state_footer_absent_when_nothing_up():
    d = Director()
    dec = d.decide(_ctx(turn=3))
    for label in ("Weather:", "Screens:", "Boosts:"):
        assert label not in dec.text


def test_field_state_footer_clears():
    """weather_cleared / screens_wore_off / a sideless boosts_cleared (Haze)
    empty the footer again."""
    d = Director()
    d.observe([
        Event("weather_set", "rain set in", notable=True,
              data={"weather": "rain"}),
        Event("screens_set", "Light Screen went up on our side", side="us",
              data={"condition": "Light Screen"}),
        Event("boost", "our Gliscor raised its Speed", side="us",
              notable=True,
              data={"mon": "Gliscor", "stat": "spe", "amount": 1}),
    ])
    dec = d.decide(_ctx(turn=4))
    assert "Weather: rain." in dec.text
    assert "Screens: our Light Screen." in dec.text
    assert "Boosts: our Gliscor +1 Speed." in dec.text
    d.observe([
        Event("weather_cleared", "the weather cleared", notable=True),
        Event("screens_wore_off", "our Light Screen wore off", side="us",
              data={"condition": "Light Screen"}),
        Event("boosts_cleared", "every stat change was wiped away",
              notable=True),
    ])
    dec2 = d.decide(_ctx(turn=6))
    for label in ("Weather:", "Screens:", "Boosts:"):
        assert label not in dec2.text


def test_boost_stages_accumulate_and_retire_on_switch():
    """+1 twice reads +2; a mon that switches out (event-less for our own
    side) must neither display while benched nor resurface dead stages when
    it re-enters later — ctx naming a new active retires them."""
    d = Director()
    d.observe([Event("boost", "our Kingambit raised its Attack", side="us",
                     notable=True,
                     data={"mon": "Kingambit", "stat": "atk", "amount": 1}),
               Event("boost", "our Kingambit raised its Attack", side="us",
                     notable=True,
                     data={"mon": "Kingambit", "stat": "atk", "amount": 1})])
    dec = d.decide(_ctx(turn=4, me_name="Kingambit"))
    assert "Boosts: our Kingambit +2 Attack." in dec.text
    dec2 = d.decide(_ctx(turn=5, me_name="Gliscor"))
    assert "Boosts:" not in dec2.text
    dec3 = d.decide(_ctx(turn=8, me_name="Kingambit"))
    assert "Boosts:" not in dec3.text


def test_their_boosts_clear_on_their_switch():
    d = Director()
    d.observe([Event("boost", "their Volcarona raised its Special Attack",
                     side="them", notable=True,
                     data={"mon": "Volcarona", "stat": "spa", "amount": 1})])
    dec = d.decide(_ctx(turn=4, opp_name="Volcarona"))
    assert "Boosts: their Volcarona +1 Special Attack." in dec.text
    d.observe([Event("opp_switch", "they go to Heatran", side="them",
                     data={"mon": "Heatran", "prev": "Volcarona"})])
    dec2 = d.decide(_ctx(turn=5, opp_name="Heatran"))
    assert "Boosts:" not in dec2.text


def test_court_change_swaps_screens_sides():
    d = Director()
    d.observe([Event("screens_set", "their Grimmsnarl put Reflect up on "
                     "their side", side="them",
                     data={"condition": "Reflect"}),
               Event("hazard_flip", "Court Change swapped the hazards and "
                     "screens onto the opposite sides", notable=True)])
    dec = d.decide(_ctx(turn=6))
    assert "Screens: our Reflect." in dec.text


def test_field_footer_end_to_end_from_protocol():
    """Scanner -> Director round trip: raw -weather / -sidestart lines end
    up in the footer with display labels and the right possessives."""
    sc = ProtocolScanner()
    d = Director()
    evs = sc.scan([
        ["", "switch", "p1a: Ninetales", "Ninetales, F", "100/100"],
        ["", "switch", "p2a: Grimmsnarl", "Grimmsnarl, M", "100/100"],
        ["", "-weather", "SunnyDay", "[from] ability: Drought",
         "[of] p1a: Ninetales"],
        ["", "-sidestart", "p2: someone", "Reflect"],
    ], role="p2")
    d.observe(evs)
    dec = d.decide(_ctx(turn=2, me_name="Grimmsnarl", opp_name="Ninetales"))
    assert "Weather: harsh sun." in dec.text
    assert "Screens: our Reflect." in dec.text


def test_hazard_footer_tracks_set_flip_and_clear():
    d = Director()
    d.observe([Event("hazard_set", "their Ting-Lu set Stealth Rock on our "
                     "side", side="us",
                     data={"condition": "Stealth Rock", "user": "Ting-Lu"})])
    dec = d.decide(_ctx(turn=3))
    assert "Hazards: our side Stealth Rock." in dec.text
    d.observe([Event("hazard_flip", "Court Change swapped the hazards and "
                     "screens onto the opposite sides", notable=True)])
    dec2 = d.decide(_ctx(turn=5))
    assert "Hazards: their side Stealth Rock." in dec2.text
    d.observe([Event("hazard_cleared", "their Great Tusk cleared Stealth "
                     "Rock from their side with Rapid Spin", side="them",
                     notable=True,
                     data={"condition": "Stealth Rock", "by": "Rapid Spin"})])
    dec3 = d.decide(_ctx(turn=7))
    assert "Hazards:" not in dec3.text


def test_court_change_swaps_scanner_hazard_ledger():
    """The 'had no hazards to clear' read runs off _hazards_up; before this
    fix a LANDED Court Change left the ledger on stale sides for the rest
    of the game."""
    sc = ProtocolScanner()
    sc.scan([
        ["", "switch", "p1a: Cinderace", "Cinderace, M", "100/100"],
        ["", "switch", "p2a: Ting-Lu", "Ting-Lu", "100/100"],
        ["", "move", "p2a: Ting-Lu", "Stealth Rock", "p1a: Cinderace"],
        ["", "-sidestart", "p1: someone", "move: Stealth Rock"],
    ], role="p2")
    assert sc._hazards_up["p1"] == {"stealth rock"}
    sc.scan([
        ["", "move", "p1a: Cinderace", "Court Change", "p2a: Ting-Lu"],
        ["", "-swapsideconditions"],
    ], role="p2")
    assert sc._hazards_up["p1"] == set()
    assert sc._hazards_up["p2"] == {"stealth rock"}
