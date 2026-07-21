"""Beat director tests — driven exactly the way the gold-set eval runner
will drive it: fabricated protocol batches through ProtocolScanner, ctx
through Director.decide, assertions on the emitted Beats. No live battle,
no AIRI, no wall clock."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.beat_director import (Director, ProtocolScanner, TurnContext,
                                    classify, Event, world_collapse_prose,
                                    endgame_solved_prose, deep_think_prose)


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
    assert "knocked out Darkrai" in kos[0].prose
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
    assert cleared.side == "us" and "our Spikes" in cleared.prose


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
    assert "their Toxapex" not in ko.prose             # Toxapex only theirs -> bare


def test_mirror_residual_faint_disambiguates():
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Gliscor", "Gliscor, M", "100/100"],
        ["", "switch", "p2a: Gliscor", "Gliscor, M", "100/100"],
        ["", "faint", "p2a: Gliscor"],           # residual, no move this turn
    ], role="p2")
    ko = next(e for e in evs if e.type == "ko")
    assert ko.prose == "our Gliscor went down"


def test_non_mirror_prose_byte_unchanged():
    """Different species on each side -> no our/their prefix (the KO/move
    prose must stay byte-identical to before the mirror fix)."""
    sc = ProtocolScanner()
    evs = sc.scan([
        ["", "switch", "p1a: Great Tusk", "Great Tusk", "100/100"],
        ["", "switch", "p2a: Gholdengo", "Gholdengo", "100/100"],
        ["", "move", "p1a: Great Tusk", "Earthquake", "p2a: Gholdengo"],
        ["", "-damage", "p2a: Gholdengo", "0 fnt"],
        ["", "faint", "p2a: Gholdengo"],
    ], role="p2")
    ko = next(e for e in evs if e.type == "ko")
    assert ko.prose == "Great Tusk's Earthquake knocked out Gholdengo"
    assert "our" not in ko.prose and "their" not in ko.prose


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


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        fn()
        print(f"ok {name}")
    print(f"\n{len(fns)} tests passed")
