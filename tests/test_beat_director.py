"""Beat director tests — driven exactly the way the gold-set eval runner
will drive it: fabricated protocol batches through ProtocolScanner, ctx
through Director.decide, assertions on the emitted Beats. No live battle,
no AIRI, no wall clock."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.beat_director import (Director, ProtocolScanner, TurnContext,
                                    classify, Event)


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
