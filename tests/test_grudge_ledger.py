"""Grudge ledger: replay parsing (species-resolved), aggregation,
threshold, and caster injection into FRACTURE's prompt only."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.grudge_ledger import (GrudgeLedger, merge_faints,
                                     parse_replay_games)
from showdown.caster import Caster


def _game(our_role, faints):
    return {"our_role": our_role, "faints": faints, "winner": "x"}


def test_merge_counts_only_our_deaths_by_killer_species():
    led = {}
    merge_faints(led, [
        _game("p1", [("p1", "Darkrai", 5, ("Gholdengo", "Shadow Ball")),
                     ("p1", "Enamorus", 9, ("Gholdengo", "Make It Rain")),
                     ("p2", "Kingambit", 7, ("Zamazenta", "Body Press"))]),
        _game("p1", [("p1", "Araquanid", 4, ("Gholdengo", "Shadow Ball"))]),
    ])
    # Gholdengo killed 3 of ours; the enemy's own faint (p2) is not a grudge
    g = led["gholdengo"]
    assert g["kos"] == 3
    assert g["victims"] == {"Darkrai": 1, "Enamorus": 1, "Araquanid": 1}
    assert g["moves"]["Shadow Ball"] == 2
    assert "zamazenta" not in led   # our own killer is not grudged


def test_grudge_for_threshold_and_format():
    led = GrudgeLedger({
        "gholdengo": {"name": "Gholdengo", "kos": 3,
                      "victims": {"Darkrai": 2, "Enamorus": 1},
                      "moves": {"Shadow Ball": 3}},
        "clefable": {"name": "Clefable", "kos": 1,
                     "victims": {"Darkrai": 1}, "moves": {}},
    })
    line = led.grudge_for("Gholdengo")
    assert line and "Gholdengo" in line and "3 times" in line
    assert "Darkrai (2x)" in line and "Shadow Ball" in line
    # under the min_kos threshold -> no grudge (one unlucky KO isn't lore)
    assert led.grudge_for("Clefable") is None
    assert led.grudge_for("Skarmory") is None      # unknown
    assert led.grudge_for(None) is None


def test_species_resolution_from_replay_log():
    """Nicknamed mons resolve to species; the grudge is against the Pokemon,
    not the pet name."""
    log = "\n".join([
        "|player|p1|Us|avatar|1500",
        "|player|p2|Them|avatar|1500",
        "|switch|p1a: Fluffy (Darkrai)|Darkrai, M|100/100",
        "|switch|p2a: Goldie (Gholdengo)|Gholdengo|100/100",
        "|move|p2a: Goldie|Shadow Ball|p1a: Fluffy",
        "|faint|p1a: Fluffy",
        "|win|Them",
    ])
    p = Path("/tmp/_grudge_replay.json")
    p.write_text(json.dumps({"log": log}))
    try:
        games = parse_replay_games([p], "Us")
    finally:
        p.unlink()
    assert len(games) == 1
    led = {}
    merge_faints(led, games)
    assert led["gholdengo"]["kos"] == 1              # species, not "Goldie"
    assert led["gholdengo"]["victims"] == {"Darkrai": 1}


def test_caster_injects_grudge_for_fracture_only():
    c = Caster("http://unused", "test-model")
    c.grudges = GrudgeLedger({
        "gholdengo": {"name": "Gholdengo", "kos": 4,
                      "victims": {"Darkrai": 4}, "moves": {}}})
    item = {"text": "[BATTLE T5] Gholdengo switches in.",
            "beats": [], "hud": {"turn": 5, "them": "Gholdengo"}}
    frac = c._prompt("FRACTURE", item)[1]["content"]
    prism = c._prompt("PRISM", item)[1]["content"]
    assert "GRUDGE LEDGER" in frac and "Gholdengo" in frac
    assert "GRUDGE LEDGER" not in prism      # analyst never gets the ledger
    # no grudge for a clean mon -> no injection
    item2 = {"text": "[BATTLE T6] x", "beats": [],
             "hud": {"turn": 6, "them": "Skarmory"}}
    assert "GRUDGE LEDGER" not in c._prompt("FRACTURE", item2)[1]["content"]


def test_missing_ledger_is_graceful():
    assert GrudgeLedger.load(None).ledger == {}
    assert GrudgeLedger.load("/nonexistent/path.json").ledger == {}
    c = Caster("http://unused", "test-model", grudge_path=None)
    p = c._prompt("FRACTURE", {"text": "[BATTLE T1] x", "beats": [],
                               "hud": {"them": "Gholdengo"}})
    assert "GRUDGE LEDGER" not in p[1]["content"]


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        fn()
        print(f"ok {name}")
    print(f"\n{len(fns)} tests passed")
