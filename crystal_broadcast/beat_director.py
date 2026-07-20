# Beat director for the commentary broadcast (PRISM + FRACTURE duo).
#
# Architecture: raw Showdown protocol batches -> ProtocolScanner -> typed
# Events -> Director -> Decision (beats + composed beat text). Everything in
# this module is pure logic with no I/O and no wall clock: the live player
# (gen9_player --airi) is one driver, and the commentary gold-set eval
# runner replaying logged protocol is another. Keep it that way — the gold
# set (~/Documents/commentator-project/latest/gold-set-draft.yaml) tests
# DIRECTOR behavior (beat detection, persona routing, priority, registers,
# silence), which is only checkable if this module runs offline.
#
# Persona routing note: beats carry persona/priority/register from the
# taxonomy, but the current delivery layer is a single AIRI character —
# the live player composes one aggregated beat text per decision exactly
# as before the refactor, and the persona metadata waits for the duo
# plumbing. Registers are director-internal hints, never spoken text.

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- beat taxonomy: beat -> (default persona, default priority) -----------
# From commentary-gold-set-template.md plus the draft's additions (status,
# status_recovery, hazards, tera, item_denial, lockdown, endgame,
# wager_segment). Personas: analyst | gremlin | either | both | none.
TAXONOMY = {
    "ko": ("gremlin", "interrupt"),
    "set_reveal": ("analyst", "interrupt"),
    "desk_swing": ("both", "interrupt"),
    "desk_contradiction": ("analyst", "interrupt"),
    "win_con": ("analyst", "normal"),
    "stat_cite": ("analyst", "normal"),
    "crit_luck": ("gremlin", "interrupt"),
    "preview": ("analyst", "normal"),
    "filler": ("either", "filler"),
    "recap": ("both", "normal"),
    "silence": ("none", "silence"),
    "refusal": ("either", "normal"),
    "status": ("gremlin", "interrupt"),
    "status_recovery": ("gremlin", "interrupt"),
    "hazards": ("analyst", "normal"),
    "field_state": ("either", "normal"),
    "tera": ("analyst", "interrupt"),
    "item_denial": ("either", "interrupt"),
    "lockdown": ("either", "interrupt"),
    "endgame": ("analyst", "interrupt"),
    "wager_segment": ("both", "normal"),
}

_PRIORITY_RANK = {"interrupt": 3, "normal": 2, "filler": 1, "silence": 0}


@dataclass
class Event:
    """One detected battle happening. `prose` is the display line that rides
    into the beat text (byte-compatible with the pre-director transcript);
    `type` + `side` + `data` are the machine-readable layer the director
    and the gold-set runner match on. side: 'us' | 'them' | None."""
    type: str
    prose: str = ""
    side: str | None = None
    notable: bool = False
    data: dict = field(default_factory=dict)


@dataclass
class Beat:
    """A classified commentary moment. persona/priority default from the
    taxonomy; register is a delivery hint for the owning persona (e.g. the
    gremlin's 'despair' vs 'celebration' on the same status event)."""
    beat: str
    persona: str
    priority: str
    prose: str = ""
    register: str | None = None
    handoff: list[str] | None = None
    data: dict = field(default_factory=dict)


@dataclass
class TurnContext:
    """Decision-time inputs, all primitives so tests can fabricate them.
    Display names throughout (the raw-id -> hallucination lesson)."""
    turn: int
    value: float
    elapsed: float
    me_name: str | None = None
    me_hp: int | None = None
    me_status: str | None = None
    opp_name: str | None = None
    opp_hp: int | None = None
    opp_status: str | None = None
    ours_fainted: frozenset = frozenset()
    theirs_fainted: frozenset = frozenset()
    choice_text: str = ""


@dataclass
class Decision:
    """Director output for one decision point. text=None means silence at
    the delivery layer; beats are still reported for eval/telemetry."""
    text: str | None
    beats: list
    silence: bool


def make_beat(beat_type: str, prose: str = "", register: str | None = None,
              persona: str | None = None, priority: str | None = None,
              handoff: list[str] | None = None, **data) -> Beat:
    d_persona, d_priority = TAXONOMY[beat_type]
    return Beat(beat=beat_type, persona=persona or d_persona,
                priority=priority or d_priority, prose=prose,
                register=register, handoff=handoff, data=data)


# --- protocol-line helpers (moved verbatim from gen9_player) ---------------

def _poke_name(token: str) -> str:
    """'p2a: Dragonite' -> 'Dragonite'."""
    return token.split(": ", 1)[1] if ": " in token else token


def _cond_name(raw: str) -> str:
    """'move: Stealth Rock' -> 'Stealth Rock'."""
    return raw.split(": ", 1)[1] if ": " in raw else raw


def _from_move(events) -> str | None:
    """Pull the '[from] move: X' cause out of a protocol line's trailing args."""
    for e in events:
        if e.startswith("[from] move:"):
            return e.split(":", 1)[1].strip()
    return None


def _from_cause(events) -> str | None:
    """Like _from_move but also catches '[from] ability: X' (Magician,
    Pickpocket) — used for item changes that a move OR ability can drive."""
    for e in events:
        if e.startswith("[from] move:") or e.startswith("[from] ability:"):
            return e.split(":", 1)[1].strip()
    return None


def _hp_frac(hp: str) -> float | None:
    """'45/100' / '0 fnt' / '45/100 brn' -> fraction, or None."""
    try:
        head = hp.strip().split(" ")[0]
        if head in ("0", "0.0"):
            return 0.0
        num, den = head.split("/")
        den_v = float(den)
        return float(num) / den_v if den_v else None
    except Exception:
        return None


def _join_phrases(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _read_phrase(value: float) -> str:
    """Qualitative position read for commentary. The raw win estimate never
    reaches the character: fed a number, the LLM recites it like a
    scoreboard and tunes out the actual board (measured: Prism quoting
    percentages over calling faints)."""
    if value >= 0.85:
        return "this one looks all but sealed for us"
    if value >= 0.70:
        return "we're clearly ahead"
    if value >= 0.58:
        return "we hold a real edge"
    if value >= 0.45:
        return "it's dead even right now"
    if value >= 0.32:
        return "we're behind in this"
    if value >= 0.15:
        return "we're in deep trouble"
    return "this is nearly gone"


def _swing_phrase(swing: float | None) -> str | None:
    if swing is None:
        return None
    if swing >= 0.10:
        return "momentum just swung our way"
    if swing >= 0.03:
        return "momentum drifting our way"
    if swing <= -0.10:
        return "momentum just swung hard against us"
    if swing <= -0.03:
        return "momentum slipping away from us"
    return "holding steady"


_STATUS_INFLICT = {
    "frz": "froze {n} solid", "brn": "burned {n}", "par": "paralyzed {n}",
    "slp": "put {n} to sleep", "psn": "poisoned {n}",
    "tox": "badly poisoned {n}",
}
_STATUS_CURE = {
    "frz": "{n} thawed out", "slp": "{n} woke up",
    "par": "{n} shook off the paralysis", "brn": "{n}'s burn healed",
    "psn": "{n} was cured of poison", "tox": "{n} was cured of poison",
}
_CANT = {
    "frz": "{n} was frozen solid and couldn't move",
    "par": "{n} was fully paralyzed and couldn't move",
    "slp": "{n} was fast asleep", "flinch": "{n} flinched",
    "recharge": "{n} had to recharge",
}

_HAZARDS = {"stealth rock", "spikes", "toxic spikes", "sticky web",
            "g-max steelsurge"}
_SCREENS = {"reflect", "light screen", "aurora veil"}
_WEATHER = {
    "raindance": "rain", "sunnyday": "harsh sun", "sandstorm": "a sandstorm",
    "snow": "snow", "hail": "hail", "snowscape": "snow",
    "desolateland": "extreme sun", "primordialsea": "heavy rain",
    "deltastream": "strong winds",
}

_STAT = {"atk": "Attack", "def": "Defense", "spa": "Special Attack",
         "spd": "Special Defense", "spe": "Speed", "accuracy": "accuracy",
         "evasion": "evasiveness"}
# volatile -> (phrase template, notable). Momentum-shutting ones (Encore,
# Taunt) force a beat; routine ones ride along in the next beat instead.
_VOL_START = {
    "substitute": ("{n} put up a Substitute", False),
    "leech seed": ("{n} was seeded", False),
    "confusion": ("{n} became confused", False),
    "encore": ("{n} was locked in by Encore", True),
    "taunt": ("{n} was shut down by Taunt", True),
    "yawn": ("{n} is growing drowsy", False),
    "disable": ("{n} had a move disabled", False),
    "attract": ("{n} became infatuated", False),
}
_VOL_END = {
    "substitute": ("{n}'s Substitute broke", True),
}


class ProtocolScanner:
    """Walk battle message batches and emit typed Events. Prose lines are
    byte-identical to the pre-director scanner so transcripts, the overlay
    parser, and recorded-demo comparisons stay stable. Holds only per-battle
    perception state (HP fractions for hit sizing, current weather)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._hp: dict = {}
        self._weather: str | None = None
        self._last_move: tuple[str, str] | None = None
        # position -> species, from switch/drag/replace details: protocol
        # position tokens carry NICKNAMES ("p1a: Speak Softly"), and prose
        # built from them leaks the nickname ("knocked off Speak Softly" —
        # found scouting real replays; ladder opponents nickname freely)
        self._species: dict[str, str] = {}

    def scan(self, messages, role=None) -> list[Event]:
        out: list[Event] = []
        cur = None

        def name_of(token) -> str:
            """Position token -> species display name (nickname-proof)."""
            pos = token.split(":")[0]
            return self._species.get(pos, _poke_name(token))

        def side_of(token) -> str | None:
            sr = token[:2]
            if role and sr == role:
                return "us"
            if role and sr != role:
                return "them"
            return None

        def side_poss(side_token):
            s = side_of(side_token)
            return {"us": "our", "them": "their"}.get(s, "one")

        def flush():
            nonlocal cur
            if cur and cur.get("move"):
                # survives the flush so effects whose protocol line follows
                # the move (Court Change's -swapsideconditions) can name
                # their user
                self._last_move = (cur["mover"], cur["move"])
            if not cur or not cur.get("move"):
                cur = None
                return
            head = f"{cur['mover']}'s {cur['move']}"
            mover_side = cur.get("mover_side")
            target_side = ({"us": "them", "them": "us"}.get(mover_side)
                           if mover_side else None)
            if cur.get("missed"):
                out.append(Event("move_missed", f"{head} missed",
                                 side=mover_side,
                                 data={"mover": cur["mover"],
                                       "move": cur["move"]}))
                cur = None
                return
            if cur.get("effect") == "no effect":
                out.append(Event(
                    "move_no_effect",
                    f"{head} had no effect on {cur['target']}",
                    side=mover_side, notable=True,
                    data={"mover": cur["mover"], "move": cur["move"],
                          "target": cur["target"]}))
                cur = None
                return
            tags = []
            if cur.get("crit"):
                tags.append("a critical hit")
            if cur.get("effect"):
                tags.append(cur["effect"])
            dmg = cur.get("dmg")
            if dmg is not None and not cur.get("ko"):
                if dmg >= 0.5:
                    tags.append("a devastating blow")
                elif dmg >= 0.33:
                    tags.append("a heavy hit")
                elif 0 < dmg <= 0.08:
                    tags.append("barely a scratch")
            if cur.get("ko"):
                # the finishing blow, attributed to the move that landed it
                line = f"{head} knocked out {cur['target']}"
                if tags:
                    line += " with " + _join_phrases(tags)
                out.append(Event("ko", line, side=target_side, notable=True,
                                 data={"mover": cur["mover"],
                                       "move": cur["move"],
                                       "target": cur["target"],
                                       "crit": cur.get("crit", False)}))
            elif tags:
                notable = (cur.get("crit")
                           or cur.get("effect") == "super effective"
                           or (dmg is not None and dmg >= 0.5))
                out.append(Event(
                    "move_hit", f"{head} landed {_join_phrases(tags)}",
                    side=target_side, notable=bool(notable),
                    data={"mover": cur["mover"], "move": cur["move"],
                          "target": cur["target"],
                          "crit": cur.get("crit", False), "dmg": dmg}))
            cur = None

        for sm in messages:
            if len(sm) < 2:
                continue
            t = sm[1]
            if t == "move":
                flush()
                cur = {"mover": name_of(sm[2]), "move": sm[3],
                       "mover_side": side_of(sm[2]),
                       "target": name_of(sm[4]) if len(sm) > 4 else None,
                       "effect": None, "crit": False, "dmg": None,
                       "missed": False}
            elif t == "-crit" and cur:
                cur["crit"] = True
            elif t == "-supereffective" and cur:
                cur["effect"] = "super effective"
            elif t == "-resisted" and cur:
                cur["effect"] = "not very effective"
            elif t == "-immune":
                if cur:
                    cur["effect"] = "no effect"
            elif t == "-miss" and cur:
                cur["missed"] = True
            elif t in ("-damage", "-heal", "-sethp"):
                key = sm[2].split(":")[0]
                frac = _hp_frac(sm[3]) if len(sm) > 3 else None
                old = self._hp.get(key)
                if frac is not None:
                    self._hp[key] = frac
                if (t == "-damage" and cur and old is not None
                        and frac is not None
                        and cur.get("target") == name_of(sm[2])):
                    cur["dmg"] = old - frac
            elif t in ("switch", "drag"):
                key = sm[2].split(":")[0]
                self._hp[key] = (_hp_frac(sm[4]) if len(sm) > 4 else 1.0)
                if len(sm) > 3 and sm[3]:
                    self._species[key] = sm[3].split(",")[0]
            elif t == "faint" and len(sm) > 2:
                mon = name_of(sm[2])
                if cur and cur.get("target") == mon:
                    cur["ko"] = True  # attribute to the finishing move
                else:
                    flush()  # residual: poison/hazard/recoil/Life Orb etc.
                    out.append(Event("ko", f"{mon} went down",
                                     side=side_of(sm[2]), notable=True,
                                     data={"target": mon, "residual": True}))
            elif t == "-status" and len(sm) > 3:
                tmpl = _STATUS_INFLICT.get(sm[3])
                if tmpl:
                    flush()  # emit the causing move first, then its effect
                    cause = _from_move(sm[4:])
                    prose = tmpl.format(n=name_of(sm[2]))
                    if cause:
                        # name the cause or downstream commentary invents
                        # one (a caster said "Spore" on a beat that only
                        # read "put Gliscor to sleep")
                        prose = f"{cause} {prose}"
                    out.append(Event(
                        "status_applied", prose,
                        side=side_of(sm[2]), notable=True,
                        data={"mon": name_of(sm[2]), "status": sm[3],
                              "cause": cause}))
            elif t == "-curestatus" and len(sm) > 3:
                tmpl = _STATUS_CURE.get(sm[3])
                if tmpl:
                    flush()
                    out.append(Event(
                        "status_cured", tmpl.format(n=name_of(sm[2])),
                        side=side_of(sm[2]), notable=True,
                        data={"mon": name_of(sm[2]), "status": sm[3]}))
            elif t == "cant" and len(sm) > 3:
                tmpl = _CANT.get(sm[3])
                if tmpl:
                    flush()
                    out.append(Event(
                        "cant_move", tmpl.format(n=name_of(sm[2])),
                        side=side_of(sm[2]), notable=True,
                        data={"mon": name_of(sm[2]), "why": sm[3]}))
            elif t == "-enditem" and len(sm) > 3:
                flush()
                mon = name_of(sm[2])
                item = sm[3]
                by = _from_cause(sm[4:])
                ate = any("[eat]" in a for a in sm[4:])
                mside = side_of(sm[2])
                if by == "Knock Off":
                    out.append(Event(
                        "item_knocked_off", f"{item} was knocked off {mon}",
                        side=mside, notable=True,
                        data={"mon": mon, "item": item}))
                elif by in ("Thief", "Covet", "Magician", "Pickpocket"):
                    out.append(Event(
                        "item_stolen", f"{mon}'s {item} was swiped away",
                        side=mside, notable=True,
                        data={"mon": mon, "item": item}))
                elif item == "Focus Sash":
                    out.append(Event(
                        "sash_saved", f"{mon}'s Focus Sash let it cling on",
                        side=mside, notable=True,
                        data={"mon": mon, "item": item}))
                elif item == "Air Balloon":
                    out.append(Event(
                        "balloon_popped", f"{mon}'s Air Balloon popped",
                        side=mside, notable=True,
                        data={"mon": mon, "item": item}))
                elif ate:
                    # a berry eaten is routine tempo, not a forced beat
                    out.append(Event("item_eaten", f"{mon} ate its {item}",
                                     side=mside,
                                     data={"mon": mon, "item": item}))
                else:
                    out.append(Event("item_used", f"{mon} used up its {item}",
                                     side=mside,
                                     data={"mon": mon, "item": item}))
            elif t == "-item" and len(sm) > 3:
                by = _from_cause(sm[4:])
                if by in ("Trick", "Switcheroo"):
                    flush()
                    out.append(Event(
                        "item_tricked",
                        f"{name_of(sm[2])} was handed a {sm[3]} by {by}",
                        side=side_of(sm[2]), notable=True,
                        data={"mon": name_of(sm[2]), "item": sm[3]}))
                elif by in ("Thief", "Covet", "Magician", "Pickpocket"):
                    flush()
                    out.append(Event(
                        "item_stolen",
                        f"{name_of(sm[2])} swiped a {sm[3]} with {by}",
                        side=side_of(sm[2]), notable=True,
                        data={"mon": name_of(sm[2]), "item": sm[3]}))
                # plain reveals (switch-in, Frisk) are not dramatic: skip
            elif t == "-terastallize" and len(sm) > 3:
                flush()
                out.append(Event(
                    "tera",
                    f"{name_of(sm[2])} Terastallized into a {sm[3]} type",
                    side=side_of(sm[2]), notable=True,
                    data={"mon": name_of(sm[2]), "tera_type": sm[3]}))
            elif t == "-boost" and len(sm) > 4:
                flush()
                stat = _STAT.get(sm[3], sm[3])
                amt = int(sm[4]) if sm[4].lstrip("-").isdigit() else 1
                adv = "sharply " if amt >= 2 else ""
                # offensive setup (atk/spa/spe) threatens a sweep -> force a
                # beat; a defensive/minor +1 just rides along
                notable = amt >= 2 or sm[3] in ("atk", "spa", "spe")
                out.append(Event(
                    "boost", f"{name_of(sm[2])} {adv}raised its {stat}",
                    side=side_of(sm[2]), notable=notable,
                    data={"mon": name_of(sm[2]), "stat": sm[3],
                          "amount": amt}))
            elif t == "-unboost" and len(sm) > 4:
                flush()
                stat = _STAT.get(sm[3], sm[3])
                amt = int(sm[4]) if sm[4].lstrip("-").isdigit() else 1
                adv = "sharply " if amt >= 2 else ""
                out.append(Event(
                    "unboost", f"{name_of(sm[2])}'s {stat} was {adv}cut",
                    side=side_of(sm[2]),
                    data={"mon": name_of(sm[2]), "stat": sm[3],
                          "amount": amt}))
            elif t == "-setboost" and len(sm) > 4:
                flush()
                out.append(Event(
                    "boost",
                    f"{name_of(sm[2])} maxed out its "
                    f"{_STAT.get(sm[3], sm[3])}",
                    side=side_of(sm[2]), notable=True,
                    data={"mon": name_of(sm[2]), "stat": sm[3],
                          "maxed": True}))
            elif t in ("-clearallboost", "-invertboost", "-clearboost"):
                flush()
                if t == "-clearallboost":
                    prose = "every stat change was wiped away"
                elif t == "-invertboost":
                    prose = (f"{name_of(sm[2])}'s stat changes were inverted"
                             if len(sm) > 2 else
                             "the stat changes were inverted")
                else:
                    prose = (f"{name_of(sm[2])}'s boosts were cleared"
                             if len(sm) > 2 else "the boosts were cleared")
                out.append(Event("boosts_cleared", prose, notable=True,
                                 side=side_of(sm[2]) if len(sm) > 2 else None))
            elif t == "-start" and len(sm) > 3:
                key = _cond_name(sm[3]).lower()
                entry = _VOL_START.get(key)
                if entry:
                    flush()
                    out.append(Event(
                        "volatile_start",
                        entry[0].format(n=name_of(sm[2])),
                        side=side_of(sm[2]), notable=entry[1],
                        data={"mon": name_of(sm[2]), "volatile": key}))
            elif t == "-end" and len(sm) > 3:
                entry = _VOL_END.get(_cond_name(sm[3]).lower())
                if entry:
                    flush()
                    out.append(Event(
                        "volatile_end",
                        entry[0].format(n=name_of(sm[2])),
                        side=side_of(sm[2]), notable=entry[1],
                        data={"mon": name_of(sm[2]),
                              "volatile": _cond_name(sm[3]).lower()}))
            elif t == "replace" and len(sm) > 2:
                flush()
                species = sm[3].split(",")[0] if len(sm) > 3 else name_of(sm[2])
                self._species[sm[2].split(":")[0]] = species
                out.append(Event(
                    "illusion_reveal",
                    f"the Illusion drops - it was {species} all along",
                    side=side_of(sm[2]), notable=True,
                    data={"mon": species}))
            elif t == "-transform" and len(sm) > 3:
                flush()
                out.append(Event(
                    "transform",
                    f"{name_of(sm[2])} transformed into {name_of(sm[3])}",
                    side=side_of(sm[2]), notable=True,
                    data={"mon": name_of(sm[2]),
                          "into": name_of(sm[3])}))
            elif t == "-prepare" and len(sm) > 3:
                flush()
                out.append(Event(
                    "charging", f"{name_of(sm[2])} is charging up {sm[3]}",
                    side=side_of(sm[2]),
                    data={"mon": name_of(sm[2]), "move": sm[3]}))
            elif t == "-sidestart" and len(sm) > 3:
                flush()
                poss = side_poss(sm[2])
                cond = _cond_name(sm[3])
                low = cond.lower()
                # setting hazards/screens is routine tempo — record it, but
                # don't force a beat (removing them below IS a swing)
                if low in _HAZARDS:
                    out.append(Event(
                        "hazard_set", f"{cond} went up on {poss} side",
                        side=side_of(sm[2]),
                        data={"condition": cond}))
                elif low in _SCREENS:
                    out.append(Event(
                        "screens_set", f"{cond} went up on {poss} side",
                        side=side_of(sm[2]),
                        data={"condition": cond}))
                elif low == "tailwind":
                    out.append(Event(
                        "tailwind_up", f"Tailwind kicked in for {poss} side",
                        side=side_of(sm[2]),
                        data={"condition": cond}))
            elif t == "-sideend" and len(sm) > 3:
                flush()
                poss = side_poss(sm[2])
                cond = _cond_name(sm[3])
                low = cond.lower()
                by = _from_move(sm[4:])
                if low in _HAZARDS or low in _SCREENS or low == "tailwind":
                    if by:
                        etype = ("hazard_cleared" if low in _HAZARDS
                                 else "side_cleared")
                        out.append(Event(
                            etype, f"{poss} {cond} was cleared away by {by}",
                            side=side_of(sm[2]), notable=True,
                            data={"condition": cond, "by": by}))
                    elif low in _SCREENS:
                        out.append(Event(
                            "screens_wore_off", f"{poss} {cond} wore off",
                            side=side_of(sm[2]),
                            data={"condition": cond}))
            elif t == "-swapsideconditions":
                # Court Change: hazards/screens change sides in one move —
                # the same class of swing as a Rapid Spin/Defog clear, but
                # the protocol emits this dedicated message instead of
                # -sideend lines, so it was invisible to the scan. Name the
                # user (from the move line just flushed) or the casters
                # invent an actor — measured, twice
                flush()
                user = (self._last_move[0]
                        if self._last_move
                        and self._last_move[1] == "Court Change" else None)
                who = f"{user}'s Court Change" if user else "Court Change"
                out.append(Event(
                    "hazard_flip",
                    f"{who} swapped the hazards and screens "
                    "onto the opposite sides", notable=True,
                    data={"user": user} if user else {}))
            elif t == "-weather" and len(sm) > 2:
                w = sm[2]
                upkeep = any("[upkeep]" in a for a in sm[3:])
                if not upkeep:
                    flush()
                    if w == "none":
                        if self._weather:
                            out.append(Event("weather_cleared",
                                             "the weather cleared",
                                             notable=True))
                        self._weather = None
                    else:
                        label = _WEATHER.get(w.lower().replace(" ", ""), w)
                        if label != self._weather:
                            out.append(Event(
                                "weather_set", f"{label} set in",
                                notable=True, data={"weather": label}))
                            self._weather = label
            elif t == "-fieldstart" and len(sm) > 2:
                cond = _cond_name(sm[2])
                flush()
                out.append(Event("field_start", f"{cond} took over the field",
                                 notable=True, data={"condition": cond}))
            elif t == "-fieldend" and len(sm) > 2:
                cond = _cond_name(sm[2])
                if cond.lower() == "trick room":
                    flush()
                    out.append(Event("field_end", "Trick Room wore off",
                                     data={"condition": cond}))
        flush()
        return out


# --- event -> beat classification -----------------------------------------

# gremlin registers by allegiance for the same event class — the docs'
# "allegiance determines the read" rule (gc-0003/0004 and kin)
_LUCK_REGISTERS = {"us": "persecution", "them": "delight"}


def classify(ev: Event, stats_fn=None) -> Beat | None:
    """Map one Event to a Beat, or None for pure color (rides along in the
    beat text without owning a moment). stats_fn(species_display) ->
    (atk, spa) or None enables the burn physical-vs-special split."""
    t = ev.type
    if t == "belief_delta":
        # set-inference confirmation ("that's a Scarf") — not a protocol
        # event; the live player injects it when the search adopts a new
        # inferred item. Speed/damage items (gc-0018/19) are either voice
        # (analyst cites the chain, gremlin claims the call); the Boots
        # negative-evidence read (gc-0020, the dog that didn't bark) is
        # analyst-only — the gremlin has nothing to shout about an absence.
        boots = ev.data.get("item") == "heavydutyboots"
        return make_beat("set_reveal", ev.prose,
                         persona="analyst" if boots else "either",
                         priority="interrupt", register="set-reveal",
                         **ev.data)
    if t == "ko":
        return make_beat("ko", ev.prose,
                         register="grief" if ev.side == "us" else "triumph",
                         **ev.data)
    if t in ("move_missed", "cant_move"):
        reg = _LUCK_REGISTERS.get(
            "us" if (t == "cant_move" and ev.side == "us")
            or (t == "move_missed" and ev.side == "us") else "them")
        # a miss hurts the mover; cant hurts the afflicted — both are luck
        # events for whoever suffered them
        return make_beat("crit_luck", ev.prose, register=reg, **ev.data)
    if t == "move_hit" and ev.data.get("crit"):
        # crit against us = persecution; ours = shameless delight (gc-0021/22)
        reg = "persecution" if ev.side == "us" else "delight"
        return make_beat("crit_luck", ev.prose, register=reg, **ev.data)
    if t == "status_applied":
        status = ev.data.get("status")
        cause = (ev.data.get("cause") or "")
        if status == "slp" and cause in ("Yawn", "Rest"):
            # deliberate sleep — Yawn's negotiated stay-in, or Rest buying
            # recovery with tempo (real replays showed Rest routing to the
            # gremlin's assassination register: wrong frame, it's a choice).
            # Analyst owns it; a gremlin shock-react here is a gold FAILURE
            return make_beat("status", ev.prose, persona="analyst",
                             priority="normal", register="negotiated",
                             **ev.data)
        if status == "frz":
            reg = "persecution" if ev.side == "us" else "rejoicing"
            return make_beat("status", ev.prose, register=reg, **ev.data)
        if status == "brn" and stats_fn is not None:
            mon = ev.data.get("mon")
            stats = stats_fn(mon) if mon else None
            if stats is not None:
                atk, spa = stats
                if spa > atk:
                    # burn on a special attacker: the headline effect does
                    # nothing — analyst critique, not gremlin rage (gc-0005)
                    return make_beat("status", ev.prose, persona="analyst",
                                     priority="normal",
                                     register="wasted-burn", **ev.data)
        reg = "despair" if ev.side == "us" else "celebration"
        return make_beat("status", ev.prose, register=reg, **ev.data)
    if t == "status_cured":
        reg = "bragging" if ev.side == "us" else "rigged"
        return make_beat("status_recovery", ev.prose, register=reg, **ev.data)
    if t == "hazard_set":
        return make_beat("hazards", ev.prose, **ev.data)
    if t == "hazard_cleared":
        # our stack swept = sunk-cost outrage; theirs = housekeeping (gc-0016)
        reg = "sunk-cost-outrage" if ev.side == "us" else "housekeeping"
        persona = "gremlin" if ev.side == "us" else "analyst"
        return make_beat("hazards", ev.prose, persona=persona,
                         priority="interrupt", register=reg, **ev.data)
    if t == "hazard_flip":
        # Court Change: dual beat, gremlin heist-scream first, analyst
        # re-derivation after (gc-0017)
        return make_beat("hazards", ev.prose, persona="both",
                         priority="interrupt", register="heist",
                         handoff=["gremlin", "analyst"], **ev.data)
    if t in ("screens_set", "screens_wore_off", "side_cleared",
             "weather_set", "weather_cleared", "field_start", "field_end",
             "tailwind_up"):
        return make_beat("field_state", ev.prose, **ev.data)
    if t == "tera":
        return make_beat("tera", ev.prose, **ev.data)
    if t in ("item_knocked_off", "item_stolen", "item_tricked"):
        return make_beat("item_denial", ev.prose, **ev.data)
    if t == "sash_saved":
        return make_beat("set_reveal", ev.prose, persona="either", **ev.data)
    if t == "volatile_start" and ev.data.get("volatile") in ("taunt",
                                                            "encore"):
        return make_beat("lockdown", ev.prose, **ev.data)
    # everything else is color: boosts, routine items, transforms, charging
    return None


class Director:
    """Consumes Events + decision-time context, decides what (if anything)
    the broadcast says. Owns all beat state: the pending-event buffer, sent
    read/disagreement/faint tracking for once-per-onset rules, and the
    ongoing-affliction counters for escalating callbacks (gc-0014).

    Pure logic: no I/O, no wall clock (elapsed arrives in the ctx), no AIRI.
    The live player adapts battle objects into ctx; the eval runner will
    fabricate ctx from replays."""

    def __init__(self, min_interval: float = 20.0, min_swing: float = 0.10,
                 floor: float = 5.0, stats_fn=None):
        self.min_interval = min_interval
        self.min_swing = min_swing
        self.floor = floor
        self.stats_fn = stats_fn
        self.reset()

    def reset(self):
        self._pending: list[Event] = []
        self._notable = False
        self._prev_value: float | None = None
        self._prev_read: str | None = None
        self._prev_disagree: str | None = None
        self._prev_fainted: tuple[frozenset, frozenset] = (frozenset(),
                                                           frozenset())
        # (side, mon) -> consecutive decision points spent asleep/frozen
        self._afflicted: dict = {}

    # --- ingestion -----------------------------------------------------
    def observe(self, events: list[Event]):
        for ev in events:
            self._pending.append(ev)
            if ev.notable:
                self._notable = True
        # keep the buffer bounded; the freshest beats matter most
        if len(self._pending) > 6:
            self._pending = self._pending[-6:]

    def note(self, prose: str, side: str | None = None):
        """A driver-side color note (e.g. 'we send Gliscor in' on forced
        switches) that should ride along in the next beat."""
        self.observe([Event("note", prose, side=side)])

    # --- match framing -------------------------------------------------
    def match_start(self, opponent: str, our_team: list[str],
                    their_team: list[str], lead: str | None = None) -> str:
        self.reset()
        text = (f"[MATCH START] New battle vs {opponent or 'the opponent'}. "
                f"Our team: {', '.join(our_team) or 'unknown'}. "
                f"Their preview: {', '.join(their_team) or 'hidden'}.")
        if lead:
            text += f" We lead {lead}."
        text += " Set the stage in a line or two."
        return text

    def match_end(self, result: str, ours_left: int, theirs_left: int,
                  opponent: str) -> tuple[str, Beat]:
        beat = make_beat("recap", result=result,
                         handoff=["gremlin", "analyst"])
        text = (f"[RESULT] {result} vs {opponent or 'the opponent'}. "
                f"Left standing: us {ours_left}, them {theirs_left}. "
                f"Wrap up the match in a line or two.")
        return text, beat

    # --- the per-decision call ------------------------------------------
    def decide(self, ctx: TurnContext) -> Decision:
        swing = (None if self._prev_value is None
                 else ctx.value - self._prev_value)
        new_ours = ctx.ours_fainted - self._prev_fainted[0]
        new_theirs = ctx.theirs_fainted - self._prev_fainted[1]
        faints = bool(new_ours or new_theirs)

        # escalating-affliction bookkeeping (before gating so the counter
        # advances even on silent decisions)
        esc_prose = self._tick_afflictions(ctx)

        if ctx.elapsed < self.floor:
            return Decision(None, [], True)
        if not (self._notable or faints or swing is None
                or abs(swing) >= self.min_swing
                or ctx.elapsed >= self.min_interval):
            return Decision(None, [], True)

        pairs = [(ev, classify(ev, self.stats_fn)) for ev in self._pending]
        beats = [b for _, b in pairs if b is not None]
        if esc_prose:
            beats.append(make_beat("status_recovery", esc_prose,
                                   priority="filler",
                                   register="escalating-grievance"))

        # desk swing / contradiction beats come from decision context, not
        # protocol events
        if swing is not None and abs(swing) >= self.min_swing:
            direction = "our way" if swing > 0 else "against us"
            beats.append(make_beat(
                "desk_swing", f"the desk read just swung {direction}",
                direction="up" if swing > 0 else "down",
                swing=round(swing, 4)))

        # ---- compose the beat text (format unchanged from pre-director:
        # transcripts, the overlay's regex parser, and the recorded demo
        # all read this shape) ----
        parts = [f"[BATTLE T{ctx.turn}]"]
        if self._pending or esc_prose:
            # crowded turns overflow the 4-line exchange window; keep the
            # HIGHEST-priority events' prose (chronological order preserved)
            # rather than the most recent — a blind last-4 dropped a Tera
            # line in favor of "raised its Speed" (caught by replay pinning)
            cand = [(i, ev, b) for i, (ev, b) in enumerate(pairs)
                    if ev.prose]
            if len(cand) > 4:
                ranked = sorted(
                    cand, key=lambda t: (
                        -(_PRIORITY_RANK.get(t[2].priority, 0)
                          if t[2] else 0), -t[0]))
                cand = sorted(ranked[:4], key=lambda t: t[0])
            hl = [ev.prose for _, ev, _ in cand]
            if esc_prose:
                hl.append(esc_prose)
            if hl:
                parts.append("Last exchange: " + "; ".join(hl) + ".")

        # KOs are normally narrated in the highlights above; only fall back
        # to a flat mention for a faint that didn't make the play-by-play
        def _squash(s):
            return re.sub(r"[^a-z0-9]", "", s.lower())

        def _ko_narrated(name):
            n = _squash(name)
            for ev in self._pending:
                hs = _squash(ev.prose)
                if n in hs and ("knockedout" in hs or "wentdown" in hs):
                    return True
            return False

        lost_theirs = [n for n in sorted(new_theirs) if not _ko_narrated(n)]
        lost_ours = [n for n in sorted(new_ours) if not _ko_narrated(n)]
        if lost_theirs:
            parts.append(f"Their {' and '.join(lost_theirs)} "
                         f"{'are' if len(lost_theirs) > 1 else 'is'} down.")
        if lost_ours:
            parts.append(f"We lost {' and '.join(lost_ours)}.")
        me = (f"{ctx.me_name} ({ctx.me_hp}% hp)" if ctx.me_name
              else "Our side")
        opp = (f"{ctx.opp_name} ({ctx.opp_hp}% hp)" if ctx.opp_name
               else "their side")
        parts.append(f"{me} vs {opp}.")
        if ctx.choice_text:
            parts.append(ctx.choice_text)

        # the desk read repeats its band for long stretches, and the
        # character litigates every repetition — only speak it when the
        # band changes or momentum genuinely swings
        read = _read_phrase(ctx.value)
        sw = _swing_phrase(swing)
        swung = sw is not None and "swung" in sw
        if read != self._prev_read or swung:
            parts.append(f"Desk read: {read}{', ' + sw if sw else ''}.")
        parts.append(f"Bodies: us {6 - len(ctx.ours_fainted)} standing, "
                     f"them {6 - len(ctx.theirs_fainted)}.")

        # board-vs-desk disagreement: the flagship beat, said once per
        # onset — phrased as plain feed copy, never a labelled "Note:"
        body_lead = len(ctx.theirs_fainted) - len(ctx.ours_fainted)
        disagree = ("material" if body_lead >= 3 and ctx.value < 0.40 else
                    "bodies" if body_lead <= -3 and ctx.value > 0.60 else None)
        if disagree and disagree != self._prev_disagree:
            if disagree == "material":
                line = ("The board and the desk read sharply disagree "
                        "here: we hold a commanding material lead yet "
                        "the read is grim.")
            else:
                line = ("The board and the desk read sharply disagree "
                        "here: we trail badly on bodies yet the read "
                        "stays upbeat.")
            beats.append(make_beat("desk_contradiction", line,
                                   kind=disagree,
                                   value=round(ctx.value, 4),
                                   body_lead=body_lead))
            parts.append(line)

        # commit state: comparisons are always against the last SENT beat
        self._prev_value = ctx.value
        self._prev_read = read
        self._prev_disagree = disagree
        self._prev_fainted = (ctx.ours_fainted, ctx.theirs_fainted)
        self._pending = []
        self._notable = False

        beats.sort(key=lambda b: -_PRIORITY_RANK.get(b.priority, 0))
        return Decision(" ".join(parts), beats, False)

    # --- ongoing-affliction callbacks (gc-0014) --------------------------
    def _tick_afflictions(self, ctx: TurnContext) -> str | None:
        """Track consecutive decision points the ACTIVE mons spend asleep or
        frozen; from the second one on, surface an escalating callback line.
        Keyed on (side, mon) and reset the moment the affliction clears or
        the mon leaves the field."""
        lines = []
        seen = set()
        for side, name, status in (("us", ctx.me_name, ctx.me_status),
                                   ("them", ctx.opp_name, ctx.opp_status)):
            if not name:
                continue
            key = (side, name)
            if status in ("slp", "frz"):
                seen.add(key)
                self._afflicted[key] = self._afflicted.get(key, 0) + 1
                n = self._afflicted[key]
                if n >= 2:
                    word = "asleep" if status == "slp" else "frozen"
                    whose = "our" if side == "us" else "their"
                    lines.append(f"{whose} {name} is STILL {word} "
                                 f"(turn {n} of it)")
        for key in list(self._afflicted):
            if key not in seen:
                del self._afflicted[key]
        return "; ".join(lines) if lines else None
