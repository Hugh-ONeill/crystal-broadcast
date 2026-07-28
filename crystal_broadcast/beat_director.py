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
# wager_segment) and the engine-signal beats (world_collapse, deep_think;
# endgame doubles as the solver-took-over beat) — these last three are not
# protocol events but internal search telemetry the live player injects,
# the same way it injects belief_delta. Personas: analyst | gremlin |
# either | both | none.
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
    # engine-signal beats (search telemetry, not protocol)
    "world_collapse": ("analyst", "normal"),
    "deep_think": ("gremlin", "interrupt"),
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


def _from_ability(events) -> str | None:
    """Pull the '[from] ability: X' cause out of a line's trailing args — an
    ability-based immunity (Levitate, Volt Absorb, Flash Fire) names the
    ability that blocked the move; a type immunity carries no such tag."""
    for e in events:
        if e.startswith("[from] ability:"):
            return e.split(":", 1)[1].strip()
    return None


def _status_cause(events) -> str | None:
    """The source of a status: a '[from] move/item/ability: X'. Items matter
    for status specifically — a Toxic Orb / Flame Orb self-inflicts, and
    without naming it the caster grabs a nearby move for the cause (measured
    live: 'the Poison from Psychic Noise' when it was really Toxic Orb)."""
    for e in events:
        for pre in ("[from] move:", "[from] item:", "[from] ability:"):
            if e.startswith(pre):
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


# --- engine-signal prose (the search narrating itself) ---------------------
# These describe what the SEARCH just did, not a board event. Copy lives here
# so the player, the gold set, and the tests share one source; the player
# fills the dynamic bits (revealed count, solved value, the active matchup)
# from live data, exactly as _belief_prose is composed player-side.

def world_collapse_prose(revealed: int) -> str:
    """The K sampled opponent-set worlds folded into one — their sets are
    revealed enough that the speed-pessimistic hedge world is dropped and the
    full search depth pours into the single real line."""
    return (f"their sets are pinned down now — with {revealed} of their "
            "moves revealed the search stopped hedging two possible worlds "
            "and is spending its full depth on one")


def endgame_solved_prose(win_prob: float) -> str:
    """The exact minimax solver took over from MCTS at low material: this
    isn't an estimate any more, it's a proven line. win_prob is OUR win
    share (0..1) from the solved value; state the verdict honestly."""
    if win_prob >= 0.75:
        verdict = "and it's winning — the line is forced from here"
    elif win_prob <= 0.25:
        verdict = "and it's lost — no line saves it"
    else:
        verdict = "and it's razor-thin, down to who mispredicts first"
    return ("the endgame solver just took over — this low-material spot is "
            f"solved exactly now, not estimated, {verdict}")


def deep_think_prose(me: str | None, opp: str | None) -> str:
    """The adaptive search hit a genuinely flat position and is burning
    extra clock on it — the 'hold on, I'm thinking' moment. Names the active
    matchup so the reacting voice has a grounded subject to grab."""
    matchup = f"{me} versus {opp}" if me and opp else "this one"
    return ("nothing separates the options here — the search is spending "
            f"extra clock to grind {matchup} out")


# archetype label -> the desk's one-line matchup read, called at team preview
# when the engine detects a recognizable game shape. EXTENSIBLE: add one row
# per engine archetype mode as the mode enum grows (sun / rain / trick_room /
# hyper_offense / ...). The label MUST come from the SAME detection that sets
# the engine's eval mode, so the call-out and the play can never disagree.
# An unknown/None label yields no beat — stay silent rather than mis-frame a
# matchup the engine didn't actually flag.
_ARCHETYPE_FRAME = {
    "stall": "both cores are wall-heavy — this is a stall mirror, a long grind "
             "won on chip, hazards and PP rather than a sweep",
    # "sun":          "...both sides leaning on the sun...",
    # "rain":         "...swift-swimmers under the rain...",
    # "trick_room":   "...speed inverted, the slow bruisers move first...",
    # "hyper_offense": "...screens and setup, every turn is tempo...",
}


def archetype_prose(label: str | None) -> str | None:
    """The preview archetype call-out for a detected matchup mode, or None if
    the mode is unknown/undetected. Extensible: one row per engine mode in
    _ARCHETYPE_FRAME — the plumbing (label -> beat) is archetype-agnostic."""
    return _ARCHETYPE_FRAME.get(label) if label else None


# residual damage sources as the protocol writes them -> what a caster says
_RESIDUAL_NAME = {
    "psn": "poison", "tox": "poison", "brn": "burn",
    "Stealth Rock": "Stealth Rock", "Spikes": "Spikes",
    "Leech Seed": "Leech Seed", "Life Orb": "Life Orb recoil",
    "Salt Cure": "Salt Cure", "Curse": "Curse", "Nightmare": "Nightmare",
    "confusion": "confusion", "recoil": "recoil", "trapped": "the trap",
}
_STATUS_INFLICT = {
    "frz": "froze {n} solid", "brn": "burned {n}", "par": "paralyzed {n}",
    "slp": "put {n} to sleep", "psn": "poisoned {n}",
    "tox": "badly poisoned {n}",
}
# Passive voice, used ONLY when the cause is an ABILITY. The active templates
# above put the cause in the subject slot ("Flame Body burned our Zamazenta"),
# which reads as something the opponent DID. Measured live 2026-07-27:
# FRACTURE turned that beat into "Moltres just used Flame Body to torch my
# Zamazenta", when Flame Body is a passive that fired off OUR OWN contact
# move. Moves and items keep the active form: a move's user really did act,
# and a Toxic/Flame Orb is self-inflicted so there is no side to confuse.
_STATUS_INFLICT_PASSIVE = {
    "frz": "{n} was frozen solid", "brn": "{n} was burned",
    "par": "{n} was paralyzed", "slp": "{n} was put to sleep",
    "psn": "{n} was poisoned", "tox": "{n} was badly poisoned",
}
# Causes where the mon statuses ITSELF. Nothing we did brought these on, so the
# beat must not offer us the credit: a Toxic/Flame Orb activates on its holder
# at end of turn and Rest is the holder's own move. Saying "we just helped it"
# of a Gliscor's own Toxic Orb hands us agency we never had — the same class of
# error as claiming we set hazards the opponent set.
_SELF_INFLICTED_STATUS = {"toxic orb", "flame orb", "rest"}
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

# ability -> the statuses it turns into an ADVANTAGE. A status the afflicted
# mon's ability wants is not a wound: Poison Heal heals from poison; Guts /
# Quick Feet / Marvel Scale / Flare Boost / Toxic Boost convert it into a
# stat boost (and Guts even ignores burn's Attack drop). Sleep and freeze
# are excluded even for the boost abilities — an immobilized mon can't cash
# the boost in.
_STATUS_SYNERGY = {
    "poisonheal": {"psn", "tox"},
    "guts": {"brn", "psn", "tox", "par"},
    "quickfeet": {"brn", "psn", "tox", "par"},
    "marvelscale": {"brn", "psn", "tox", "par"},
    "flareboost": {"brn"},
    "toxicboost": {"psn", "tox"},
}
_SYNERGY_NAME = {
    "poisonheal": "Poison Heal", "guts": "Guts", "quickfeet": "Quick Feet",
    "marvelscale": "Marvel Scale", "flareboost": "Flare Boost",
    "toxicboost": "Toxic Boost",
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


def _trick_event(by: str, first, second, last_move) -> Event:
    """ONE event for a Trick/Switcheroo swap, naming who used it.

    A swap emits two `-item` lines, one per side. Emitting a beat for each
    spawned two responses from the duo for a single play, and neither beat
    said WHO used the move — both read as passive ("X was handed a Choice
    Scarf by Trick"). Live 2026-07-27: our Gholdengo tricked its Choice Scarf
    onto their Garganacl, and FRACTURE narrated it as the opponent setting her
    up, because nothing in the beat said the play was ours.

    `first`/`second` are (species, item, side, display); `second` is None when
    only one side's item moved. `last_move` is the scanner's (mover, move),
    which identifies the user because flush() runs before the first -item.
    """
    user = (last_move[0] if last_move and last_move[1] == by else None)
    halves = [h for h in (first, second) if h]

    mine = next((h for h in halves if h[0] == user), None)
    theirs = next((h for h in halves if h is not mine), None)

    if mine and theirs:
        # `mine[1]` is what the USER now holds, i.e. what it took
        prose = (f"{mine[3]} used {by} on {theirs[3]}, giving away its "
                 f"{theirs[1]} and taking the {mine[1]}")
        side = mine[2]
    elif mine:
        prose = f"{mine[3]} used {by} and came away with the {mine[1]}"
        side = mine[2]
    elif user and halves:
        h = halves[0]
        prose = f"{user} used {by}, handing {h[3]} a {h[1]}"
        side = h[2]
    else:
        # user unknown: keep the original passive wording rather than invent
        h = halves[0]
        prose = f"{h[3]} was handed a {h[1]} by {by}"
        side = h[2]

    data = {"mon": halves[0][0], "item": halves[0][1]}
    if user:
        data["user"] = user
    return Event("item_tricked", prose, side=side, notable=True, data=data)


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
        # (mover species, move, mover side) — side is what lets an effect
        # tell a SELF-inflicted drop from one the opponent caused
        self._last_move: tuple | None = None
        # slot -> last residual damage source, for naming a
        # non-move KO's actual cause
        self._residual: dict = {}
        # position -> species, from switch/drag/replace details: protocol
        # position tokens carry NICKNAMES ("p1a: Speak Softly"), and prose
        # built from them leaks the nickname ("knocked off Speak Softly" —
        # found scouting real replays; ladder opponents nickname freely)
        self._species: dict[str, str] = {}
        # side ('p1'/'p2') -> set of species that have appeared on that team
        # (from team-preview |poke| lines + every switch). A species on BOTH
        # sides is a mirror MATCH, so a bare name is ambiguous even when the
        # opponent's copy isn't the current active — this is what let FRACTURE
        # call our fainted Clefable "their Cleric" while they had a Toxapex in.
        self._team_species: dict[str, set] = {}


    def _causer(self, trailing, qual, side_of, name_of):
        """Who caused a field-wide effect: an '[of]' ability holder, else the
        move that just resolved. Returns (prose_fragment, cause) or (None,
        None). Weather/hazards/terrain read as things that merely HAPPEN
        otherwise ('Rain set in'), and which side set them is usually the
        whole tactical point."""
        abil = _from_ability(trailing)
        holder = next((e.split("]", 1)[1].strip() for e in trailing
                       if e.startswith("[of]") and "]" in e), None)
        if abil:
            return (f"{qual(holder)}'s {abil}" if holder else abil), abil
        lm = self._last_move
        if lm:
            return lm[0], lm[1]
        return None, None

    def scan(self, messages, role=None) -> list[Event]:
        out: list[Event] = []
        cur = None
        pending_trick = None   # first half of a Trick/Switcheroo swap

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

        def qual_species(species, s) -> str:
            """Mirror-match qualifier for an ALREADY-RESOLVED species.

            Use this whenever the species was captured EARLIER than the prose
            is built. self._species maps a POSITION to whoever occupies it
            NOW, so re-resolving a position token at flush time silently reads
            the wrong Pokemon if the slot changed in between. Live 2026-07-27:
            Garganacl's Ice Punch knocked out our Darkrai, Gholdengo switched
            into p1a before the deferred flush ran, and the beat went out as
            'Garganacl's Ice Punch knocked out our Gholdengo'. The side is
            still taken from the position token, which is stable."""
            if not species or not s:
                return species
            if (species in self._team_species.get("p1", ())
                    and species in self._team_species.get("p2", ())):
                return f"{'our' if s == 'us' else 'their'} {species}"
            return species

        def qual(token) -> str:
            """Species display, prefixed 'our '/'their ' in a mirror MATCH —
            when the same species is on BOTH teams' rosters (self._team_species,
            fed by preview + switches). A bare name is ambiguous then even when
            the opponent's copy is NOT the current active: measured live,
            FRACTURE called our fainted Clefable 'their Cleric' while their
            Toxapex was in. (The earlier version only checked the opposing
            active slot and missed exactly that case.) Non-mirror prose is
            byte-unchanged. Needs role (side) known; role=None (eval) no-ops.

            Safe ONLY when the slot still holds the mon you mean — i.e. at the
            moment the line is scanned. For anything deferred, resolve the
            species up front and call qual_species."""
            return qual_species(name_of(token), side_of(token))

        def flush():
            nonlocal cur
            if cur and cur.get("move"):
                # survives the flush so effects whose protocol line follows
                # the move (Court Change's -swapsideconditions) can name
                # their user
                self._last_move = (cur["mover"], cur["move"],
                                   cur.get("mover_side"))
            if not cur or not cur.get("move"):
                cur = None
                return
            # qualified display names (our/their in a mirror) for PROSE only;
            # cur['mover']/['target'] stay bare species for data + matching.
            # Species come from cur (captured when the move was scanned), NOT
            # from re-resolving the position token: flush() is DEFERRED to the
            # next move, and a faint + switch-in repoints the slot before it
            # runs. Side still comes from the token, which does not move.
            mover_disp = (qual_species(cur["mover"], side_of(cur["mover_pos"]))
                          if cur.get("mover_pos") else cur["mover"])
            target_disp = (qual_species(cur.get("target"),
                                        side_of(cur["target_pos"]))
                           if cur.get("target_pos") else cur.get("target"))
            move_name = cur["move"]
            if cur.get("via"):
                move_name += f" (via {cur['via']})"
            head = f"{mover_disp}'s {move_name}"
            mover_side = cur.get("mover_side")
            target_side = ({"us": "them", "them": "us"}.get(mover_side)
                           if mover_side else None)
            if cur.get("missed"):
                # name the target, same rule as the hit and ko branches: a
                # miss with no victim leaves the caster to guess who dodged
                miss = f"{head} missed"
                if target_disp and target_disp != mover_disp:
                    miss += f" {target_disp}"
                out.append(Event("move_missed", miss,
                                 side=mover_side,
                                 data={"mover": cur["mover"],
                                       "move": cur["move"]}))
                cur = None
                return
            if cur.get("effect") == "no effect":
                imm = cur.get("immune_ability")
                prose = f"{head} had no effect on {target_disp}"
                if imm:
                    # ability immunity: name the cause so the desk credits
                    # the RIGHT ability (Levitate/Volt Absorb), not an invented
                    # one — and a bare 'no effect' stays a type matchup
                    prose += f" — {target_disp}'s {imm} blocked it"
                out.append(Event(
                    "move_no_effect", prose,
                    side=mover_side, notable=True,
                    data={"mover": cur["mover"], "move": cur["move"],
                          "target": cur["target"], "immune_ability": imm}))
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
                line = f"{head} knocked out {target_disp}"
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
                # name the TARGET. Without it the prose states a mover and an
                # effect but no victim ("Great Tusk's Ice Spinner landed super
                # effective and a devastating blow"), and the caster fills the
                # gap with whoever it assumes: measured live 2026-07-27,
                # FRACTURE turned exactly that beat into "I land a massive Ice
                # Spinner", claiming a hit she had actually TAKEN. The ko
                # branch above has always named its target; this is the same
                # rule. Self-targeted moves read oddly with "on X", so skip it
                # when mover and target are the same mon.
                # Put the TARGET next to the verb, not at the end of the line.
                # Naming it at all was the 2026-07-27 fix; trailing it after a
                # clause like "landed not very effective and a heavy hit" was
                # still weak enough to re-anchor — measured live 2026-07-28,
                # "Kingambit's Iron Head landed not very effective and a heavy
                # hit on Toxapex" came back as "Kingambit takes a heavy hit
                # from that Iron Head", inverting who hit whom. "landed not
                # very effective" is also not well-formed English (an
                # adverbial conjoined to a noun phrase), which gives a model
                # every excuse to re-read the sentence.
                if target_disp and target_disp != mover_disp:
                    hit = f"{head} hit {target_disp} — {_join_phrases(tags)}"
                else:
                    hit = f"{head} landed {_join_phrases(tags)}"
                out.append(Event(
                    "move_hit", hit,
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
                       "mover_side": side_of(sm[2]), "mover_pos": sm[2],
                       "target": name_of(sm[4]) if len(sm) > 4 else None,
                       "target_pos": sm[4] if len(sm) > 4 else None,
                       # a move CALLED by another (Sleep Talk, Metronome...):
                       # label it so 'Crunch' on an asleep mon reads as the
                       # Sleep Talk call it is, not a stray direct move
                       "via": _from_move(sm[5:]),
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
                    # an ability-based immunity names its cause; a type
                    # immunity does not — capture it so the beat can tell
                    # 'Levitate blocked it' from a bare Ghost type matchup
                    cur["immune_ability"] = _from_ability(sm[3:])
            elif t == "-miss" and cur:
                cur["missed"] = True
            elif t in ("-damage", "-heal", "-sethp"):
                key = sm[2].split(":")[0]
                frac = _hp_frac(sm[3]) if len(sm) > 3 else None
                old = self._hp.get(key)
                # residual chip carries its source ("[from] psn", "[from]
                # Stealth Rock", "[from] item: Life Orb"). Remember the last
                # one per slot: a faint that is NOT a move's finishing blow
                # otherwise reads as "X went down" straight after X's own
                # attack, and the caster credits that attack with the kill —
                # measured live 2026-07-27, FRACTURE announced "EARTHQUAKE
                # TOOK THE BODY" when Ting-Lu had died to its own poison.
                if t == "-damage":
                    src = next((a.split("]", 1)[1].strip() for a in sm[4:]
                                if a.startswith("[from]") and "]" in a), None)
                    if src:
                        self._residual[key] = src.split(":")[-1].strip()
                # A HEAL was previously bookkeeping only and emitted no event,
                # so the record showed our hit and then a target back near
                # full with nothing in between. Measured live 2026-07-28: a
                # Toxapex kept clicking Recover and PRISM read six straight
                # turns of "that puts Toxapex into range for a KO" while it sat
                # at 99%. Undoing a hit is exactly as much of a swing as
                # landing one, so it has to be said out loud.
                if (t == "-heal" and frac is not None and old is not None
                        and frac > old):
                    gain = frac - old
                    # passive trickles (Leftovers, Poison Heal ~6%) are noise;
                    # a real recovery move puts back a third or more
                    if gain >= 0.10:
                        # flush FIRST: it emits the pending hit and only then
                        # promotes the healer's own move into _last_move, which
                        # is what lets us say "with Recover" instead of naming
                        # whatever move happened to resolve before it
                        flush()
                        src = next((a.split("]", 1)[1].strip() for a in sm[4:]
                                    if a.startswith("[from]") and "]" in a),
                                   None)
                        cause = src.split(":")[-1].strip() if src else None
                        lm = self._last_move
                        if not cause and lm and lm[0] == name_of(sm[2]):
                            cause = lm[1]       # its own Recover/Roost/Soft-Boiled
                        pct = int(round(frac * 100))
                        prose = (f"{qual(sm[2])} healed back to {pct}%"
                                 + (f" with {cause}" if cause else ""))
                        out.append(Event(
                            "heal", prose, side=side_of(sm[2]),
                            notable=gain >= 0.25,
                            data={"mon": name_of(sm[2]), "gain": gain,
                                  "cause": cause, "hp": frac}))
                if frac is not None:
                    self._hp[key] = frac
                if (t == "-damage" and cur and old is not None
                        and frac is not None and cur.get("target_pos")
                        and cur["target_pos"].split(":")[0]
                        == sm[2].split(":")[0]):
                    cur["dmg"] = old - frac
            elif t == "poke" and len(sm) > 3:
                # team preview ('|poke|p1|Clefable, M|item'): seed both rosters
                # so a species shared by both teams reads as a mirror MATCH
                # from turn 1 (switches also feed this, below)
                self._team_species.setdefault(sm[2][:2], set()).add(
                    sm[3].split(",")[0])
            elif t in ("switch", "drag"):
                key = sm[2].split(":")[0]
                self._hp[key] = (_hp_frac(sm[4]) if len(sm) > 4 else 1.0)
                if len(sm) > 3 and sm[3]:
                    species = sm[3].split(",")[0]
                    self._species[key] = species
                    self._team_species.setdefault(key[:2], set()).add(species)
            elif t == "faint" and len(sm) > 2:
                mon = name_of(sm[2])
                if cur and cur.get("target") == mon:
                    cur["ko"] = True  # attribute to the finishing move
                else:
                    flush()  # residual: poison/hazard/recoil/Life Orb etc.
                    why = self._residual.pop(sm[2].split(":")[0], None)
                    prose = (f"{qual(sm[2])} went down to the {_RESIDUAL_NAME.get(why, why)}"
                             if why else f"{qual(sm[2])} went down")
                    out.append(Event("ko", prose,
                                     side=side_of(sm[2]), notable=True,
                                     data={"target": mon, "residual": True,
                                           "cause": why}))
            elif t == "-status" and len(sm) > 3:
                tmpl = _STATUS_INFLICT.get(sm[3])
                if tmpl:
                    flush()  # emit the causing move first, then its effect
                    cause = _status_cause(sm[4:])
                    abil = _from_ability(sm[4:])
                    if abil and sm[3] in _STATUS_INFLICT_PASSIVE:
                        # an ability is a PASSIVE trigger, usually off our own
                        # move making contact. Active voice ("Flame Body
                        # burned X") reads as the opponent acting, so go
                        # passive and attribute the ability to its holder.
                        holder = next(
                            (e.split("]", 1)[1].strip() for e in sm[4:]
                             if e.startswith("[of]") and "]" in e), None)
                        src = f"{qual(holder)}'s {abil}" if holder else abil
                        prose = (_STATUS_INFLICT_PASSIVE[sm[3]]
                                 .format(n=qual(sm[2])) + f" by {src}")
                    else:
                        prose = tmpl.format(n=qual(sm[2]))
                        if cause:
                            # name the cause (move OR item like Toxic Orb) or
                            # downstream commentary invents one (a caster said
                            # "Spore" on a beat that only read "put to sleep",
                            # and "the poison from Psychic Noise" for a Toxic
                            # Orb)
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
                        "status_cured", tmpl.format(n=qual(sm[2])),
                        side=side_of(sm[2]), notable=True,
                        data={"mon": name_of(sm[2]), "status": sm[3]}))
            elif t == "cant" and len(sm) > 3:
                tmpl = _CANT.get(sm[3])
                if tmpl:
                    flush()
                    out.append(Event(
                        "cant_move", tmpl.format(n=qual(sm[2])),
                        side=side_of(sm[2]), notable=True,
                        data={"mon": name_of(sm[2]), "why": sm[3]}))
            elif t == "-enditem" and len(sm) > 3:
                flush()
                mon = name_of(sm[2])        # bare species for data + matching
                mon_q = qual(sm[2])         # our/their in a mirror, for prose
                item = sm[3]
                by = _from_cause(sm[4:])
                ate = any("[eat]" in a for a in sm[4:])
                mside = side_of(sm[2])
                if by == "Knock Off":
                    # name the user: Knock Off is something somebody DID
                    lm = self._last_move
                    who = lm[0] if lm and lm[1] == "Knock Off" else None
                    # lead with "{mover}'s {move}" like the ko/hit branches:
                    # it names the actor AND grounds the move name. Without
                    # the literal "Knock Off" in the beat, a caster naming the
                    # move it actually saw trips the ungrounded-entity guard.
                    prose = (f"{who}'s {by} took the {item} off {mon_q}"
                             if who else f"{item} was knocked off {mon_q}")
                    out.append(Event(
                        "item_knocked_off", prose,
                        side=mside, notable=True,
                        data={"mon": mon, "item": item, "user": who}))
                elif by in ("Thief", "Covet", "Magician", "Pickpocket"):
                    # the -item half of a theft already names the thief
                    # actively; this half was passive, so the same steal read
                    # as an event with no perpetrator
                    lm = self._last_move
                    who = lm[0] if lm and lm[1] == by else None
                    prose = (f"{who} swiped {mon_q}'s {item} with {by}"
                             if who else f"{mon_q}'s {item} was swiped away")
                    out.append(Event(
                        "item_stolen", prose,
                        side=mside, notable=True,
                        data={"mon": mon, "item": item, "user": who}))
                elif item == "Focus Sash":
                    out.append(Event(
                        "sash_saved", f"{mon_q}'s Focus Sash let it cling on",
                        side=mside, notable=True,
                        data={"mon": mon, "item": item}))
                elif item == "Air Balloon":
                    out.append(Event(
                        "balloon_popped", f"{mon_q}'s Air Balloon popped",
                        side=mside, notable=True,
                        data={"mon": mon, "item": item}))
                elif ate:
                    # a berry eaten is routine tempo, not a forced beat
                    out.append(Event("item_eaten", f"{mon_q} ate its {item}",
                                     side=mside,
                                     data={"mon": mon, "item": item}))
                else:
                    out.append(Event("item_used", f"{mon_q} used up its {item}",
                                     side=mside,
                                     data={"mon": mon, "item": item}))
            elif t == "-item" and len(sm) > 3:
                by = _from_cause(sm[4:])
                if by in ("Trick", "Switcheroo"):
                    flush()
                    half = (name_of(sm[2]), sm[3], side_of(sm[2]),
                            qual(sm[2]))
                    if pending_trick is None:
                        # a swap emits TWO -item lines, one per side. Hold the
                        # first: emitting both would spawn two beats for one
                        # play, i.e. two responses from the duo.
                        pending_trick = (by, half)
                    else:
                        out.append(_trick_event(pending_trick[0],
                                                pending_trick[1], half,
                                                self._last_move))
                        pending_trick = None
                elif by in ("Thief", "Covet", "Magician", "Pickpocket"):
                    flush()
                    out.append(Event(
                        "item_stolen",
                        f"{qual(sm[2])} swiped a {sm[3]} with {by}",
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
                # WHAT raised it — the mirror of the unboost attribution below,
                # which this branch never got. A bare "Kingambit sharply raised
                # its Attack" sitting in the same beat as "Kingambit's Iron
                # Head landed" reads as one sentence, and the caster duly said
                # "the Attack boost from Iron Head" — Iron Head does not boost.
                # An unattributed +2 is unanswerable even to a human reading
                # the transcript, which is the whole point of the record.
                mon_q, mon = qual(sm[2]), name_of(sm[2])
                abil = _from_ability(sm[4:])
                holder = next((e.split("]", 1)[1].strip() for e in sm[4:]
                               if e.startswith("[of]") and "]" in e), None)
                lm = self._last_move
                if abil:
                    src = f"{qual(holder)}'s {abil}" if holder else abil
                    prose = f"{mon_q}'s {stat} was {adv}raised by {src}"
                    cause = abil
                elif lm and lm[0] == mon:
                    prose = f"{mon_q} {adv}raised its {stat} with {lm[1]}"
                    cause = lm[1]
                else:
                    prose = f"{mon_q} {adv}raised its {stat}"
                    cause = None
                out.append(Event(
                    "boost", prose,
                    side=side_of(sm[2]), notable=notable,
                    data={"mon": mon, "stat": sm[3],
                          "amount": amt, "cause": cause}))
            elif t == "-unboost" and len(sm) > 4:
                flush()
                stat = _STAT.get(sm[3], sm[3])
                amt = int(sm[4]) if sm[4].lstrip("-").isdigit() else 1
                adv = "sharply " if amt >= 2 else ""
                # WHO cut it. "X's Defense was cut" is the same sentence
                # whether X dropped it with its own Close Combat or the
                # opponent's Intimidate did — opposite readings, and the
                # caster picks one. Ability causes name their holder; an
                # untagged drop right after X's own move is self-inflicted.
                mon_q, mon = qual(sm[2]), name_of(sm[2])
                abil = _from_ability(sm[4:])
                holder = next((e.split("]", 1)[1].strip() for e in sm[4:]
                               if e.startswith("[of]") and "]" in e), None)
                lm = self._last_move
                if abil:
                    src = f"{qual(holder)}'s {abil}" if holder else abil
                    prose = f"{mon_q}'s {stat} was {adv}cut by {src}"
                    cause = abil
                elif lm and lm[0] == mon:
                    prose = (f"{mon_q} {adv}dropped its own {stat} "
                             f"using {lm[1]}")
                    cause = lm[1]
                elif lm:
                    prose = f"{lm[0]}'s {lm[1]} {adv}cut {mon_q}'s {stat}"
                    cause = lm[1]
                else:
                    prose = f"{mon_q}'s {stat} was {adv}cut"
                    cause = None
                out.append(Event(
                    "unboost", prose,
                    side=side_of(sm[2]),
                    data={"mon": mon, "stat": sm[3],
                          "amount": amt, "cause": cause}))
            elif t == "-setboost" and len(sm) > 4:
                flush()
                out.append(Event(
                    "boost",
                    f"{qual(sm[2])} maxed out its "
                    f"{_STAT.get(sm[3], sm[3])}",
                    side=side_of(sm[2]), notable=True,
                    data={"mon": name_of(sm[2]), "stat": sm[3],
                          "maxed": True}))
            elif t in ("-clearallboost", "-invertboost", "-clearboost"):
                flush()
                if t == "-clearallboost":
                    prose = "every stat change was wiped away"
                elif t == "-invertboost":
                    prose = (f"{qual(sm[2])}'s stat changes were inverted"
                             if len(sm) > 2 else
                             "the stat changes were inverted")
                else:
                    prose = (f"{qual(sm[2])}'s boosts were cleared"
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
                        entry[0].format(n=qual(sm[2])),
                        side=side_of(sm[2]), notable=entry[1],
                        data={"mon": name_of(sm[2]), "volatile": key}))
            elif t == "-end" and len(sm) > 3:
                entry = _VOL_END.get(_cond_name(sm[3]).lower())
                if entry:
                    flush()
                    out.append(Event(
                        "volatile_end",
                        entry[0].format(n=qual(sm[2])),
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
                    f"{qual(sm[2])} transformed into {qual(sm[3])}",
                    side=side_of(sm[2]), notable=True,
                    data={"mon": name_of(sm[2]),
                          "into": name_of(sm[3])}))
            elif t == "-prepare" and len(sm) > 3:
                flush()
                out.append(Event(
                    "charging", f"{qual(sm[2])} is charging up {sm[3]}",
                    side=side_of(sm[2]),
                    data={"mon": name_of(sm[2]), "move": sm[3]}))
            elif t == "-sidestart" and len(sm) > 3:
                flush()
                poss = side_poss(sm[2])
                cond = _cond_name(sm[3])
                low = cond.lower()
                # setting hazards/screens is routine tempo — record it, but
                # don't force a beat (removing them below IS a swing)
                who, cause = self._causer(sm[4:], qual, side_of, name_of)
                if low in _HAZARDS:
                    prose = (f"{who} set {cond} on {poss} side" if who
                             else f"{cond} went up on {poss} side")
                    out.append(Event(
                        "hazard_set", prose,
                        side=side_of(sm[2]),
                        data={"condition": cond, "user": who}))
                elif low in _SCREENS:
                    prose = (f"{who} put {cond} up on {poss} side" if who
                             else f"{cond} went up on {poss} side")
                    out.append(Event(
                        "screens_set", prose,
                        side=side_of(sm[2]),
                        data={"condition": cond, "user": who}))
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
                # Name the actor and keep the possessive on the SIDE, mirroring
                # the setter above. "our Stealth Rock was cleared away by Rapid
                # Spin" was passive and attached "our" to the condition, so it
                # read as "the rocks WE set" rather than "the rocks on our
                # side" — and both personas duly inverted it, FRACTURE claiming
                # she set rocks the opponent had set and casting our own
                # spinner as the thief, PRISM calling our own hazard removal
                # "the loss of entry hazards". Whose side the hazard sat on is
                # the entire tactical point, so it must not be guessable.
                who, _cause = self._causer(sm[4:], qual, side_of, name_of)
                if low in _HAZARDS or low in _SCREENS or low == "tailwind":
                    if by:
                        etype = ("hazard_cleared" if low in _HAZARDS
                                 else "side_cleared")
                        if who:
                            prose = (f"{who} cleared {cond} from "
                                     f"{poss} side with {by}")
                        else:
                            prose = (f"{cond} was cleared from "
                                     f"{poss} side by {by}")
                        out.append(Event(
                            etype, prose,
                            side=side_of(sm[2]), notable=True,
                            data={"condition": cond, "by": by, "user": who}))
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
                            who, cause = self._causer(sm[3:], qual, side_of,
                                                      name_of)
                            prose = (f"{who} set {label} up" if who
                                     else f"{label} set in")
                            out.append(Event(
                                "weather_set", prose,
                                notable=True,
                                data={"weather": label, "user": who}))
                            self._weather = label
            elif t == "-fieldstart" and len(sm) > 2:
                cond = _cond_name(sm[2])
                flush()
                who, cause = self._causer(sm[3:], qual, side_of, name_of)
                prose = (f"{who} brought up {cond}" if who
                         else f"{cond} took over the field")
                out.append(Event("field_start", prose,
                                 notable=True,
                                 data={"condition": cond, "user": who}))
            elif t == "-fieldend" and len(sm) > 2:
                cond = _cond_name(sm[2])
                if cond.lower() == "trick room":
                    flush()
                    out.append(Event("field_end", "Trick Room wore off",
                                     data={"condition": cond}))
        flush()
        if pending_trick is not None:
            # only one side's item moved (the target held nothing): still one
            # beat, and it still names who did it
            out.append(_trick_event(pending_trick[0], pending_trick[1], None,
                                    self._last_move))
        return out


# --- event -> beat classification -----------------------------------------

# gremlin registers by allegiance for the same event class — the docs'
# "allegiance determines the read" rule (gc-0003/0004 and kin)
_LUCK_REGISTERS = {"us": "persecution", "them": "delight"}


def classify(ev: Event, stats_fn=None, ability_fn=None) -> Beat | None:
    """Map one Event to a Beat, or None for pure color (rides along in the
    beat text without owning a moment). stats_fn(species_display) ->
    (atk, spa) enables the burn physical-vs-special split; ability_fn(
    species_display, side) -> set of possible normalized ability ids enables
    the status-synergy read (a status the mon's ability wants is a boon)."""
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
    if t == "world_collapse":
        # the search dropped its parallel opponent-set worlds down to one:
        # a certainty gain, PRISM's dry meta-observation (gremlin has no
        # outrage to hang on the machine simplifying its own model)
        return make_beat("world_collapse", ev.prose,
                         register="worlds-collapsed", **ev.data)
    if t == "endgame_solved":
        # exact minimax replaced the estimate: analyst's flagship "the game
        # is provably over" moment. Routes through the endgame beat.
        return make_beat("endgame", ev.prose, register="solved", **ev.data)
    if t == "deep_think":
        # the search paused on a flat position — FRACTURE's "hold on, I'm
        # thinking" bit (she owns the pilot's deliberation), delivered
        # out-of-band so it lands DURING the pause (see Director.interject)
        return make_beat("deep_think", ev.prose, register="deliberating",
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
        mon = ev.data.get("mon")
        # a status the afflicted mon's ability WANTS is a boon, not a wound:
        # Toxic on a Poison Heal Gliscor heals it; burn on a Guts attacker
        # boosts it. Read it as the plan working — never the gremlin's grief.
        if ability_fn is not None and mon and status:
            abilities = ability_fn(mon, ev.side) or set()
            synergy = {a for a in abilities
                       if status in _STATUS_SYNERGY.get(a, ())}
            if synergy:
                # certain when EVERY possible ability wants it (our own mons
                # resolve to a single known ability -> definitive)
                certain = all(status in _STATUS_SYNERGY.get(a, ())
                              for a in abilities)
                name = _SYNERGY_NAME[sorted(synergy)[0]]
                if ev.side == "us":
                    tail = (f" — that just feeds {mon}'s {name}" if certain
                            else f" — if that's the {name} set, it just fed it")
                    reg = "status-boon"
                elif cause.lower() in _SELF_INFLICTED_STATUS:
                    # they did this to themselves; it is their engine starting,
                    # not our mistake
                    tail = (f" — that switches on {mon}'s {name}" if certain
                            else f" — if that's the {name} set, that just came"
                                 f" online")
                    reg = "status-backfire"
                else:
                    tail = (f" — but that turns on {mon}'s {name}" if certain
                            else f" — but if that's the {name} set, we just "
                                 f"helped it")
                    reg = "status-backfire"
                return make_beat("status", ev.prose + tail, persona="analyst",
                                 priority="normal",
                                 register=reg if certain else reg + "-hedge",
                                 **ev.data)
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
                 floor: float = 5.0, stats_fn=None, ability_fn=None,
                 min_turn_gap: int = 0, quiet_turn_gap: int = 3):
        # Turn gating (min_turn_gap > 0) replaces the wall-clock floor and
        # interval. Under PTS scheduling the audience watches on the CLIENT's
        # timeline, so "am I talking too much" is a question about viewer
        # time, and viewer time is proportional to TURNS — not to how fast the
        # engine resolved them. Measured 2026-07-27: with pacing off a 40-turn
        # game resolved in ~3 minutes, so the 5s floor silenced nearly every
        # turn and the whole broadcast got FIVE beats for a 10+ minute watch.
        # 0 keeps the old time gating, so nothing changes unless asked.
        self.min_turn_gap = min_turn_gap
        self.quiet_turn_gap = quiet_turn_gap
        self.min_interval = min_interval
        self.min_swing = min_swing
        self.floor = floor
        self.stats_fn = stats_fn
        self.ability_fn = ability_fn
        self.reset()

    def reset(self):
        self._pending: list[Event] = []
        self._notable = False
        self._prev_value: float | None = None
        self._prev_read: str | None = None
        self._prev_disagree: str | None = None
        self._last_beat_turn: int | None = None
        self._prev_fainted: tuple[frozenset, frozenset] = (frozenset(),
                                                           frozenset())
        # (side, mon) -> consecutive decision points spent asleep/frozen
        self._afflicted: dict = {}
        # (side, mon) whose current sleep is a deliberate Rest — the
        # escalating-affliction callback must not grieve over a chosen recovery
        self._rest_sleepers: set = set()

    # --- ingestion -----------------------------------------------------
    def observe(self, events: list[Event]):
        for ev in events:
            self._pending.append(ev)
            if ev.notable:
                self._notable = True
            # a Rest is a chosen recovery, not an enemy affliction — remember
            # it so _tick_afflictions doesn't have the gremlin litigate the
            # sleep turn after turn ("our Dondozo is STILL asleep")
            if (ev.type == "status_applied"
                    and ev.data.get("status") == "slp"
                    and ev.data.get("cause") == "Rest"):
                self._rest_sleepers.add((ev.side, ev.data.get("mon")))
        # keep the buffer bounded; the freshest beats matter most
        if len(self._pending) > 6:
            self._pending = self._pending[-6:]

    def note(self, prose: str, side: str | None = None):
        """A driver-side color note (e.g. 'we send Gliscor in' on forced
        switches) that should ride along in the next beat."""
        self.observe([Event("note", prose, side=side)])

    # --- match framing -------------------------------------------------
    def match_start(self, opponent: str, our_team: list[str],
                    their_team: list[str], lead: str | None = None,
                    archetype: str | None = None) -> str:
        self.reset()
        text = (f"[MATCH START] New battle vs {opponent or 'the opponent'}. "
                f"Our team: {', '.join(our_team) or 'unknown'}. "
                f"Their preview: {', '.join(their_team) or 'hidden'}.")
        if lead:
            text += f" We lead {lead}."
        # engine-detected matchup archetype (stall mirror, sun, rain, ...):
        # a preview read the desk calls and the gremlin reacts to. Byte-
        # unchanged when no archetype is flagged.
        frame = archetype_prose(archetype)
        if frame:
            text += f" The engine reads the matchup: {frame}."
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

    def interject(self, kind: str, turn: int, prose: str,
                  **data) -> tuple[str, Beat] | None:
        """Compose a standalone, OUT-OF-BAND engine beat — one that must land
        WHEN it happens (mid-decision) rather than folded into the next
        per-turn decide() aggregation. The 'hold on, I'm thinking' stall is
        the case: it has to be spoken while the search is still grinding, not
        after the move resolves. Framing beats (match_start/match_end) are the
        same out-of-band pattern. Returns (feed_text, Beat), or None if the
        kind classifies to nothing. The feed text keeps the '[BATTLE Tn]'
        shape the overlay's beat gate and turn parser expect."""
        beat = classify(Event(kind, prose, notable=True, data=dict(data)),
                        self.stats_fn, self.ability_fn)
        if beat is None:
            return None
        return f"[BATTLE T{turn}] {beat.prose}", beat

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

        if self.min_turn_gap:
            # viewer-time gating: turns since the last beat we SENT
            gap = (None if self._last_beat_turn is None
                   else ctx.turn - self._last_beat_turn)
            if gap is not None and gap < self.min_turn_gap:
                return Decision(None, [], True)
            quiet_due = gap is None or gap >= self.quiet_turn_gap
        else:
            if ctx.elapsed < self.floor:
                return Decision(None, [], True)
            quiet_due = ctx.elapsed >= self.min_interval
        if not (self._notable or faints or swing is None
                or abs(swing) >= self.min_swing or quiet_due):
            return Decision(None, [], True)

        pairs = [(ev, classify(ev, self.stats_fn, self.ability_fn))
                 for ev in self._pending]
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
            # prefer the classified beat's prose: identical to the event's
            # for most beats, but carries added reads (the status-synergy
            # "— that feeds its Poison Heal" tail) into the spoken text
            hl = [(b.prose if b and b.prose else ev.prose)
                  for _, ev, b in cand]
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
        # in a species mirror, tag the actives our/their — an unqualified
        # 'Corviknight (5% hp) vs Corviknight (81% hp)' left the caster unable
        # to tell which side is ours and it flipped HP ownership (measured live)
        mirror = bool(ctx.me_name) and ctx.me_name == ctx.opp_name
        me_n = f"our {ctx.me_name}" if mirror else ctx.me_name
        opp_n = f"their {ctx.opp_name}" if mirror else ctx.opp_name
        me = (f"{me_n} ({ctx.me_hp}% hp)" if ctx.me_name else "Our side")
        opp = (f"{opp_n} ({ctx.opp_hp}% hp)" if ctx.opp_name else "their side")
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
        self._last_beat_turn = ctx.turn
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
                # a deliberate Rest sleep is a chosen recovery — count it but
                # never surface the gremlin grievance for it
                if n >= 2 and key not in self._rest_sleepers:
                    word = "asleep" if status == "slp" else "frozen"
                    whose = "our" if side == "us" else "their"
                    lines.append(f"{whose} {name} is STILL {word} "
                                 f"(turn {n} of it)")
        for key in list(self._afflicted):
            if key not in seen:
                del self._afflicted[key]
                self._rest_sleepers.discard(key)   # woke / left the field
        return "; ".join(lines) if lines else None
