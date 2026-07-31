#!/usr/bin/env python3
"""Shared dex access: the entity index behind the ungrounded-entity contract,
plus the stat lookup the director uses for its burn split.

Lives here rather than in commentary_eval so the LIVE caster can use the same
index without importing the eval (which imports the caster — that way lies a
cycle). poke_env is imported LAZILY and every caller degrades gracefully if it
is absent: this package's only hard dependency is websockets.
"""
from __future__ import annotations

import re


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


class GameData:
    """Lazy GenData wrappers: stats for the director's burn split and the
    entity index for the ungrounded-entity contract check."""

    def __init__(self):
        self._gen = None
        self._status_moves = None
        self._entity_names = None
        self._kits: dict = {}

    @property
    def gen(self):
        if self._gen is None:
            from poke_env.data import GenData
            self._gen = GenData.from_gen(9)
        return self._gen

    def status_moves(self) -> dict:
        """{status_code: set(lowercased move display names)} — every move
        that can inflict each status, from its primary `status` field or its
        `secondary.status` (so all freezing moves, not just Ice Beam).
        Used to ground a caster naming the move behind a status on the board.
        (Tri Attack's 1/3 burn/freeze/para rides an onHit callback with no
        status field, so it isn't captured — a known, harmless gap.)"""
        if self._status_moves is None:
            m: dict = {}
            for e in self.gen.moves.values():
                code = e.get("status") or (e.get("secondary") or {}).get(
                    "status")
                if code and e.get("name"):
                    m.setdefault(code, set()).add(e["name"].lower())
            self._status_moves = m
        return self._status_moves

    def stats(self, display_name: str):
        entry = self.gen.pokedex.get(_norm(display_name))
        if entry and "baseStats" in entry:
            bs = entry["baseStats"]
            return bs.get("atk", 0), bs.get("spa", 0)
        return None

    def move_type(self, display_name: str) -> str | None:
        """'Icicle Spear' -> 'Ice'. None for an unknown move."""
        entry = self.gen.moves.get(_norm(display_name))
        t = (entry or {}).get("type")
        return t.title() if isinstance(t, str) else None

    def species_types(self, display_name: str) -> list[str]:
        """'Ceruledge' -> ['Fire', 'Ghost']. Empty for an unknown species.
        This is the DEX typing — a Terastallized mon is a different question
        and the caller has to supply the tera type."""
        entry = self.gen.pokedex.get(_norm(display_name))
        ts = (entry or {}).get("types") or []
        return [t.title() for t in ts if isinstance(t, str)]

    def species_kit(self, display_name: str) -> list[str]:
        """Display names of every move a species can learn, plus its
        abilities. Team preview publishes the roster, and what those six
        CAN run is dex knowledge rather than anything observed — so at
        preview it is fair evidence for the desk to reason from, while a
        move on a mon that is not in the game at all still is not.
        Cached: the lists are large and the same six recur all match."""
        key = _norm(display_name)
        hit = self._kits.get(key)
        if hit is not None:
            return hit
        entry = self.gen.pokedex.get(key) or {}
        out = []
        # Walk the PREVO chain: Showdown stores an inherited move on the
        # stage that first learns it, so Kingambit's own learnset has no
        # Sucker Punch — it comes up from Pawniard. Reading only the final
        # stage silently under-reports the kit by a third.
        seen, cur = set(), key
        while cur and cur not in seen:
            seen.add(cur)
            ls = (self.gen.learnset.get(cur) or {})
            ls = ls.get("learnset", ls)
            for mid in ls:
                name = (self.gen.moves.get(mid) or {}).get("name")
                if isinstance(name, str):
                    out.append(name)
            nxt = (self.gen.pokedex.get(cur) or {}).get("prevo")
            cur = _norm(nxt) if isinstance(nxt, str) and nxt else None
        for ab in (entry.get("abilities") or {}).values():
            if isinstance(ab, str):
                out.append(ab)
        self._kits[key] = out
        return out

    def effectiveness(self, atk_type: str, def_types) -> float | None:
        """Damage multiplier of one attacking type into a defender's typing.

        poke_env's chart is indexed [DEFENDER][ATTACKER], which is easy to get
        backwards — and the chart is genuinely asymmetric, so a transposed
        lookup returns a plausible wrong number rather than an error. Verified
        against the engine's own table: Ice into Fire is 0.5, Ice into Fairy is
        1.0 (NOT resisted), Dragon into Fairy is 0.0.
        """
        if not atk_type or not def_types:
            return None
        chart = getattr(self.gen, "type_chart", None)
        if not chart:
            return None
        mult = 1.0
        for d in def_types:
            row = chart.get(d.upper())
            if row is None:
                return None
            v = row.get(atk_type.upper())
            if v is None:
                return None
            mult *= float(v)
        return mult

    def entity_names(self) -> list[str]:
        """Species + moves + ABILITIES.

        Abilities were missing until 2026-07-27 and are the class the analyst
        actually hallucinates: PRISM invoked "Multiscale" on a beat that never
        mentioned it, and earlier demos produced "Good as Gold" and a mangled
        Drain Punch. They are not a top-level GenData table, so they come off
        the pokedex entries. Callers rely on EXACT-CASE matching, which is
        what keeps common-word abilities (Guts, Static, Pressure) from
        false-flagging ordinary prose.
        """
        if self._entity_names is not None:
            return self._entity_names
        names = [e["name"] for e in self.gen.pokedex.values() if "name" in e]
        names += [e["name"] for e in self.gen.moves.values() if "name" in e]
        for e in self.gen.pokedex.values():
            ab = e.get("abilities") or {}
            names += [v for v in ab.values() if isinstance(v, str)]
        # multi-word first so "Iron Valiant" wins over "Iron"
        self._entity_names = sorted(set(names), key=len, reverse=True)
        return self._entity_names


DATA = GameData()
