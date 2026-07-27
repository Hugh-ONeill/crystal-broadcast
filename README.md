# Crystal Broadcast

An AI commentary system for live Pokemon battles. A **beat director** decides
what is worth saying and when, a **two-persona caster** says it, and a
**presentation-clock scheduler** makes sure it lands when the viewer actually
sees the moment.

Built to narrate [crystal-battle](https://github.com/Hugh-ONeill/crystal-battle),
a competitive MCTS agent, but it depends on nothing from it: the dependency runs
one way, engine to broadcast.

## The inversion

Most AI-VTuber projects put the LLM in the driver's seat and let it play the
game. Here a strong non-LLM agent plays, and the LLM is a *grounded observer*:
every factual claim is checked against the battle protocol, mechanics are
retrieved from a citation-backed RAG, and a gold set scores the commentary the
way an eval suite scores a model.

## Parts

| | |
|---|---|
| `beat_director.py` | Protocol to typed events to **beats**. Pure logic: no I/O, no wall clock, offline-drivable, which is what makes it testable and replay-auditable. |
| `caster.py` | The duo. PRISM (analyst, cited, refusal-gated) and FRACTURE (gremlin, vibes-only, grudge-powered). Shared transcript so one can correct the other. |
| `pts_clock.py` / `presentation_clock.py` / `broadcast_clock.js` | **Presentation-time scheduling.** The client animates a turn over seconds while the engine resolves it in milliseconds; measured lag ranged 1.00s to 158.56s in a single match, so a fixed pace cannot work. The injected hook reports when each protocol line is actually shown; the caster holds each finished line until the viewer gets there. |
| `beat_audit.py` | Turn-keyed audit: turn to protocol to events to beats. A beat read alone sounds fine; only against its own turn can you see it credit the wrong actor. |
| `commentary_eval.py` + `gold/` | Gold-set eval, deterministic director layer plus a caster layer. |
| `grudge_ledger.py` | FRACTURE's Book of Grudges, mined from real replays. She never invents a vendetta. |
| `commentary_overlay.py` + `overlay.html` | Broadcast lower-third. Renders only what the battle frame cannot: the desk read, the exchange, the citations. |
| `serve_client.py` + `broadcast_client.css` | Self-hosted Showdown client in broadcast mode, chrome stripped, scene scaled. |

## Attribution is the hard part

The recurring bug class is not hallucination. It is beats that are *factually
true* but assign agency wrongly: a KO credited to the mon that replaced the
victim, our own Trick narrated as the opponent's trap, an ability that fired off
our contact move read as the opponent using it. String-based checks cannot see
any of these, which is what `beat_audit.py` exists for.

## Inspirations

- [Neuro-sama](https://www.twitch.tv/vedal987), for the demonstration that the
  hard problem is the director, not the model.
- [Project AIRI](https://github.com/moeru-ai/airi), whose event protocol the
  caster bridge was originally written against. An inspiration, not a
  dependency.
