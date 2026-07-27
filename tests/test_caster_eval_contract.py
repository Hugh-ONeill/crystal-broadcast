"""Guards for the failure that actually happened: a manual-only check rotting.

`commentary_eval.run_caster` monkeypatches `Caster.publish` with a `collect`
stub to capture spoken lines. When publish gained its `citations` argument
(crystal-battle 12182a6, the sources card) the stub was not updated, so EVERY
caster-level run died with a TypeError on the first entry. Nobody noticed for a
week, because that level is only ever run by hand while the deterministic level
is gated by pytest on every run.

Two guards, deliberately cheap:
  * a signature pin, which needs nothing running and catches the exact drift;
  * a live smoke test, skipped unless a model endpoint is up, which exercises
    the real speak() -> publish() path end to end.

Neither imports the wobble band: the gold set's spoken-line pass rate is
generation variance and stays a manual judgement call.
"""
import asyncio
import inspect
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from crystal_broadcast.caster import Caster, DEFAULT_UPSTREAM

# The call shape commentary_eval's `collect` stub must accept. If you change
# Caster.publish, this test fails ON PURPOSE: update the stub in
# commentary_eval.run_caster in the same commit.
EXPECTED_PUBLISH_PARAMS = ["self", "beat_text", "persona", "line", "hud",
                           "citations"]


def test_publish_signature_is_pinned_to_the_eval_stub():
    params = list(inspect.signature(Caster.publish).parameters)
    assert params == EXPECTED_PUBLISH_PARAMS, (
        "Caster.publish changed. commentary_eval.run_caster replaces it with a "
        "`collect` stub that must accept the same call — update BOTH, then "
        "update EXPECTED_PUBLISH_PARAMS here. Last time this drifted the "
        "caster-level gold set was dead for a week."
    )


def _endpoint_up(url: str) -> bool:
    host, _, port = url.split("//", 1)[1].partition(":")
    try:
        with socket.create_connection((host, int(port or 80)), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _endpoint_up(DEFAULT_UPSTREAM),
                    reason=f"no model endpoint at {DEFAULT_UPSTREAM}")
def test_speak_reaches_publish_without_drift():
    """Real generation through the real publish. Catches signature drift and
    anything else that raises inside speak(); asserts nothing about the
    CONTENT, which is what makes it a gate rather than a wobble."""
    caster = Caster(DEFAULT_UPSTREAM, "gemma4:26b-a4b-it-q4_K_M",
                    expert_url=None)
    published = []
    real_publish = caster.publish

    async def spy(*args, **kwargs):
        published.append(args)
        return await real_publish(*args, **kwargs)

    caster.publish = spy
    asyncio.run(caster.speak({
        "text": "[BATTLE T5] Last exchange: Garganacl's Salt Cure landed on "
                "our Gholdengo. our Gholdengo (61% hp) vs Garganacl (87% hp). "
                "Bodies: us 5 standing, them 6.",
        "beats": [],
        "hud": {"us": "Gholdengo", "them": "Garganacl", "turn": 5},
    }))
    assert published, "speak() never reached publish()"
    assert len(published[0]) == len(EXPECTED_PUBLISH_PARAMS) - 1
