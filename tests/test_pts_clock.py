"""PTS scheduling tests. The socket is deliberately not exercised: ingest()
is split out from the feed so the gate logic is testable without a network,
the same way the director is driven without a live battle."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.pts_clock import PresentationClock


def _presented(line):
    return {"kind": "presented", "line": line, "t": 0}


def test_turn_none_publishes_immediately():
    """MATCH START carries no turn and must not be held."""
    c = PresentationClock(max_hold=5)
    assert asyncio.run(c.wait_for(None)) == 0.0


def test_already_reached_does_not_hold():
    c = PresentationClock(max_hold=5)
    c.ingest(_presented("|turn|12"))
    assert c.reached(9) and c.reached(12)
    assert asyncio.run(c.wait_for(9)) < 0.05


def test_holds_until_the_viewer_reaches_the_turn():
    c = PresentationClock(max_hold=5)
    c.ingest(_presented("|turn|3"))

    async def scenario():
        async def advance():
            await asyncio.sleep(0.05)
            c.ingest(_presented("|turn|6"))     # still short of 7
            await asyncio.sleep(0.05)
            c.ingest(_presented("|turn|7"))
        waiter = asyncio.create_task(c.wait_for(7))
        await asyncio.gather(advance(), waiter)
        return waiter.result()

    held = asyncio.run(scenario())
    assert held >= 0.09, "must have waited for turn 7, not turn 6"
    assert c.holds == 1


def test_hold_is_bounded_so_a_closed_page_cannot_mute_commentary():
    """If the broadcast page is shut, the viewer never advances. Late is
    acceptable; silent is not."""
    c = PresentationClock(max_hold=0.2)
    held = asyncio.run(c.wait_for(99))
    assert 0.2 <= held < 1.0
    assert c.timeouts == 1


def test_final_beat_waits_for_the_battle_to_end():
    """[RESULT] must not spoil the finish while the last exchange is still
    animating."""
    c = PresentationClock(max_hold=5)
    c.ingest(_presented("|turn|40"))

    async def scenario():
        async def finish():
            await asyncio.sleep(0.05)
            c.ingest(_presented("|win|CBAiri"))
        waiter = asyncio.create_task(c.wait_for(40, final=True))
        await asyncio.gather(finish(), waiter)
        return waiter.result()

    assert asyncio.run(scenario()) >= 0.04
    assert c.ended


def test_a_new_battle_rewinds_the_viewer():
    c = PresentationClock(max_hold=5)
    c.ingest(_presented("|turn|30"))
    assert c.reached(30)
    c.ingest(_presented("|start"))
    assert not c.reached(1), "next battle starts from the beginning again"
    assert not c.ended


def test_queued_events_do_not_advance_the_viewer():
    """Arrival is the SERVER's clock; only 'presented' is the viewer's."""
    c = PresentationClock(max_hold=5)
    c.ingest({"kind": "queued", "line": "|turn|20", "t": 0})
    assert c.highest_turn is None
