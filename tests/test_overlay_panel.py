"""Panel duo-caption rendering: speaker tags, exchange history, height."""
import io
import contextlib
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown import overlay_kitty as ok


def _render_plain(state, history):
    shutil.get_terminal_size = lambda fallback=(110, 40): os.terminal_size(
        (110, 40))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok.render(state, connected=True, history=history)
    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", buf.getvalue())


def test_duo_exchange_rendered_with_tags():
    state = {"turn": 7, "text": "It was the search's switch.",
             "persona": "PRISM", "us": "Gholdengo", "us_hp": 74,
             "them": "Iron Valiant", "them_hp": 63,
             "us_alive": 6, "them_alive": 6, "mom": 0.66,
             "read": "we hold a real edge", "moment": None}
    history = [("FRACTURE", "THAT WAS MY SWITCH. All me."),
               ("PRISM", "It was the search's switch.")]
    plain = _render_plain(state, history)
    assert "FRACTURE  THAT WAS MY SWITCH. All me." in plain
    assert "PRISM  It was the search's switch." in plain
    # header carries both names now
    assert "◆ PRISM" in plain and "FRACTURE" in plain.split("\n")[1]


def test_single_voice_fallback():
    """No history (legacy/AIRI mode): current line renders untagged."""
    state = {"turn": 3, "text": "A quiet turn of hazard setting.",
             "persona": None, "us": "Gliscor", "us_hp": 100,
             "them": "Kingambit", "them_hp": 100,
             "us_alive": 6, "them_alive": 6, "mom": 0.5,
             "read": "dead even", "moment": None}
    plain = _render_plain(state, None)
    assert "A quiet turn of hazard setting." in plain


def test_long_lines_budget_newest_first():
    long = ("This is a deliberately long analyst line that wraps across "
            "many panel rows to make sure the newest speaker keeps their "
            "space while the older line yields, because the exchange reads "
            "newest-up and the previous line is context, not content, and "
            "it should truncate before the current line ever does. " * 3)
    state = {"turn": 9, "text": long, "persona": "PRISM",
             "us": "A", "us_hp": 1, "them": "B", "them_hp": 1,
             "us_alive": 1, "them_alive": 1, "mom": 0.5,
             "read": "x", "moment": None}
    history = [("FRACTURE", "Short scream."), ("PRISM", long)]
    plain = _render_plain(state, history)
    assert "PRISM" in plain
    assert plain.count("deliberately long analyst line") >= 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("panel tests passed")
