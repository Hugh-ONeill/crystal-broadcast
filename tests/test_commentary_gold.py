"""The commentary gold set's deterministic layer must stay green: this is
the regression gate for the director's routing/registers/silence. The
caster-level (LLM) layer is run manually — generation wobbles."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_gold_director_level_green():
    r = subprocess.run(
        [sys.executable, str(ROOT / "showdown" / "commentary_eval.py")],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"gold set regressed:\n{r.stdout}{r.stderr}"


if __name__ == "__main__":
    test_gold_director_level_green()
    print("ok gold set (director level)")
