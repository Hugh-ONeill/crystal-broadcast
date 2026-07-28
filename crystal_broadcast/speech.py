#!/usr/bin/env python3
"""Thin client for the duo TTS service. Opt-in, and degrades to silence.

Stdlib only, on purpose. The synth is Qwen3-TTS on CUDA living in its own venv
(`~/Developer/caster-avatars/tts_server.py`); importing any of that here would
drag torch into a package whose only hard dependency is `websockets`. So this
speaks HTTP, exactly like the caster already does to Ollama and grounded-rag.

Everything fails soft. If the service is down, slow, or absent, `speak()`
returns None and the broadcast carries on as text — audio is opt-in and a
missing voice must never cost a line.
"""
from __future__ import annotations

import json
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8133"

# Fallback when a line has not been synthesised yet. The caster's duration_fn
# is called ON THE EVENT LOOP, so it can never wait for a render; this is a
# reading-rate estimate used only until the real duration is known.
_WORDS_PER_SECOND = 3.1


def estimate_seconds(line: str) -> float:
    return max(1.0, len((line or "").split()) / _WORDS_PER_SECOND)


class Speech:
    """Synthesises lines and remembers how long they turned out to be.

    The remembering is the point: `duration_fn` has to answer instantly on the
    event loop, so a line rendered in a worker thread leaves its real duration
    behind for the scheduler to read.
    """

    def __init__(self, url: str = DEFAULT_URL, play: bool = True,
                 out_dir: str | None = None, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.play = play
        self.timeout = timeout
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self._seconds: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()
        self._seq = 0

    # --- health -------------------------------------------------------
    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=2) as r:
                return json.loads(r.read()).get("ok", False)
        except Exception:
            return False

    # --- the caster's contract ----------------------------------------
    def duration_fn(self, persona: str, line: str) -> float:
        """Seconds a line takes to say. Never blocks: the real figure once the
        render is done, a reading-rate estimate before that."""
        with self._lock:
            known = self._seconds.get((persona, line))
        return known if known is not None else estimate_seconds(line)

    # --- synthesis ----------------------------------------------------
    def speak(self, persona: str, line: str,
              register: str | None = None) -> float | None:
        """Render (and optionally play) one line. Returns its duration, or
        None if the service could not be reached. Worker-thread only."""
        line = (line or "").strip()
        if not line:
            return None
        body = json.dumps({"persona": persona, "text": line,
                           "register": register}).encode()
        req = urllib.request.Request(
            f"{self.url}/speak", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                audio = r.read()
                seconds = float(r.headers.get("X-Speech-Seconds") or 0) or None
        except (urllib.error.URLError, OSError, ValueError):
            return None            # silence beats losing the line
        if seconds is None:
            return None
        with self._lock:
            self._seconds[(persona, line)] = seconds
            self._seq += 1
            seq = self._seq
        path = None
        if self.out_dir:
            path = self.out_dir / f"{seq:04d}_{persona.lower()}.wav"
            path.write_bytes(audio)
        if self.play:
            self._play(audio, path)
        return seconds

    def _play(self, audio: bytes, path: Path | None) -> None:
        """Best effort. A missing player must not raise into the caster."""
        try:
            if path is not None:
                subprocess.Popen(["paplay", str(path)],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            else:
                p = subprocess.Popen(["paplay"], stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                p.stdin.write(audio)
                p.stdin.close()
        except Exception:
            pass
