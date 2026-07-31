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
import queue
import subprocess
import threading
import time
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

    # After this many consecutive failed renders the breaker opens and
    # speak() stops calling the service until the cooldown expires. Take 80:
    # the service died at T17 and every following line then sat on the full
    # timeout before returning None — six lines at 30s each, which is what
    # dragged beat-age-at-voicing to 83 SECONDS and made true commentary air
    # over a board it no longer described. A dead TTS should cost the
    # broadcast its audio, not its timing.
    BREAKER_AFTER = 2
    BREAKER_COOLDOWN = 30.0

    def __init__(self, url: str = DEFAULT_URL, play: bool = True,
                 out_dir: str | None = None, timeout: float = 12.0,
                 sink: str | None = None):
        self.url = url.rstrip("/")
        self.play = play
        # 12s, not 30: a render runs at roughly half realtime, so anything
        # slower than the line itself has already missed its moment. The
        # timeout is the per-line cost of discovering the service is gone,
        # and it is paid on the publish path.
        self.timeout = timeout
        # monotonic deadline while the breaker is open; 0.0 when closed
        self._breaker_until = 0.0
        # Playback target. A recorder can only capture audio that reached a
        # SINK, so a take needs playback — but playing to the default sink
        # means a seven-minute take is audible in the room, and which device
        # that even is changes when headphones come and go. Pointing at a null
        # sink keeps the capture and drops the noise.
        self.sink = sink
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        # consecutive failed renders; drives the mute warning in speak()
        self._fails = 0
        self._seconds: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()
        self._seq = 0
        # ONE playback thread, fed by a queue. Playback used to be a bare
        # Popen per line, so every clip started the moment it finished
        # rendering: a handoff pair spoke simultaneously, and consecutive
        # beats stacked on top of each other. Heard on take 13 — they talked
        # over each other AND over themselves. A clip must never begin before
        # the previous one ends.
        self._plays: queue.Queue = queue.Queue()
        self._player = threading.Thread(target=self._play_loop, daemon=True)
        self._player.start()

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
        # Breaker open: skip the call entirely rather than pay the timeout
        # again. The caller falls back to the reading-rate estimate, so the
        # duo degrades to text-only AT THE RIGHT MOMENTS instead of dragging
        # the whole broadcast behind the picture.
        now = time.monotonic()
        if self._breaker_until:
            if now < self._breaker_until:
                return None
            # cooldown expired: probe cheaply before reopening the tap, so a
            # still-dead service costs 2s, not a full render timeout
            if not self.available():
                self._breaker_until = now + self.BREAKER_COOLDOWN
                return None
            print("speech: service is back — resuming audio", flush=True)
            self._breaker_until = 0.0
            self._fails = 0
        body = json.dumps({"persona": persona, "text": line,
                           "register": register}).encode()
        req = urllib.request.Request(
            f"{self.url}/speak", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                audio = r.read()
                seconds = float(r.headers.get("X-Speech-Seconds") or 0) or None
        except (urllib.error.URLError, OSError, ValueError) as e:
            # LOUDLY. Returning None silently is how the duo went mute for
            # the last six beats of take 80 — text commentary kept flowing
            # and the video simply had no voices from T17 to the wrap-up,
            # with not one line in any log to say so. Silence beats losing
            # the line, but silent silence is undiagnosable.
            self._fails += 1
            if self._fails == 1 or self._fails % 5 == 0:
                print(f"speech: render FAILED ({self._fails} so far) — "
                      f"{e!r}; the duo is MUTE until this recovers",
                      flush=True)
            if self._fails >= self.BREAKER_AFTER and not self._breaker_until:
                self._breaker_until = (time.monotonic()
                                       + self.BREAKER_COOLDOWN)
                print(f"speech: breaker OPEN after {self._fails} failures — "
                      f"skipping TTS for {self.BREAKER_COOLDOWN:.0f}s so the "
                      f"commentary stays in sync with the picture",
                      flush=True)
            return None
        if seconds is None:
            self._fails += 1
            print("speech: service returned no duration — treating as a "
                  "failed render; the duo is MUTE for this line", flush=True)
            return None
        if self._fails:
            print(f"speech: recovered after {self._fails} failed render(s)",
                  flush=True)
            self._fails = 0
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
        """Hand the clip to the player thread. Never plays inline: overlapping
        speech is worse than late speech."""
        self._plays.put((audio, path))

    def _play_loop(self) -> None:
        """Play queued clips strictly one at a time, blocking on each."""
        while True:
            audio, path = self._plays.get()
            cmd = ["paplay"]
            if self.sink:
                cmd.append(f"--device={self.sink}")
            try:
                if path is not None:
                    subprocess.run(cmd + [str(path)],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, check=False)
                else:
                    subprocess.run(cmd, input=audio,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, check=False)
            except Exception as e:
                # a missing player must not kill the thread — but it must
                # not take the audio down silently either (take 80)
                print(f"speech: playback failed — {e!r}", flush=True)
            finally:
                self._plays.task_done()

    def backlog(self) -> int:
        """Clips still waiting to play. A non-zero backlog means anything new
        would land on top of speech already in flight."""
        return self._plays.qsize()
