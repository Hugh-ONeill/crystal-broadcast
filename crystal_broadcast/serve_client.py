#!/usr/bin/env python3
"""Serve the self-hosted Pokemon Showdown client for the broadcast stack.

WHY THIS EXISTS: the local pokemon-showdown server on :8000 does NOT serve a
client. Its root is an 858-byte stub that redirects the browser to
`https://localhost.psim.us/<path>`, i.e. Showdown's CDN-hosted client, which
then connects back to :8000. That has three consequences we can't live with:

  1. The CDN serves the OLD client, whose index-old.html framebusts
     (`if (self === top) ... else 'IN FRAME, please visit Showdown directly'`),
     so broadcast.html can never composite an overlay over it.
  2. The battle view depends on the public internet mid-match.
  3. Cross-origin: nothing local can read the client's DOM, which blocks both
     chrome-hiding CSS and the animation-clock hook the PTS delay buffer needs.

Self-hosting pokemon-showdown-client fixes all three. The NEW client
(index-new.html) does not framebust and pulls no scripts from the CDN.

Routing: psim.us serves the client for ANY path (/battle-gen9ou-123 included)
and the client reads the room out of the URL. A plain static server 404s those,
so this one falls back to index-new.html for anything that isn't a real file.

Run:  python showdown/serve_client.py [--port 8127] [--root <client dir>]
The client's config/routes.json `client` field must match host:port, or
Config.defaultserver won't apply and it'll try to reach sim3.psim.us.
"""
from __future__ import annotations

import argparse
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_ROOT = (Path.home() / "Developer/grimoire/pokemon-showdown-client"
                / "play.pokemonshowdown.com")
FALLBACK = "index-new.html"


class ClientHandler(SimpleHTTPRequestHandler):
    """Static files, with an SPA-style fallback so room paths reach the
    client instead of 404ing."""

    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.exists(path) and not self.path.startswith("/js/"):
            # a room path (/battle-gen9ou-123, /lobby, ...) — hand it the
            # client and let its router sort it out, exactly as psim.us does
            self.path = "/" + FALLBACK
        return super().send_head()

    def log_message(self, fmt, *args):  # quiet: this runs behind a match
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8127)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--bind", default="127.0.0.1")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not (root / FALLBACK).exists():
        raise SystemExit(f"no {FALLBACK} under {root} — build the client first "
                         f"(./build in pokemon-showdown-client)")

    handler = partial(ClientHandler, directory=str(root))
    srv = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"showdown client: http://{args.bind}:{args.port}/ (root {root})",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
