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

BROADCAST MODE: request /broadcast-client.html and you get testclient-new.html
with broadcast_client.css and a sizing script injected, which strips the client
down to just the battle scene. Injecting here (rather than editing the client)
keeps the fork pristine, and sidesteps two problems: testclient-new.html
rejects any query string that isn't `?~~host:port`, so a ?broadcast=1 flag is
not available; and the page is cross-origin from broadcast.html on :8129, so
it cannot be styled from there. The name still ends in .html, which is what
makes PSRouter hash-route to the room.

Run:  python showdown/serve_client.py [--port 8127] [--root <client dir>]
Requires `./build full` in pokemon-showdown-client — a plain `./build` leaves
data/ empty, the client async-falls-back to the CDN for pokedex/graphics, and
the battle panel throws in BattleScene before they land.
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
BROADCAST_PATH = "/broadcast-client.html"
BROADCAST_CSS_PATH = "/broadcast-client.css"
BROADCAST_SOURCE = "testclient-new.html"
HERE = Path(__file__).parent

# scale the fixed 640x360 scene to fill the frame; a CSS-only version isn't
# possible because scale() needs a unitless number and calc() on vw yields a
# length. Set as a custom property so the stylesheet owns the actual rule.
SIZER = """
<link rel="stylesheet" href="%s">
<script>
(function () {
	function fit() {
		var s = Math.min(window.innerWidth / 640, window.innerHeight / 360);
		document.documentElement.style.setProperty('--battle-scale', s);
	}
	window.addEventListener('resize', fit);
	fit();
})();
</script>
""" % BROADCAST_CSS_PATH


class ClientHandler(SimpleHTTPRequestHandler):
    """Static files, with an SPA-style fallback so room paths reach the
    client instead of 404ing, plus the injected broadcast entry point."""

    def do_GET(self):
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path == BROADCAST_CSS_PATH:
            return self._send_bytes(
                (HERE / "broadcast_client.css").read_bytes(), "text/css")
        if path == BROADCAST_PATH:
            return self._send_bytes(self._broadcast_html(), "text/html")
        return super().do_GET()

    def _broadcast_html(self) -> bytes:
        src = (Path(self.directory) / BROADCAST_SOURCE).read_text()
        if "</head>" not in src:
            raise RuntimeError(f"no </head> in {BROADCAST_SOURCE}")
        return src.replace("</head>", SIZER + "</head>", 1).encode()

    def _send_bytes(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

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
