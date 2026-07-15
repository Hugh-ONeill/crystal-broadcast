#!/bin/bash
# Bring up the Prism commentary panel stack:
#   * commentary_overlay.py  — feed server (AIRI -> ws://127.0.0.1:8130 + http)
#   * overlay_kitty.sh       — the tiled caption panel (kitty, class prism-panel)
#
# Replaces prism_relay.py for a recording: no Showdown login, no server patch.
# It's a normal TILED window — arrange it with the battle browser however you
# like (e.g. browser large, panel as a bottom strip) and wf-recorder captures
# the workspace. Re-runnable: skips pieces already up.
#
# Launch from a normal shell or a Hyprland keybind so the windows persist:
#   bash showdown/overlay_start.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../.venv/bin/python"
[ -x "$PY" ] || PY=python3
LOGDIR="${TMPDIR:-/tmp}"

# 1) feed server (detached)
if ss -tln 2>/dev/null | grep -q ':8130'; then
  echo "feed server: already up"
else
  setsid "$PY" "$HERE/commentary_overlay.py" </dev/null \
    >"$LOGDIR/prism-overlay-feed.log" 2>&1 &
  disown
  for _ in $(seq 1 40); do ss -tln 2>/dev/null | grep -q ':8130' && break; sleep 0.1; done
  echo "feed server: started (log $LOGDIR/prism-overlay-feed.log)"
fi

# 2) caption panel (fresh kitty instance, tiled by the WM)
if hyprctl clients -j 2>/dev/null | grep -q '"class": "prism-panel"'; then
  echo "caption panel: already up"
else
  hyprctl dispatch exec -- bash "$HERE/overlay_kitty.sh" >/dev/null 2>&1
  echo "caption panel: launched"
fi
