#!/bin/bash
# Prism commentary panel: a normal TILED kitty window (not a floating
# overlay). On a tiling WM the clean answer is a panel — Hyprland tiles it
# next to the battle browser, wf-recorder captures the whole workspace, and
# there's no float/pin/transparency/click-through to fight.
#
# Launches a FRESH kitty instance (KITTY_* env unset) so it maps as its own
# window even when started from inside another kitty session, and reads the
# caption feed (commentary_overlay.py). Class "prism-panel".
#
#   hyprctl dispatch exec -- bash showdown/overlay_kitty.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../.venv/bin/python"   # needs websockets (project venv has it)
[ -x "$PY" ] || PY=python3

exec env -u KITTY_LISTEN_ON -u KITTY_PID -u KITTY_WINDOW_ID -u KITTY_PUBLIC_KEY \
  kitty \
    --class prism-panel \
    --title prism-panel \
    -o background=#0a0d15 \
    -o foreground=#f2f5fb \
    -o font_size=23 \
    -o window_padding_width=16 \
    -o cursor_blink_interval=0 \
    -o confirm_os_window_close=0 \
    -o remember_window_size=no \
    "$PY" "$HERE/overlay_kitty.py"
