#!/bin/bash
set -e

XVFB_WHD="${XVFB_WHD:-1024x768x24}"
DISPLAY="${DISPLAY:-:99}"
export DISPLAY

Xvfb "$DISPLAY" -screen 0 "$XVFB_WHD" -nolisten tcp &
XVFB_PID=$!

for i in $(seq 1 50); do
    if xdotool getdisplaygeometry >/dev/null 2>&1; then
        break
    fi
    sleep 0.2
done

if ! xdotool getdisplaygeometry >/dev/null 2>&1; then
    echo "Xvfb failed to start on $DISPLAY" >&2
    exit 1
fi

mkdir -p /work/game
if [ ! -f /work/game/PARATROO.COM ]; then
    cp /assets/ParaTrooper.1982.com /work/game/PARATROO.COM
fi

x11vnc -display "$DISPLAY" -forever -shared -passwd paratrooper -quiet -bg -o /tmp/x11vnc.log

export HOME=/root
mkdir -p /root
dosbox-x -conf /config/dosbox-x.conf &
DOSBOX_PID=$!

trap 'kill $DOSBOX_PID $XVFB_PID 2>/dev/null' EXIT TERM INT

wait $DOSBOX_PID
