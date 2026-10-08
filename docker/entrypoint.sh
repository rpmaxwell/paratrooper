#!/bin/bash
set -e

XVFB_WHD="${XVFB_WHD:-1024x768x24}"
DISPLAY="${DISPLAY:-:99}"
export DISPLAY

# a container restarted after a crash keeps /tmp: drop the dead server's
# lock, or Xvfb refuses to start ("Server is already active for display")
DISPLAY_NUM="${DISPLAY#:}"
rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}"

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

if [ -z "${VNC_PASSWORD:-}" ]; then
    echo "VNC_PASSWORD is not set (run.sh passes it from your environment or .env)" >&2
    exit 1
fi
# store it as an x11vnc password file so it never appears on a command line
x11vnc -storepasswd "$VNC_PASSWORD" /tmp/vncpasswd >/dev/null 2>&1
unset VNC_PASSWORD
x11vnc -display "$DISPLAY" -forever -shared -rfbauth /tmp/vncpasswd -quiet -bg -o /tmp/x11vnc.log

export HOME=/root
mkdir -p /root
dosbox-x -conf /config/dosbox-x.conf &
DOSBOX_PID=$!

trap 'kill $DOSBOX_PID $XVFB_PID 2>/dev/null' EXIT TERM INT

wait $DOSBOX_PID
