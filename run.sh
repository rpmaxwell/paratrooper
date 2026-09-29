#!/bin/bash
# Build and (re)start a Paratrooper DOSBox-X container.
# Usage: ./run.sh [instance_id]
#   ./run.sh          -> container "paratrooper", VNC on 5900
#   ./run.sh 1        -> container "paratrooper-1", VNC on 5901
# Each instance is fully independent (own Xvfb/dosbox-x/x11vnc), for
# Stage 5.2 parallel-instance throughput testing.
# VNC (view + play manually): open vnc://localhost:<port> (macOS Screen Sharing)
set -e
cd "$(dirname "$0")"

INSTANCE_ID="${1:-}"
if [ -n "$INSTANCE_ID" ]; then
    NAME="paratrooper-${INSTANCE_ID}"
    VNC_PORT=$((5900 + INSTANCE_ID))
    CAPTURES_DIR="$(pwd)/captures/instance-${INSTANCE_ID}"
else
    NAME="paratrooper"
    VNC_PORT=5900
    CAPTURES_DIR="$(pwd)/captures"
fi

docker build -t paratrooper-dosbox -f docker/Dockerfile docker/

docker rm -f "$NAME" >/dev/null 2>&1 || true

mkdir -p "$CAPTURES_DIR"

docker run -d --name "$NAME" \
    -p "${VNC_PORT}:5900" \
    -v ~/Documents/paratrooper/ParaTrooper.1982.com:/assets/ParaTrooper.1982.com:ro \
    -v "$(pwd)/docker/dosbox-x.conf:/config/dosbox-x.conf:ro" \
    -v "${CAPTURES_DIR}:/captures" \
    -v "$(pwd)/scripts:/scripts:ro" \
    paratrooper-dosbox

echo "Container '$NAME' started. VNC: open vnc://localhost:${VNC_PORT}"
echo "Run scripts with: docker exec $NAME python3 /scripts/<name>.py"
