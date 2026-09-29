#!/bin/bash
# Usage: ./stop.sh [instance_id]  (matches run.sh)
INSTANCE_ID="${1:-}"
NAME="paratrooper${INSTANCE_ID:+-$INSTANCE_ID}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "Stopped $NAME."
