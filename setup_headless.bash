#!/bin/bash
set -x
export DISPLAY=:99.0
which Xvfb
Xvfb :99 -screen 0 2024x2024x24 > /dev/null 2>&1 &
# sleep 3
set +x
exec "$@"