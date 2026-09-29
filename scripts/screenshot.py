#!/usr/bin/env python3
"""Stage 0 validation: grab one screenshot of the Xvfb display."""
import sys
import time
import mss

out = sys.argv[1] if len(sys.argv) > 1 else "/captures/screenshot.png"

with mss.mss() as sct:
    monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
    img = sct.grab(monitor)
    mss.tools.to_png(img.rgb, img.size, output=out)

print(f"saved {out} at {time.time()}")
