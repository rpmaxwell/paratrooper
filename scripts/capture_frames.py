#!/usr/bin/env python3
"""Stage 1 validation: capture N frames at a fixed interval, save as PNGs,
and report timing so we can confirm frames aren't torn and timing is
consistent."""
import argparse
import time

import mss
import mss.tools

parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=10)
parser.add_argument("--interval", type=float, default=0.5, help="seconds between frames")
parser.add_argument("--outdir", default="/captures")
args = parser.parse_args()

with mss.mss() as sct:
    monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
    timestamps = []
    for i in range(args.count):
        t0 = time.time()
        img = sct.grab(monitor)
        mss.tools.to_png(img.rgb, img.size, output=f"{args.outdir}/frame_{i:03d}.png")
        timestamps.append(time.time())
        elapsed = time.time() - t0
        sleep_left = args.interval - elapsed
        if sleep_left > 0:
            time.sleep(sleep_left)

deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
print(f"captured {len(timestamps)} frames to {args.outdir}")
if deltas:
    print(f"inter-frame delta: min={min(deltas):.4f}s max={max(deltas):.4f}s avg={sum(deltas)/len(deltas):.4f}s")
