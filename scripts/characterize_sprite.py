#!/usr/bin/env python3
"""Characterize the falling-trooper sprite's pixel footprint across many
samples: bounding box size, where the raw centroid sits within that box,
and how stable any of this is -- to find a more reliable aim-point
definition than the raw cyan-pixel centroid (which can be pulled around
by a few animation-dependent pixels on an 8-15px-tall sprite).

Passive: no aiming/firing, just watches and logs isolated single-blob
targets (skips anything ambiguous/multi-blob) across many falls,
including free-fall vs. canopy-phase (split by y relative to
aim_solver.FALL_BREAK_Y) since the sprite shape likely differs between
phases.
"""
import subprocess
import sys
import time
from collections import deque

import mss
import numpy as np

from actions import do_action
from read_score import is_done
from threats import threat_points
from aim_solver import FALL_BREAK_Y

TARGET_Y_MIN = 250
TARGET_Y_MAX = 540
OUT_PATH = "/captures/sprite_samples.json"


def grab(sct, monitor):
    return np.array(sct.grab(monitor))[:, :, :3][:, :, ::-1]


def cluster(points, radius=15):
    remaining = set(points)
    clusters = []
    while remaining:
        seed = next(iter(remaining))
        remaining.discard(seed)
        q = deque([seed])
        blob = [seed]
        while q:
            cx, cy = q.popleft()
            near = [p for p in remaining if abs(p[0] - cx) <= radius and abs(p[1] - cy) <= radius]
            for p in near:
                remaining.discard(p)
                q.append(p)
                blob.append(p)
        clusters.append(blob)
    return clusters


def isolated_blob(frame):
    """Returns the single blob's raw points if there's EXACTLY one
    falling-trooper blob on screen right now, else None -- avoids any
    ambiguity about which blob is which."""
    pts = [(x, y) for x, y in threat_points(frame) if TARGET_Y_MIN <= y <= TARGET_Y_MAX]
    if not pts:
        return None
    blobs = cluster(pts)
    if len(blobs) != 1:
        return None
    return blobs[0]


def main(duration_s=90.0, poll=0.05):
    import json
    samples = []
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        t0 = time.time()
        step = 0
        while time.time() - t0 < duration_s:
            frame = grab(sct, monitor)
            if is_done(frame):
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.3)
                continue
            blob = isolated_blob(frame)
            if blob is not None:
                xs = [p[0] for p in blob]
                ys = [p[1] for p in blob]
                samples.append({
                    "pixels": blob,
                    "centroid": [sum(xs) / len(xs), sum(ys) / len(ys)],
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "n": len(blob),
                })
            if step % 8 == 0:
                do_action(3)  # keep the round alive
            else:
                do_action(0)
            step += 1
            time.sleep(poll)

    with open(OUT_PATH, "w") as f:
        json.dump(samples, f)
    print(f"saved {len(samples)} isolated single-blob samples to {OUT_PATH}")

    # summarize
    free = [s for s in samples if s["centroid"][1] < FALL_BREAK_Y]
    canopy = [s for s in samples if s["centroid"][1] >= FALL_BREAK_Y]
    for label, group in [("free-fall", free), ("canopy", canopy)]:
        if not group:
            print(f"{label}: no samples")
            continue
        widths = np.array([b["bbox"][2] - b["bbox"][0] for b in group])
        heights = np.array([b["bbox"][3] - b["bbox"][1] for b in group])
        ns = np.array([b["n"] for b in group])
        # fraction of the way down the bbox the centroid sits
        frac_y = np.array([
            (b["centroid"][1] - b["bbox"][1]) / max(b["bbox"][3] - b["bbox"][1], 1e-6)
            for b in group
        ])
        print(f"\n{label} (n={len(group)}):")
        print(f"  width:  mean={widths.mean():.2f} std={widths.std():.2f} min={widths.min()} max={widths.max()}")
        print(f"  height: mean={heights.mean():.2f} std={heights.std():.2f} min={heights.min()} max={heights.max()}")
        print(f"  pixel count: mean={ns.mean():.2f} std={ns.std():.2f} min={ns.min()} max={ns.max()}")
        print(f"  centroid y as fraction of bbox height: mean={frac_y.mean():.2f} std={frac_y.std():.2f}")


if __name__ == "__main__":
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    main(duration)
