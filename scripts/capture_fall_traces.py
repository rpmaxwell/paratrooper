#!/usr/bin/env python3
"""Capture raw (t, x, y) point sequences for individual falling-trooper
tracks and dump them as JSON, so we can fit a two-phase (fast free-fall,
then slow canopy-open descent) model offline instead of assuming one
constant velocity for the whole fall.
"""
import json
import subprocess
import sys
import time
from collections import deque

import mss
import numpy as np

from actions import do_action
from read_score import is_done
from threats import threat_points

FALLING_Y_MIN = 250
GATE_RADIUS = 25.0
MAX_MISS_S = 0.25
MIN_TRACK_POINTS = 15


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


def blob_centroids(frame):
    pts = [(x, y) for x, y in threat_points(frame) if y >= FALLING_Y_MIN]
    if not pts:
        return []
    blobs = cluster(pts)
    return [(sum(x for x, _ in b) / len(b), sum(y for _, y in b) / len(b)) for b in blobs]


class Track:
    def __init__(self, t0, x0, y0):
        self.points = [(t0, x0, y0)]
        self.last_seen = t0

    def extend(self, t, x, y):
        self.points.append((t, x, y))
        self.last_seen = t

    def last_xy(self):
        return self.points[-1][1], self.points[-1][2]


def main(duration_s=90.0, poll=0.03, outpath="/captures/fall_traces.json"):
    active_tracks = []
    finished = []
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        frame = grab(sct, monitor)
        if is_done(frame):
            subprocess.run(["xdotool", "key", "space"], check=True)
            time.sleep(0.3)

        t0 = time.time()
        step = 0
        while time.time() - t0 < duration_s:
            frame = grab(sct, monitor)
            if is_done(frame):
                for tr in active_tracks:
                    if len(tr.points) >= MIN_TRACK_POINTS:
                        finished.append(tr.points)
                active_tracks = []
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.3)
                continue

            t = time.time() - t0
            centroids = blob_centroids(frame)
            unmatched = list(centroids)
            still_active = []
            for tr in active_tracks:
                lx, ly = tr.last_xy()
                best_i, best_d = None, GATE_RADIUS
                for i, (cx, cy) in enumerate(unmatched):
                    d = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                    if d < best_d:
                        best_i, best_d = i, d
                if best_i is not None:
                    cx, cy = unmatched.pop(best_i)
                    tr.extend(t, cx, cy)
                    still_active.append(tr)
                elif t - tr.last_seen < MAX_MISS_S:
                    still_active.append(tr)
                else:
                    if len(tr.points) >= MIN_TRACK_POINTS:
                        finished.append(tr.points)
            active_tracks = still_active
            for cx, cy in unmatched:
                active_tracks.append(Track(t, cx, cy))

            if step % 8 == 0:
                do_action(3)
            else:
                do_action(0)
            step += 1
            time.sleep(poll)

        for tr in active_tracks:
            if len(tr.points) >= MIN_TRACK_POINTS:
                finished.append(tr.points)

    with open(outpath, "w") as f:
        json.dump(finished, f)
    print(f"saved {len(finished)} tracks to {outpath}")
    for i, pts in enumerate(finished):
        ys = [p[2] for p in pts]
        print(f"  track {i}: n={len(pts)} y_range=({min(ys):.0f},{max(ys):.0f}) t_range=({pts[0][0]:.2f},{pts[-1][0]:.2f})")


if __name__ == "__main__":
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    main(duration)
