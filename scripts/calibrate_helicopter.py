#!/usr/bin/env python3
"""Stage 1.1: measure helicopter horizontal speed and altitude band,
passively (no aiming/firing needed for the tracking itself; fire
occasionally to keep the round alive). Tracks cyan blobs in the
helicopter y-band (above FALLING_Y_MIN, below the title-screen exclusion)
across frames the same way calibrate_fall_speed.py tracks troopers.
"""
import subprocess
import sys
import time
from collections import deque

import mss
import numpy as np

from actions import do_action
from read_score import is_done
from threats import CYAN, _mask_coords

HELI_Y0, HELI_Y1 = 184, 250  # above the falling-trooper band
GATE_RADIUS = 40.0  # helicopters move faster than troopers between polls
MAX_MISS_S = 0.3
MIN_TRACK_POINTS = 10
MIN_TRACK_DURATION_S = 0.5


def grab(sct, monitor):
    return np.array(sct.grab(monitor))[:, :, :3][:, :, ::-1]


def cluster(points, radius=20):
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


def heli_centroids(frame):
    xs, ys = _mask_coords(frame, CYAN, y0=HELI_Y0, y1=HELI_Y1)
    pts = list(zip(xs.tolist(), ys.tolist()))
    if not pts:
        return []
    blobs = cluster(pts)
    # a real helicopter blob should have a decent number of pixels; drop tiny noise
    blobs = [b for b in blobs if len(b) >= 5]
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


def finalize(track):
    if len(track.points) < MIN_TRACK_POINTS:
        return None
    ts = np.array([p[0] for p in track.points])
    xs = np.array([p[1] for p in track.points])
    ys = np.array([p[2] for p in track.points])
    duration = ts[-1] - ts[0]
    if duration < MIN_TRACK_DURATION_S:
        return None
    x_slope, x_int = np.polyfit(ts, xs, 1)
    y_mean, y_std = ys.mean(), ys.std()
    resid = xs - (x_slope * ts + x_int)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return {
        "n": len(track.points),
        "duration": duration,
        "x_speed": x_slope,
        "y_mean": y_mean,
        "y_std": y_std,
        "x_range": (float(xs.min()), float(xs.max())),
        "rmse": rmse,
    }


def main(duration_s=90.0, poll=0.04):
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
                    res = finalize(tr)
                    if res:
                        finished.append(res)
                active_tracks = []
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.3)
                continue

            t = time.time() - t0
            centroids = heli_centroids(frame)
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
                    res = finalize(tr)
                    if res:
                        finished.append(res)
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
            res = finalize(tr)
            if res:
                finished.append(res)

    print(f"\n{len(finished)} helicopter tracks:")
    for i, r in enumerate(finished):
        print(
            f"track {i:2d}: n={r['n']:3d} dur={r['duration']:.2f}s "
            f"x_speed={r['x_speed']:+7.2f} px/s y_mean={r['y_mean']:6.1f} y_std={r['y_std']:.2f} "
            f"x_range={r['x_range']} rmse={r['rmse']:.2f}"
        )

    clean = [r for r in finished if r["rmse"] < 5.0]
    if clean:
        speeds = np.array([r["x_speed"] for r in clean])
        ys = np.array([r["y_mean"] for r in clean])
        print(f"\nCLEAN tracks (rmse<5.0, n={len(clean)}):")
        print(f"  |x_speed|: mean={np.abs(speeds).mean():.2f} std={np.abs(speeds).std():.2f} px/s (signed values: {sorted(speeds.round(1).tolist())})")
        print(f"  altitude y: mean={ys.mean():.2f} std={ys.std():.2f}")
    else:
        print("\nNo clean tracks -- widen GATE_RADIUS or check y-band bounds.")


if __name__ == "__main__":
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    main(duration)
