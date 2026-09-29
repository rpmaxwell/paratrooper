#!/usr/bin/env python3
"""Stage 1.1: tight paratrooper fall-speed calibration across many
independent drops, tracked passively (no aiming/firing needed for the
tracking itself, though we fire occasionally to keep the round alive so
score/wave keeps advancing and we get later-wave drops too).

Clusters cyan threat pixels per-frame into blobs, associates blobs across
frames into per-trooper tracks (nearest-neighbor, small gate radius), then
linear-fits y(t) and x(t) per track to get fall speed and horizontal drift
independently for every clean, isolated drop -- not just one.

Run against the idle/dedicated instance only (sends real key input).
"""
import subprocess
import sys
import time
from collections import deque

import mss
import numpy as np

from actions import do_action
from read_score import is_done, read_score
from threats import threat_points

FALLING_Y_MIN = 250
GATE_RADIUS = 25.0  # max px a blob can move between polls and still be "the same" track
MAX_MISS_S = 0.25  # finalize a track after this long with no matching blob
MIN_TRACK_POINTS = 8
MIN_TRACK_DURATION_S = 0.8


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
    def __init__(self, t0, x0, y0, score0):
        self.points = [(t0, x0, y0)]
        self.last_seen = t0
        self.score0 = score0

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
    y_slope, y_int = np.polyfit(ts, ys, 1)
    x_slope, x_int = np.polyfit(ts, xs, 1)
    y_pred = y_slope * ts + y_int
    resid = ys - y_pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return {
        "n": len(track.points),
        "duration": duration,
        "fall_speed": y_slope,
        "x_drift": x_slope,
        "rmse": rmse,
        "y_range": (float(ys.min()), float(ys.max())),
        "score0": track.score0,
    }


def main(duration_s=90.0, poll=0.05):
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
                # finalize whatever's active before the reset wipes state
                for tr in active_tracks:
                    res = finalize(tr)
                    if res:
                        finished.append(res)
                active_tracks = []
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.3)
                continue

            t = time.time() - t0
            score = read_score(frame)
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
                    still_active.append(tr)  # brief occlusion, keep alive
                else:
                    res = finalize(tr)
                    if res:
                        finished.append(res)
            active_tracks = still_active
            for cx, cy in unmatched:
                active_tracks.append(Track(t, cx, cy, score))

            if step % 8 == 0:
                do_action(3)  # occasional fire to prevent pile-up ending the round
            else:
                do_action(0)
            step += 1
            time.sleep(poll)

        for tr in active_tracks:
            res = finalize(tr)
            if res:
                finished.append(res)

    print(f"\n{len(finished)} clean tracks:")
    for i, r in enumerate(finished):
        print(
            f"track {i:2d}: n={r['n']:3d} dur={r['duration']:.2f}s "
            f"fall_speed={r['fall_speed']:+7.2f} px/s x_drift={r['x_drift']:+6.2f} px/s "
            f"rmse={r['rmse']:.2f} y_range={r['y_range']} score0={r['score0']}"
        )

    # Filter to the cleanest tracks (low fit residual = single isolated
    # trooper, not two overlapping blobs or a helicopter-adjacent glitch).
    clean = [r for r in finished if r["rmse"] < 5.0 and r["fall_speed"] > 0]
    if clean:
        speeds = np.array([r["fall_speed"] for r in clean])
        drifts = np.array([r["x_drift"] for r in clean])
        print(f"\nCLEAN tracks (rmse<5.0, n={len(clean)}):")
        print(f"  fall_speed: mean={speeds.mean():.2f} std={speeds.std():.2f} min={speeds.min():.2f} max={speeds.max():.2f} px/s")
        print(f"  x_drift:    mean={drifts.mean():.2f} std={drifts.std():.2f} px/s")

        # Wave/score correlation: does fall speed increase with score?
        scores = np.array([r["score0"] for r in clean])
        if len(set(scores.tolist())) > 1:
            corr = np.corrcoef(scores, speeds)[0, 1]
            print(f"  score-vs-speed correlation: {corr:.3f} (score range {scores.min():.0f}-{scores.max():.0f})")
        else:
            print(f"  all clean tracks at score={scores[0]:.0f} -- no wave variation observed in this run")
    else:
        print("\nNo clean tracks found -- widen GATE_RADIUS or check for occlusion issues.")


if __name__ == "__main__":
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    main(duration)
