#!/usr/bin/env python3
"""Diagnose the "noticeable delay before the gun reacts" the user
observed watching over VNC. calibrate_latency.py only measured the
Python-side decision overhead in an idealized single-shot test (~32ms) --
this instead runs the *actual* practice loop, unmodified in structure,
and logs every rejected iteration (target not found / invalid barrel /
no solution) with its own timestamp, so a real multi-hundred-ms stall
shows up as a visible run of rejections rather than being averaged away.
"""
import subprocess
import sys
import time
from collections import deque

import mss
import numpy as np

from actions import do_action
from read_score import is_done
from threats import detect_barrel_angle, threat_points
from aim_solver import execute_intercept, solve_intercept, valid_barrel_angle

TARGET_Y_MIN = 250
# threat_points() only excludes y>=555, but a landed walker's legs merge
# into the ground line and produce cyan pixels as high as y=553-554 --
# just under that cutoff -- which then get mistaken for a still-falling
# target, causing the solver to spin fruitlessly trying to intercept a
# stationary ground artifact (this was the real cause of the multi-
# hundred-ms to multi-second "reaction delay" observed watching over VNC).
# A real target still meaningfully airborne is well clear of this band.
TARGET_Y_MAX = 540


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


def find_target(frame):
    pts = [(x, y) for x, y in threat_points(frame) if TARGET_Y_MIN <= y <= TARGET_Y_MAX]
    if not pts:
        return None
    blobs = cluster(pts)
    target = min(blobs, key=lambda blob: min(p[1] for p in blob))
    xs = [p[0] for p in target]
    ys = [p[1] for p in target]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def main(n_engagements=20):
    engagements = 0
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        was_tracking = False  # were we mid-engagement last iteration?
        t_first_seen = None
        rejects_this_engagement = []

        while engagements < n_engagements:
            t_iter = time.time()
            frame = grab(sct, monitor)
            if is_done(frame):
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.3)
                was_tracking = False
                continue

            target = find_target(frame)
            if target is None:
                if was_tracking:
                    was_tracking = False  # target vanished (landed/left screen) without a shot
                do_action(0)
                time.sleep(0.03)
                continue

            if not was_tracking:
                t_first_seen = t_iter
                rejects_this_engagement = []
                was_tracking = True

            barrel_angle = detect_barrel_angle(frame)
            if not valid_barrel_angle(barrel_angle):
                rejects_this_engagement.append((time.time() - t_first_seen, "invalid_barrel_angle", barrel_angle))
                time.sleep(0.02)
                continue

            tx, ty = target
            solution = solve_intercept(barrel_angle, tx, ty)
            if solution is None:
                rejects_this_engagement.append((time.time() - t_first_seen, "no_solution", barrel_angle, tx, ty))
                do_action(0)
                time.sleep(0.03)
                continue

            direction, hold_s = solution
            t_action = time.time()
            reaction_delay = t_action - t_first_seen
            execute_intercept(direction, hold_s)
            was_tracking = False
            engagements += 1
            print(
                f"engagement {engagements:2d}: reaction_delay={reaction_delay*1000:6.1f}ms "
                f"target=({tx:.0f},{ty:.0f}) rejects={len(rejects_this_engagement)} "
                f"{rejects_this_engagement if rejects_this_engagement else ''}"
            )
            time.sleep(0.05)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n)
