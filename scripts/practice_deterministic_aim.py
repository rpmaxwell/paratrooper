#!/usr/bin/env python3
"""Validation gate for aim_solver.py: repeatedly pick a clean, isolated
falling paratrooper, compute the deterministic intercept, execute it, and
check score before/after for a hit. Reports the hit rate over N attempts.
Run against a dedicated/idle instance -- this sends real key input.
"""
import subprocess
import sys
import time
from collections import deque

import mss
import numpy as np

import time as _time

from actions import do_action
from read_score import is_done, read_score
from threats import detect_barrel_angle, threat_points
from aim_solver import (
    LATENCY_OFFSET_S,
    PIVOT,
    ROTATE_LEFT_LIMIT_DEG,
    ROTATE_RIGHT_LIMIT_DEG,
    ROTATION_SPEED_DEG_S,
    TICK_SAFETY_MARGIN_S,
    _target_angle_at,
    solve_intercept,
    valid_barrel_angle,
)

TARGET_Y_MIN = 250  # exclude the helicopter band
# threat_points() only excludes y>=555, but a landed walker's legs merge
# into the ground line and produce cyan pixels as high as y=553-554 --
# just under that cutoff -- which then get mistaken for a still-falling
# target, causing the solver to spin fruitlessly trying to intercept a
# stationary ground artifact. Diagnosed via diagnose_reaction_latency.py:
# this was the real cause of multi-hundred-ms to multi-second "reaction
# delay" observed watching over VNC before this fix.
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


# A single isolated trooper's cyan footprint is consistently <=~11px tall
# (characterize_sprite.py: 90th percentile height 15, and that stable
# shape is identical across every clean sample -- top bar, narrow torso,
# split legs). Two troopers close enough vertically to fall within the
# cluster() radius merge into one blob whose centroid sits in the empty
# space between them, not on either trooper -- reject anything taller
# than a single trooper could plausibly be rather than aim at that
# corrupted centroid.
MAX_BLOB_HEIGHT = 20


def find_target(frame):
    """Picks the topmost (freshest, most time-margin) clean, single-trooper-
    sized cluster -- skips anything tall enough to be two merged troopers."""
    pts = [(x, y) for x, y in threat_points(frame) if TARGET_Y_MIN <= y <= TARGET_Y_MAX]
    if not pts:
        return None
    blobs = cluster(pts)
    valid = [b for b in blobs if max(p[1] for p in b) - min(p[1] for p in b) <= MAX_BLOB_HEIGHT]
    if not valid:
        return None
    target = min(valid, key=lambda blob: min(p[1] for p in blob))
    xs = [p[0] for p in target]
    ys = [p[1] for p in target]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def main(n_attempts=15):
    results = []
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        while len(results) < n_attempts:
            frame = grab(sct, monitor)
            if is_done(frame):
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.3)
                continue

            target = find_target(frame)
            if target is None:
                do_action(0)
                time.sleep(0.1)
                continue

            barrel_angle = detect_barrel_angle(frame)
            if not valid_barrel_angle(barrel_angle):
                time.sleep(0.05)
                continue

            tx, ty = target
            t_capture = time.time()
            solution = solve_intercept(barrel_angle, tx, ty)
            if solution is None:
                do_action(0)
                time.sleep(0.1)
                continue

            direction, hold_s = solution
            prev_score = read_score(frame)

            # Manual step-through (instead of aim_solver.execute_intercept)
            # so we can diagnose rotation-model error separately from
            # target/bullet-model error: capture the barrel's *actual*
            # angle right before firing, and compare it to (a) what the
            # model predicted the barrel would be at, and (b) where the
            # model assumed the target still was at fire time.
            if direction != 0:
                do_action(1 if direction > 0 else 2)
                sleep_s = max(0.0, hold_s - LATENCY_OFFSET_S + TICK_SAFETY_MARGIN_S)
            else:
                sleep_s = 0.0
            if sleep_s > 0:
                time.sleep(sleep_s)
            t_prefire = time.time()
            prefire_frame = grab(sct, monitor)
            actual_barrel_at_fire = detect_barrel_angle(prefire_frame)
            predicted_barrel_at_fire = min(
                max(barrel_angle + direction * ROTATION_SPEED_DEG_S * hold_s, ROTATE_RIGHT_LIMIT_DEG),
                ROTATE_LEFT_LIMIT_DEG,
            )
            predicted_target_angle_raw = _target_angle_at(tx, ty, hold_s)
            do_action(3)
            exec_time = t_prefire - t_capture

            time.sleep(0.15)
            frame2 = grab(sct, monitor)
            new_score = read_score(frame2)
            delta = new_score - prev_score
            fire_cost = -1.0 if prev_score > 0 else 0.0
            hit_reward = delta - fire_cost
            hit = hit_reward > 0
            results.append(hit)
            rot_err = (
                None
                if actual_barrel_at_fire is None
                else actual_barrel_at_fire - predicted_barrel_at_fire
            )
            aim_err = (
                None
                if actual_barrel_at_fire is None
                else actual_barrel_at_fire - predicted_target_angle_raw
            )
            print(
                f"attempt {len(results):2d}: target=({tx:6.1f},{ty:6.1f}) barrel0={barrel_angle:6.1f} "
                f"dir={direction:+d} hold={hold_s:.3f}s pre_exec={exec_time:.3f}s "
                f"rot_err={rot_err} aim_err={aim_err} "
                f"score {prev_score}->{new_score} hit={hit}"
            )
            time.sleep(0.3)

    n_hits = sum(results)
    print(f"\n{n_hits}/{len(results)} hits ({100 * n_hits / len(results):.0f}%)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(n)
