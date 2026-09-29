#!/usr/bin/env python3
"""Ground-truth aim error measurement, bypassing the fall/rotation model
entirely: at the exact pre-fire frame, detect BOTH the actual barrel angle
and the actual target position directly, and compute the true angular
miss = actual_barrel_at_fire - angle_to(PIVOT, actual_target_at_fire).

This is deliberately model-free -- it doesn't matter whether our fall
speed, decision latency, or rotation speed constants are exactly right;
it just asks "at the instant we fired, how far off was the barrel from
truly pointing at wherever the target really was in that same frame."
That isolates hitbox width / pure execution error from every calibration
assumption upstream of it.

Still applies the same deliberate angular offsets as calibrate_hitbox.py
so we can compare "intended offset" vs "true measured miss" directly.
"""
import json
import subprocess
import sys
import time
from collections import deque, defaultdict

import mss
import numpy as np

from actions import do_action
from read_score import is_done, read_score
from threats import PIVOT, angle_to, detect_barrel_angle, threat_points
from aim_solver import (
    LATENCY_OFFSET_S,
    ROTATION_SPEED_DEG_S,
    TICK_SAFETY_MARGIN_S,
    _clip,
    _fallen_y,
    solve_intercept,
    valid_barrel_angle,
)

TARGET_Y_MIN = 250
TARGET_Y_MAX = 540  # excludes landed-walker/ground-line merge artifacts (see calibrate_hitbox.py)

OFFSETS_DEG = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
OUT_PATH = "/captures/true_miss_trials.json"
MAX_TRACK_DIST = 80.0  # max px a blob may have moved from its predicted spot and still count as the same target


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


MAX_BLOB_HEIGHT = 20  # see practice_deterministic_aim.py -- rejects merged multi-trooper blobs


def all_targets(frame):
    pts = [(x, y) for x, y in threat_points(frame) if TARGET_Y_MIN <= y <= TARGET_Y_MAX]
    if not pts:
        return []
    blobs = cluster(pts)
    out = []
    for b in blobs:
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        if max(ys) - min(ys) > MAX_BLOB_HEIGHT:
            continue
        out.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return out


def find_target(frame):
    """Topmost (freshest) target, for the initial lock-on."""
    targets = all_targets(frame)
    if not targets:
        return None
    return min(targets, key=lambda t: t[1])


def find_nearest(frame, expected_xy, max_dist=MAX_TRACK_DIST):
    """The target closest to an expected position, for re-locating the
    SAME target later -- more robust than "topmost" once other troopers
    may be in view."""
    targets = all_targets(frame)
    if not targets:
        return None
    ex, ey = expected_xy
    best, best_d = None, max_dist
    for tx, ty in targets:
        d = ((tx - ex) ** 2 + (ty - ey) ** 2) ** 0.5
        if d < best_d:
            best, best_d = (tx, ty), d
    return best


def biased_hold(barrel_angle, direction, hold_s, offset_deg):
    base_final_angle = _clip(barrel_angle + direction * ROTATION_SPEED_DEG_S * hold_s)
    desired_angle = _clip(base_final_angle + offset_deg)
    delta = desired_angle - barrel_angle
    if abs(delta) < 1e-6:
        return 0, 0.0
    new_direction = 1 if delta > 0 else -1
    new_hold = abs(delta) / ROTATION_SPEED_DEG_S
    return new_direction, new_hold


def main(n_per_offset=15):
    offset_cycle = list(OFFSETS_DEG)
    trials = []
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        oi = 0
        total_needed = n_per_offset * len(offset_cycle)
        while len(trials) < total_needed:
            frame = grab(sct, monitor)
            if is_done(frame):
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.3)
                continue

            target = find_target(frame)
            if target is None:
                do_action(0)
                time.sleep(0.05)
                continue

            barrel0 = detect_barrel_angle(frame)
            if not valid_barrel_angle(barrel0):
                time.sleep(0.03)
                continue

            tx, ty = target
            solution = solve_intercept(barrel0, tx, ty)
            if solution is None:
                do_action(0)
                time.sleep(0.05)
                continue

            direction, hold_s = solution
            offset_deg = offset_cycle[oi % len(offset_cycle)]
            oi += 1
            b_direction, b_hold_s = biased_hold(barrel0, direction, hold_s, offset_deg)

            prev_score = read_score(frame)
            if b_direction != 0:
                do_action(1 if b_direction > 0 else 2)
                sleep_s = max(0.0, b_hold_s - LATENCY_OFFSET_S + TICK_SAFETY_MARGIN_S)
            else:
                sleep_s = 0.0
            if sleep_s > 0:
                time.sleep(sleep_s)

            prefire_frame = grab(sct, monitor)
            actual_barrel_at_fire = detect_barrel_angle(prefire_frame)
            # where the model *expects* the same target to be right now --
            # used only to re-locate the correct blob, not for the error calc
            expected_xy = (tx, _fallen_y(ty, b_hold_s))
            actual_target_at_fire = find_nearest(prefire_frame, expected_xy)

            do_action(3)
            time.sleep(0.15)
            frame2 = grab(sct, monitor)
            new_score = read_score(frame2)
            delta = new_score - prev_score
            fire_cost = -1.0 if prev_score > 0 else 0.0
            hit = (delta - fire_cost) > 0

            true_required_angle = None
            true_miss_deg = None
            if actual_target_at_fire is not None and actual_barrel_at_fire is not None:
                true_required_angle = angle_to(PIVOT, actual_target_at_fire)
                true_miss_deg = actual_barrel_at_fire - true_required_angle

            trials.append({
                "offset_deg": offset_deg,
                "barrel0": barrel0,
                "target0": [tx, ty],
                "expected_xy": list(expected_xy),
                "actual_target_at_fire": list(actual_target_at_fire) if actual_target_at_fire else None,
                "actual_barrel_at_fire": actual_barrel_at_fire,
                "true_required_angle": true_required_angle,
                "true_miss_deg": true_miss_deg,
                "hit": hit,
            })
            n_done = len(trials)
            print(f"[{n_done:3d}/{total_needed}] offset={offset_deg:+3d}deg "
                  f"true_miss={true_miss_deg if true_miss_deg is None else round(true_miss_deg,2)} hit={hit}")
            time.sleep(0.15)

    with open(OUT_PATH, "w") as f:
        json.dump(trials, f)
    print(f"\nsaved {len(trials)} trials to {OUT_PATH}")

    clean = [t for t in trials if t["true_miss_deg"] is not None]
    print(f"clean (target relocated at fire): {len(clean)}/{len(trials)}")
    misses = np.array([t["true_miss_deg"] for t in clean])
    hits = np.array([t["hit"] for t in clean])
    print(f"true_miss_deg: mean={misses.mean():.2f} std={misses.std():.2f}")
    abs_miss = np.abs(misses)
    bins = [0, 1, 2, 3, 4, 5, 7, 10, 15, 999]
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (abs_miss >= lo) & (abs_miss < hi)
        n = mask.sum()
        if n > 0:
            print(f"|true_miss| in [{lo:2d},{hi:3d}): n={n:3d} hit_rate={100*hits[mask].mean():.0f}%")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(n)
