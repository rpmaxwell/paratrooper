#!/usr/bin/env python3
"""Stage 1.4: map the real hitbox width by deliberately firing at small,
controlled angular offsets from the aim_solver's computed intercept
angle, and recording hit/miss vs. offset. Tells us how much residual
calibration error (Stages 1.1-1.3) we can actually tolerate, and whether
some of the remaining miss rate is a genuinely narrow hitbox rather than
leftover aim error.

Logs full per-trial data (not just aggregate hit counts) to a JSON file:
starting barrel angle, the base (unbiased) solved angle/hold, the biased
target angle/hold actually executed, the *measured* actual barrel angle
right before firing, and the target position -- so hit/miss can be
regressed against rotation distance and direction traveled, not just the
intentional offset, without needing to re-run trials to get that data.
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
from threats import detect_barrel_angle, threat_points
from aim_solver import (
    LATENCY_OFFSET_S,
    ROTATE_LEFT_LIMIT_DEG,
    ROTATE_RIGHT_LIMIT_DEG,
    ROTATION_SPEED_DEG_S,
    TICK_SAFETY_MARGIN_S,
    _clip,
    solve_intercept,
    valid_barrel_angle,
)

TARGET_Y_MIN = 250
# threat_points() only excludes y>=555, but a landed walker's legs merge
# into the ground line and produce cyan pixels as high as y=553-554 --
# just under that cutoff -- which then get mistaken for a still-falling
# target. This was a real, previously-undiagnosed contaminant in this
# script's offset sweep: firing at a static ground artifact always
# misses regardless of the intentional offset, which would flatten out
# a real hitbox falloff signal. Found via diagnose_reaction_latency.py.
TARGET_Y_MAX = 540

# Angular offsets (degrees) applied to the solved intercept angle, cycled
# through round-robin across trials. Signed: positive = aim further left
# (larger angle) than the solved intercept.
OFFSETS_DEG = [-10, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10]

OUT_PATH = "/captures/hitbox_trials.json"


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


def find_target(frame):
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


def biased_hold(barrel_angle, direction, hold_s, offset_deg):
    """Recompute (direction, hold_s) to aim offset_deg away (in absolute
    angle-space) from the originally-solved intercept angle."""
    base_final_angle = _clip(barrel_angle + direction * ROTATION_SPEED_DEG_S * hold_s)
    desired_angle = _clip(base_final_angle + offset_deg)
    delta = desired_angle - barrel_angle
    if abs(delta) < 1e-6:
        return 0, 0.0, desired_angle
    new_direction = 1 if delta > 0 else -1
    new_hold = abs(delta) / ROTATION_SPEED_DEG_S
    return new_direction, new_hold, desired_angle


def main(n_per_offset=10):
    offset_cycle = list(OFFSETS_DEG)
    counts = defaultdict(lambda: [0, 0])  # offset -> [hits, attempts]
    trials = []
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        oi = 0
        total_needed = n_per_offset * len(offset_cycle)
        total_done = 0
        while total_done < total_needed:
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
            b_direction, b_hold_s, desired_angle = biased_hold(barrel0, direction, hold_s, offset_deg)
            rotation_amount = b_direction * ROTATION_SPEED_DEG_S * b_hold_s  # signed degrees traveled

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
            do_action(3)

            time.sleep(0.15)
            frame2 = grab(sct, monitor)
            new_score = read_score(frame2)
            delta = new_score - prev_score
            fire_cost = -1.0 if prev_score > 0 else 0.0
            hit = (delta - fire_cost) > 0

            counts[offset_deg][1] += 1
            if hit:
                counts[offset_deg][0] += 1
            total_done += 1
            trials.append({
                "offset_deg": offset_deg,
                "barrel0": barrel0,
                "base_direction": direction,
                "base_hold_s": hold_s,
                "biased_direction": b_direction,
                "biased_hold_s": b_hold_s,
                "rotation_amount_deg": rotation_amount,
                "desired_angle": desired_angle,
                "actual_barrel_at_fire": actual_barrel_at_fire,
                "target_x": tx,
                "target_y": ty,
                "hit": hit,
            })
            print(f"[{total_done:3d}/{total_needed}] offset={offset_deg:+3d}deg rot={rotation_amount:+6.1f}deg "
                  f"hit={hit} ({counts[offset_deg][0]}/{counts[offset_deg][1]} so far for this offset)")
            time.sleep(0.15)

    with open(OUT_PATH, "w") as f:
        json.dump(trials, f)
    print(f"\nsaved {len(trials)} raw trials to {OUT_PATH}")

    print("\noffset(deg)  hits/attempts  hit_rate")
    for off in sorted(counts.keys()):
        h, a = counts[off]
        print(f"{off:+4d}         {h:2d}/{a:<2d}          {100*h/a:.0f}%")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(n)
