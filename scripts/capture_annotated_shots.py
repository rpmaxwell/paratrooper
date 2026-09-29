#!/usr/bin/env python3
"""Save annotated pre-fire frames (aim ray + detected target blob marked)
for direct visual inspection, instead of trying to diagnose the miss
pattern through more derived numeric proxies. Fires at the plain (no
intentional offset) solved intercept -- this is about seeing what's
really happening, not stress-testing the offset sweep.
"""
import subprocess
import sys
import time
from collections import deque

import mss
import numpy as np
from PIL import Image, ImageDraw

from actions import do_action
from read_score import is_done, read_score
from threats import PIVOT, angle_to, detect_barrel_angle, threat_points
from aim_solver import LATENCY_OFFSET_S, TICK_SAFETY_MARGIN_S, solve_intercept, valid_barrel_angle

TARGET_Y_MIN = 250
TARGET_Y_MAX = 540
OUT_DIR = "/captures/annotated"


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


def all_targets(frame):
    pts = [(x, y) for x, y in threat_points(frame) if TARGET_Y_MIN <= y <= TARGET_Y_MAX]
    if not pts:
        return []
    blobs = cluster(pts)
    out = []
    for b in blobs:
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        out.append(((sum(xs) / len(xs), sum(ys) / len(ys)), b))
    return out


def find_target(frame):
    targets = all_targets(frame)
    if not targets:
        return None
    return min(targets, key=lambda t: t[0][1])


def annotate_and_save(frame, barrel_angle, target_blobs, path, hit):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    px, py = PIVOT
    # draw the barrel's aim ray, long enough to cross the whole screen
    import math
    rad = math.radians(barrel_angle)
    dx, dy = math.cos(rad), -math.sin(rad)
    draw.line([(px, py), (px + dx * 900, py + dy * 900)], fill=(255, 0, 0), width=1)
    # mark every detected target blob's bounding box + centroid
    for (cx, cy), pts in target_blobs:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        draw.rectangle([min(xs) - 2, min(ys) - 2, max(xs) + 2, max(ys) + 2], outline=(255, 255, 0), width=1)
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(255, 255, 0))
    draw.text((10, 10), f"barrel={barrel_angle:.1f} hit={hit}", fill=(255, 255, 255))
    img.save(path)


def main(n_shots=15):
    subprocess.run(["mkdir", "-p", OUT_DIR], check=True)
    saved = 0
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        while saved < n_shots:
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

            tx, ty = target[0]
            solution = solve_intercept(barrel0, tx, ty)
            if solution is None:
                do_action(0)
                time.sleep(0.05)
                continue

            direction, hold_s = solution
            prev_score = read_score(frame)
            if direction != 0:
                do_action(1 if direction > 0 else 2)
                sleep_s = max(0.0, hold_s - LATENCY_OFFSET_S + TICK_SAFETY_MARGIN_S)
            else:
                sleep_s = 0.0
            if sleep_s > 0:
                time.sleep(sleep_s)

            prefire_frame = grab(sct, monitor)
            actual_barrel_at_fire = detect_barrel_angle(prefire_frame)
            target_blobs = all_targets(prefire_frame)

            do_action(3)
            time.sleep(0.15)
            frame2 = grab(sct, monitor)
            new_score = read_score(frame2)
            delta = new_score - prev_score
            fire_cost = -1.0 if prev_score > 0 else 0.0
            hit = (delta - fire_cost) > 0

            if actual_barrel_at_fire is not None:
                path = f"{OUT_DIR}/shot_{saved:03d}_{'hit' if hit else 'miss'}.png"
                annotate_and_save(prefire_frame, actual_barrel_at_fire, target_blobs, path, hit)
                print(f"saved {path} (n_target_blobs={len(target_blobs)})")
                saved += 1
            time.sleep(0.15)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(n)
