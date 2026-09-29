#!/usr/bin/env python3
"""Characterize the bomb sprite empirically instead of guessing at it.
Plays a normal paratrooper-priority game (reusing practice_deterministic_aim.py's
target selection + the updated solve_intercept) so the game keeps
advancing, and on every frame separately scans for a small WHITE blob in
the sky that isn't a helicopter (helicopters are much wider) or UI text --
a candidate bomb. Logs its position over time and saves annotated frames
so the trajectory (constant diagonal velocity, per the user's description
watching over VNC) can be confirmed directly rather than assumed.
"""
import json
import subprocess
import sys
import time
from collections import deque

import mss
import numpy as np
from PIL import Image, ImageDraw

from actions import do_action
from read_score import is_done, read_score
from threats import CYAN, WHITE, PIVOT, BARREL_RADIUS, detect_barrel_angle, threat_points
from aim_solver import solve_intercept, valid_barrel_angle

TARGET_Y_MIN = 250
TARGET_Y_MAX = 540
MAX_BLOB_HEIGHT = 20

# Sky region to scan for bomb candidates: below the menu bar, above the
# walker/ground band, excluding the done-screen text band (only relevant
# when not playing) and the score line.
SKY_Y0, SKY_Y1 = 185, 555
# A candidate blob's bounding box must be small (a single bomb sprite),
# not helicopter-sized (helicopters observed ~20-40px wide sprites).
MAX_CANDIDATE_SPAN = 8

OUT_DIR = "/captures/bomb_hunt"
OUT_JSON = "/captures/bomb_hunt_trace.json"


def grab(sct, monitor):
    return np.array(sct.grab(monitor))[:, :, :3][:, :, ::-1]


def cluster(points, radius=6):
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


def find_paratrooper_target(frame):
    pts = [(x, y) for x, y in threat_points(frame) if TARGET_Y_MIN <= y <= TARGET_Y_MAX]
    if not pts:
        return None
    blobs = cluster(pts, radius=15)
    valid = [b for b in blobs if max(p[1] for p in b) - min(p[1] for p in b) <= MAX_BLOB_HEIGHT]
    if not valid:
        return None
    target = min(valid, key=lambda blob: min(p[1] for p in blob))
    xs = [p[0] for p in target]
    ys = [p[1] for p in target]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def find_bomb_candidates(frame):
    """Small WHITE blobs in the sky region, excluding anything as wide as
    a helicopter sprite. Returns list of (cx, cy, w, h, n_px)."""
    mask = np.all(frame == WHITE, axis=-1)
    mask[:SKY_Y0, :] = False
    mask[SKY_Y1:, :] = False
    ys, xs = np.where(mask)
    if xs.size == 0:
        return []
    pts = list(zip(xs.tolist(), ys.tolist()))
    blobs = cluster(pts, radius=6)
    out = []
    for b in blobs:
        bxs = [p[0] for p in b]
        bys = [p[1] for p in b]
        w = max(bxs) - min(bxs)
        h = max(bys) - min(bys)
        if w <= MAX_CANDIDATE_SPAN and h <= MAX_CANDIDATE_SPAN:
            out.append((sum(bxs) / len(bxs), sum(bys) / len(bys), w, h, len(b)))
    return out


def annotate_and_save(frame, candidates, path):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for cx, cy, w, h, n in candidates:
        draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], outline=(255, 0, 0), width=1)
        draw.text((cx + 10, cy - 6), f"{n}px", fill=(255, 0, 0))
    img.save(path)


def main(max_seconds=600):
    subprocess.run(["mkdir", "-p", OUT_DIR], check=True)
    trace = []
    saved = 0
    t_start = time.time()
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        while time.time() - t_start < max_seconds:
            frame = grab(sct, monitor)
            if is_done(frame):
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.3)
                continue

            candidates = find_bomb_candidates(frame)
            if candidates:
                t = time.time() - t_start
                for cx, cy, w, h, n in candidates:
                    trace.append({"t": t, "x": cx, "y": cy, "w": w, "h": h, "n_px": n})
                if saved < 60:
                    path = f"{OUT_DIR}/bomb_candidate_{saved:03d}.png"
                    annotate_and_save(frame, candidates, path)
                    print(f"[t={t:6.1f}s] candidate(s): {candidates} saved {path}")
                    saved += 1
                else:
                    print(f"[t={t:6.1f}s] candidate(s): {candidates}")

            # keep playing paratroopers normally so the game (and any bomber
            # waves) keeps advancing instead of idling
            target = find_paratrooper_target(frame)
            if target is None:
                do_action(0)
                time.sleep(0.05)
                continue
            barrel_angle = detect_barrel_angle(frame)
            if not valid_barrel_angle(barrel_angle):
                time.sleep(0.03)
                continue
            tx, ty = target
            solution = solve_intercept(barrel_angle, tx, ty)
            if solution is None:
                do_action(0)
                time.sleep(0.05)
                continue
            direction, hold_s = solution
            from aim_solver import LATENCY_OFFSET_S, TICK_SAFETY_MARGIN_S
            if direction != 0:
                do_action(1 if direction > 0 else 2)
                sleep_s = max(0.0, hold_s - LATENCY_OFFSET_S + TICK_SAFETY_MARGIN_S)
            else:
                sleep_s = 0.0
            if sleep_s > 0:
                time.sleep(sleep_s)
            do_action(3)
            time.sleep(0.05)

    with open(OUT_JSON, "w") as f:
        json.dump(trace, f)
    print(f"\nsaved {len(trace)} candidate observations to {OUT_JSON}, {saved} annotated frames to {OUT_DIR}")


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 600
    main(secs)
