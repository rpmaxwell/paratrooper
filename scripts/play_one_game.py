#!/usr/bin/env python3
"""Play exactly one episode (paratrooper defense only, no bomb-chasing --
that logic was built against the phase-1 doom artifact, not a real threat)
and then STOP without restarting, so the user can watch the ending live
over VNC and describe what actually happened, instead of us guessing from
automated heuristics.

Usage: python3 play_one_game.py
Optionally pass a starting delay (seconds) if you need time to switch to
the VNC window before it starts reacting.
"""
import sys
import time
from collections import deque

import mss
import numpy as np

from actions import do_action
from read_score import is_done, read_score
from threats import PIVOT, detect_barrel_angle, threat_points
from aim_solver import LATENCY_OFFSET_S, TICK_SAFETY_MARGIN_S, solve_intercept, valid_barrel_angle

TARGET_Y_MIN = 250
TARGET_Y_MAX = 540
MAX_BLOB_HEIGHT = 20
SPLIT_Y_GAP = 12
MIN_BLOB_PIXELS = 15
MIN_BLOB_FILL_RATIO = 0.35


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


def split_oversized_blob(blob, gap=SPLIT_Y_GAP):
    pts = sorted(blob, key=lambda p: p[1])
    groups = [[pts[0]]]
    for p in pts[1:]:
        if p[1] - groups[-1][-1][1] > gap:
            groups.append([p])
        else:
            groups[-1].append(p)
    return groups


def is_real_trooper_shape(blob):
    xs = [p[0] for p in blob]
    ys = [p[1] for p in blob]
    if len(blob) < MIN_BLOB_PIXELS:
        return False
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    # helicopter skids (32x4 cyan) are dense too, and from the second
    # helicopter wave on they fly low enough (y~265) to fall inside the
    # target band -- a trooper is ~8 wide x ~11 tall, never flat and wide
    if h <= 5 and w >= 14:
        return False
    fill = len(blob) / ((w + 1) * (h + 1))
    return fill >= MIN_BLOB_FILL_RATIO


def find_paratrooper_target(frame):
    pts = [(x, y) for x, y in threat_points(frame) if TARGET_Y_MIN <= y <= TARGET_Y_MAX]
    if not pts:
        return None
    blobs = cluster(pts, radius=15)
    valid = []
    for b in blobs:
        h = max(p[1] for p in b) - min(p[1] for p in b)
        if h <= MAX_BLOB_HEIGHT:
            if is_real_trooper_shape(b):
                valid.append(b)
        else:
            for sub in split_oversized_blob(b):
                sh = max(p[1] for p in sub) - min(p[1] for p in sub)
                if sh <= MAX_BLOB_HEIGHT and is_real_trooper_shape(sub):
                    valid.append(sub)
    if not valid:
        return None
    target = min(valid, key=lambda blob: min(p[1] for p in blob))
    xs = [p[0] for p in target]
    ys = [p[1] for p in target]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def main(start_delay=0.0, max_seconds=600):
    if start_delay > 0:
        print(f"starting in {start_delay:.0f}s...")
        time.sleep(start_delay)

    t_start = time.time()
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]

        # skip past whatever game-over state we might already be sitting in
        frame = grab(sct, monitor)
        if is_done(frame):
            print("clearing stale game-over screen before starting...")
            import subprocess
            subprocess.run(["xdotool", "key", "space"], check=True)
            time.sleep(0.5)

        print(f"[{0.0:7.1f}s] playing...")
        while time.time() - t_start < max_seconds:
            frame = grab(sct, monitor)
            if is_done(frame):
                final_score = read_score(frame)
                print(f"[{time.time()-t_start:7.1f}s] GAME OVER final_score={final_score} -- "
                      f"stopping here, describe what you saw")
                return

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
            if direction != 0:
                do_action(1 if direction > 0 else 2)
                sleep_s = max(0.0, hold_s - LATENCY_OFFSET_S + TICK_SAFETY_MARGIN_S)
            else:
                sleep_s = 0.0
            if sleep_s > 0:
                time.sleep(sleep_s)
            do_action(3)
            time.sleep(0.05)

        print(f"[{time.time()-t_start:7.1f}s] hit max_seconds without game-over -- stopping")


if __name__ == "__main__":
    delay = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    main(delay)
