#!/usr/bin/env python3
"""Test which of two hypotheses explains why every "bomb" detection
this session was followed by GAME OVER 5.4-5.7s later, with no exception
(2026-09-24, user observation while watching live):

  (a) the "bomb" IS the visual for the already-documented doom sequence
      (paratrooper_env.py: DOOM_LANDED_THRESHOLD=4 -- once a side's 4th
      landing happens, walkers freeze then "assemble" and end the episode
      a few seconds later) -- i.e. by the time we see the "bomb", doom has
      already been triggered and nothing was ever interceptable.
  (b) the "bomb"-chasing behavior itself is the problem: while tracking
      it, play_with_bombs.py ignores paratroopers, so a side's landings
      climb to the doom threshold *during* that distraction, and the
      "bomb" was an unrelated (maybe real, maybe not) object that just
      happened to correlate because it causes several seconds of
      real-target neglect.

Plays normally (paratroopers as usual, does NOT fire at "bomb"-shaped
blobs) and logs walker counts per side every time one is detected, so we
can see whether a side is already AT or NEAR 4 landings at first
detection (supports a) or still low (supports b).
"""
import subprocess
import sys
import time
from collections import deque

import mss
import numpy as np

from actions import do_action
from read_score import is_done, read_score
from threats import CYAN, WHITE, PIVOT, detect_barrel_angle, threat_points, count_walkers_per_side
from aim_solver import LATENCY_OFFSET_S, TICK_SAFETY_MARGIN_S, solve_intercept, valid_barrel_angle

TARGET_Y_MIN = 250
TARGET_Y_MAX = 540
MAX_BLOB_HEIGHT = 20

BOMB_SKY_Y0, BOMB_SKY_Y1 = 260, 555
BOMB_W_RANGE = (6, 8)
BOMB_H_RANGE = (6, 8)
BOMB_NPX_RANGE = (38, 56)
BOMB_EDGE_MARGIN = 20


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


def cluster_pt(points, radius=15):
    return cluster(points, radius)


def find_paratrooper_target(frame):
    pts = [(x, y) for x, y in threat_points(frame) if TARGET_Y_MIN <= y <= TARGET_Y_MAX]
    if not pts:
        return None
    blobs = cluster_pt(pts, radius=15)
    valid = [b for b in blobs if max(p[1] for p in b) - min(p[1] for p in b) <= MAX_BLOB_HEIGHT]
    if not valid:
        return None
    target = min(valid, key=lambda blob: min(p[1] for p in blob))
    xs = [p[0] for p in target]
    ys = [p[1] for p in target]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def find_bomb_shape(frame):
    mask = np.all(frame == WHITE, axis=-1)
    mask[:BOMB_SKY_Y0, :] = False
    mask[BOMB_SKY_Y1:, :] = False
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    pts = list(zip(xs.tolist(), ys.tolist()))
    blobs = cluster(pts, radius=6)
    for b in blobs:
        bxs = [p[0] for p in b]
        bys = [p[1] for p in b]
        w = max(bxs) - min(bxs)
        h = max(bys) - min(bys)
        cx, cy = sum(bxs) / len(bxs), sum(bys) / len(bys)
        if not (BOMB_W_RANGE[0] <= w <= BOMB_W_RANGE[1]):
            continue
        if not (BOMB_H_RANGE[0] <= h <= BOMB_H_RANGE[1]):
            continue
        if not (BOMB_NPX_RANGE[0] <= len(b) <= BOMB_NPX_RANGE[1]):
            continue
        if cx < BOMB_EDGE_MARGIN or cx > 1024 - BOMB_EDGE_MARGIN:
            continue
        return (cx, cy, w, h, len(b))
    return None


def main(max_seconds=900):
    t_start = time.time()
    last_bomb_report = 0
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        while time.time() - t_start < max_seconds:
            frame = grab(sct, monitor)
            if is_done(frame):
                final_score = read_score(frame)
                print(f"[{time.time()-t_start:7.1f}s] GAME OVER final_score={final_score}")
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.3)
                continue

            bomb = find_bomb_shape(frame)
            now = time.time()
            if bomb is not None and now - last_bomb_report > 0.3:
                left, right = count_walkers_per_side(frame)
                score = read_score(frame)
                print(f"[{now-t_start:7.1f}s] bomb-shape seen at y={bomb[1]:.0f} "
                      f"walkers_left={left} walkers_right={right} score={score}")
                last_bomb_report = now

            # always defend paratroopers normally -- never fire at the
            # bomb-shaped blob, to isolate whether doom happens anyway
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


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 900
    main(secs)
