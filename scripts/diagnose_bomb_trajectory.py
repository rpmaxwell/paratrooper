#!/usr/bin/env python3
"""Diagnose why bomb shots keep landing at almost exactly the same y
(388 x3, 402 x2 across 5 live misses, 2026-09-23/24) despite very
different observed velocities -- that's not natural continuous-crossing
jitter, something is likely changing at/near that height (a game-side
behavior change, or a detection artifact of ours). Plays normally
(paratroopers shot as usual) but logs the FULL raw sequence of bomb-shaped
detections for each encounter, unfiltered by the shape/diagonal/angle
gates play_with_bombs.py applies before firing, tagged by a track id so
each bomb's whole visible trajectory (not just the last point before
firing) can be inspected directly.
"""
import json
import math
import subprocess
import sys
import time
from collections import deque

import mss
import numpy as np

from actions import do_action
from read_score import is_done, read_score
from threats import CYAN, WHITE, PIVOT, detect_barrel_angle, threat_points
from aim_solver import LATENCY_OFFSET_S, TICK_SAFETY_MARGIN_S, solve_intercept, valid_barrel_angle

TARGET_Y_MIN = 250
TARGET_Y_MAX = 540
MAX_BLOB_HEIGHT = 20

BOMB_SKY_Y0, BOMB_SKY_Y1 = 260, 555
BOMB_W_RANGE = (6, 8)
BOMB_H_RANGE = (6, 8)
BOMB_NPX_RANGE = (38, 56)
BOMB_EDGE_MARGIN = 20
BOMB_MAX_TRACK_DIST = 150
BOMB_MAX_TRACK_GAP_S = 0.4

OUT_JSON = "/captures/bomb_trajectory_traces.json"


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


def split_oversized_blob(blob, gap=12):
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
    if len(blob) < 15:
        return False
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return len(blob) / ((w + 1) * (h + 1)) >= 0.35


def find_paratrooper_target(frame):
    pts = [(x, y) for x, y in threat_points(frame) if TARGET_Y_MIN <= y <= TARGET_Y_MAX]
    if not pts:
        return None
    blobs = cluster_pt(pts, radius=15)
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


def find_bomb_candidates(frame):
    """ALL white blobs matching the bomb size filter, unfiltered by
    motion/angle -- returns list of (cx, cy, w, h, n_px)."""
    mask = np.all(frame == WHITE, axis=-1)
    mask[:BOMB_SKY_Y0, :] = False
    mask[BOMB_SKY_Y1:, :] = False
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
        cx, cy = sum(bxs) / len(bxs), sum(bys) / len(bys)
        if not (BOMB_W_RANGE[0] <= w <= BOMB_W_RANGE[1]):
            continue
        if not (BOMB_H_RANGE[0] <= h <= BOMB_H_RANGE[1]):
            continue
        if not (BOMB_NPX_RANGE[0] <= len(b) <= BOMB_NPX_RANGE[1]):
            continue
        if cx < BOMB_EDGE_MARGIN or cx > 1024 - BOMB_EDGE_MARGIN:
            continue
        out.append((cx, cy, w, h, len(b)))
    return out


def main(max_seconds=900):
    t_start = time.time()
    traces = []  # list of finished tracks, each a list of sample dicts
    current_track = None  # list of sample dicts, or None
    last_seen = None  # (t, x, y, w, h, n) of the most recent sample in current_track

    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        while time.time() - t_start < max_seconds:
            frame = grab(sct, monitor)
            if is_done(frame):
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.3)
                continue

            now = time.time()
            candidates = find_bomb_candidates(frame)

            matched = None
            if candidates and last_seen is not None:
                lt, lx, ly, lw, lh, ln = last_seen
                if now - lt <= BOMB_MAX_TRACK_GAP_S:
                    best, best_d = None, BOMB_MAX_TRACK_DIST
                    for c in candidates:
                        cx, cy, cw, ch, cn = c
                        if cw == lw and ch == lh and abs(cn - ln) <= 4:
                            d = abs(cx - lx) + abs(cy - ly)
                            if d < best_d:
                                best, best_d = c, d
                    matched = best

            if matched is not None:
                cx, cy, cw, ch, cn = matched
                dt = now - last_seen[0]
                vx = (cx - last_seen[1]) / dt
                vy = (cy - last_seen[2]) / dt
                angle = math.degrees(math.atan2(abs(vy), abs(vx)))
                sample = {"t": now - t_start, "x": cx, "y": cy, "w": cw, "h": ch, "n": cn,
                          "vx": vx, "vy": vy, "angle": angle, "dt": dt}
                current_track.append(sample)
                last_seen = (now, cx, cy, cw, ch, cn)
                print(f"[{sample['t']:7.2f}s] track y={cy:.0f} v=({vx:.0f},{vy:.0f}) "
                      f"angle={angle:.0f}deg dt={dt*1000:.0f}ms")
            elif candidates:
                # no match to continue -- start a fresh track with the
                # first candidate (arbitrary if several)
                if current_track:
                    traces.append(current_track)
                c = candidates[0]
                current_track = [{"t": now - t_start, "x": c[0], "y": c[1], "w": c[2], "h": c[3],
                                   "n": c[4], "vx": None, "vy": None, "angle": None, "dt": None}]
                last_seen = (now, c[0], c[1], c[2], c[3], c[4])
                print(f"[{now-t_start:7.2f}s] NEW track y={c[1]:.0f}")
            else:
                if current_track is not None and len(current_track) > 1:
                    traces.append(current_track)
                    print(f"[{now-t_start:7.2f}s] track ended, n={len(current_track)}")
                current_track = None
                last_seen = None

            if candidates:
                time.sleep(0.02)
                continue

            # no bomb candidate this frame -- fall through to normal
            # paratrooper defense so the game keeps progressing
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

    if current_track and len(current_track) > 1:
        traces.append(current_track)

    with open(OUT_JSON, "w") as f:
        json.dump(traces, f)
    print(f"\nsaved {len(traces)} bomb tracks to {OUT_JSON}")
    for i, tr in enumerate(traces):
        ys = [s["y"] for s in tr]
        print(f"track {i}: n={len(tr)} y_range=({min(ys):.0f},{max(ys):.0f}) "
              f"t_range=({tr[0]['t']:.2f},{tr[-1]['t']:.2f})")


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 900
    main(secs)
