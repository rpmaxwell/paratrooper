#!/usr/bin/env python3
"""Diagnose target identification during normal (possibly cluttered) play,
per user report watching over VNC (2026-09-23): the gun keeps shooting at
"the spray" after a hit, many paratroopers get ignored entirely, easy
shots are missed, and all of this gets worse with more clutter on screen.
This is not just an aim-accuracy question -- it's about what
find_paratrooper_target-style blob detection is actually seeing and
choosing (or failing to choose) frame to frame.

Plays a normal game (no isolation requirement, unlike
calibrate_isolated_miss.py) and instruments every frame:
  - every raw cyan blob in the threat region (not just the one selected),
    with its size, so clutter level is measured directly, not guessed
  - "ignored" events: threat pixels present in the target y-range but
    find_paratrooper_target's own filters (oversized/merged-blob
    rejection) throw all of them out, leaving nothing to shoot at even
    though something is visibly there
  - a short post-fire burst capture for a sample of shots (hit and miss),
    saved as an annotated frame sequence, to see directly whether
    something (a hit-effect/"spray") appears where the target was and
    could plausibly be re-acquired as a new target on a later frame
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
from threats import CYAN, PIVOT, BARREL_RADIUS, THREAT_Y_MAX, detect_barrel_angle, threat_points
from aim_solver import LATENCY_OFFSET_S, TICK_SAFETY_MARGIN_S, solve_intercept, valid_barrel_angle

TARGET_Y_MIN = 250
TARGET_Y_MAX = 540
MAX_BLOB_HEIGHT = 20

OUT_DIR = "/captures/target_id_debug"
OUT_JSON = "/captures/target_id_trials.json"
MAX_ANNOTATED_PER_CATEGORY = 25
BURST_N_FRAMES = 10
BURST_INTERVAL_S = 0.04


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


SPLIT_Y_GAP = 12


def split_oversized_blob(blob, gap=SPLIT_Y_GAP):
    pts = sorted(blob, key=lambda p: p[1])
    groups = [[pts[0]]]
    for p in pts[1:]:
        if p[1] - groups[-1][-1][1] > gap:
            groups.append([p])
        else:
            groups[-1].append(p)
    return groups


def all_cyan_blobs(frame):
    """Every cyan blob in the target y-range, post-fix (2026-09-23):
    oversized/merged clusters are re-split by y-gap instead of discarded
    whole, recovering the individual troopers a radius=15 cluster() call
    can accidentally chain together -- returns (cx, cy, w, h, n_px) each,
    all within MAX_BLOB_HEIGHT except any residual that couldn't be split
    further (kept so `select_target` can still flag it as oversized)."""
    pts = [(x, y) for x, y in threat_points(frame) if TARGET_Y_MIN <= y <= TARGET_Y_MAX]
    if not pts:
        return []
    raw_blobs = cluster(pts, radius=15)
    blobs = []
    for b in raw_blobs:
        h = max(p[1] for p in b) - min(p[1] for p in b)
        if h <= MAX_BLOB_HEIGHT:
            blobs.append(b)
        else:
            blobs.extend(split_oversized_blob(b))
    out = []
    for b in blobs:
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        out.append((sum(xs) / len(xs), sum(ys) / len(ys), w, h, len(b)))
    return out


def select_target(blobs):
    """Same logic as find_paratrooper_target elsewhere: reject any
    residual oversized blob (a split() couldn't fully separate), pick the
    topmost of what's left."""
    valid = [b for b in blobs if b[3] <= MAX_BLOB_HEIGHT]
    if not valid:
        return None
    return min(valid, key=lambda b: b[1])


def any_non_black_diff(frame_a, frame_b):
    """Pixel positions that changed between two frames, with each side's
    color -- used to see literally what appeared/disappeared after a
    shot, regardless of color (a hit-effect might not be cyan)."""
    diff_mask = np.any(frame_a != frame_b, axis=-1)
    ys, xs = np.where(diff_mask)
    out = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        out.append({"x": int(x), "y": int(y), "before": frame_a[y, x].tolist(), "after": frame_b[y, x].tolist()})
    return out


def annotate(frame, blobs, selected, path, extra_text=""):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for b in blobs:
        cx, cy, w, h, n = b
        color = (0, 255, 0) if b is selected else ((255, 0, 0) if h > MAX_BLOB_HEIGHT else (255, 255, 0))
        draw.rectangle([cx - w / 2 - 2, cy - h / 2 - 2, cx + w / 2 + 2, cy + h / 2 + 2], outline=color, width=1)
        draw.text((cx + 6, cy - 6), f"h={h:.0f} n={n}", fill=color)
    if extra_text:
        draw.text((10, 10), extra_text, fill=(255, 255, 255))
    img.save(path)


def main(max_seconds=900):
    subprocess.run(["mkdir", "-p", OUT_DIR], check=True)
    t_start = time.time()
    trials = []
    clutter_hist = {}
    n_ignored_events = 0
    n_ignored_saved = 0
    n_hit_saved = 0
    n_miss_saved = 0
    n_shots = 0
    n_hits = 0

    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        last_status = time.time()
        while time.time() - t_start < max_seconds:
            frame = grab(sct, monitor)
            if is_done(frame):
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.3)
                continue

            now = time.time()
            if now - last_status > 30:
                print(f"[{now-t_start:6.1f}s] shots={n_shots} hits={n_hits} "
                      f"ignored_events={n_ignored_events} clutter_hist={clutter_hist}")
                last_status = now

            blobs = all_cyan_blobs(frame)
            n_blobs = len(blobs)
            clutter_hist[n_blobs] = clutter_hist.get(n_blobs, 0) + 1

            target = select_target(blobs)
            if target is None:
                if n_blobs > 0:
                    # something is visibly there, but every blob got
                    # rejected (all oversized/merged) -- exactly the
                    # "paratroopers ignored entirely" symptom
                    n_ignored_events += 1
                    if n_ignored_saved < MAX_ANNOTATED_PER_CATEGORY:
                        path = f"{OUT_DIR}/ignored_{n_ignored_saved:03d}.png"
                        annotate(frame, blobs, None, path, f"IGNORED n_blobs={n_blobs}")
                        n_ignored_saved += 1
                        print(f"[{now-t_start:6.1f}s] ignored event: n_blobs={n_blobs} "
                              f"sizes={[(round(b[3]),b[4]) for b in blobs]} saved {path}")
                do_action(0)
                time.sleep(0.05)
                continue

            barrel_angle = detect_barrel_angle(frame)
            if not valid_barrel_angle(barrel_angle):
                time.sleep(0.03)
                continue

            tx, ty = target[0], target[1]
            solution = solve_intercept(barrel_angle, tx, ty)
            if solution is None:
                do_action(0)
                time.sleep(0.05)
                continue

            direction, hold_s = solution
            prev_score = read_score(frame)
            pre_fire_blobs = blobs
            pre_fire_clutter = n_blobs
            if direction != 0:
                do_action(1 if direction > 0 else 2)
                sleep_s = max(0.0, hold_s - LATENCY_OFFSET_S + TICK_SAFETY_MARGIN_S)
            else:
                sleep_s = 0.0
            if sleep_s > 0:
                time.sleep(sleep_s)

            prefire_frame = grab(sct, monitor)
            do_action(3)
            fire_t = time.time()

            burst = []
            for _ in range(BURST_N_FRAMES):
                while time.time() - fire_t < len(burst) * BURST_INTERVAL_S:
                    pass
                burst.append(grab(sct, monitor))

            new_score = read_score(burst[-1])
            delta = new_score - prev_score
            fire_cost = -1.0 if prev_score > 0 else 0.0
            hit = (delta - fire_cost) > 0
            n_shots += 1
            if hit:
                n_hits += 1

            save_this = (hit and n_hit_saved < MAX_ANNOTATED_PER_CATEGORY) or \
                        (not hit and n_miss_saved < MAX_ANNOTATED_PER_CATEGORY)
            if save_this:
                tag = "hit" if hit else "miss"
                idx = n_hit_saved if hit else n_miss_saved
                pre_path = f"{OUT_DIR}/{tag}_{idx:03d}_pre.png"
                annotate(prefire_frame, pre_fire_blobs, target, pre_path,
                          f"PRE clutter={pre_fire_clutter} target=({tx:.0f},{ty:.0f})")
                for bi, bf in enumerate(burst):
                    post_blobs = all_cyan_blobs(bf)
                    post_path = f"{OUT_DIR}/{tag}_{idx:03d}_post{bi:02d}.png"
                    annotate(bf, post_blobs, None, post_path,
                              f"POST+{bi*BURST_INTERVAL_S*1000:.0f}ms clutter={len(post_blobs)}")
                if hit:
                    n_hit_saved += 1
                else:
                    n_miss_saved += 1
                print(f"[{time.time()-t_start:6.1f}s] saved {tag} #{idx} clutter={pre_fire_clutter}")

            trials.append({
                "t": now - t_start,
                "hit": hit,
                "pre_fire_clutter": pre_fire_clutter,
                "target": [tx, ty],
                "target_size": [target[2], target[3], target[4]],
            })
            time.sleep(0.05)

    with open(OUT_JSON, "w") as f:
        json.dump(trials, f)
    print(f"\nsaved {len(trials)} trials to {OUT_JSON}")
    print(f"shots={n_shots} hits={n_hits} hit_rate={100*n_hits/n_shots if n_shots else 0:.0f}%")
    print(f"ignored_events={n_ignored_events}")
    print(f"clutter histogram: {clutter_hist}")

    # hit rate by clutter level
    import collections
    bybucket = collections.defaultdict(lambda: [0, 0])
    for t in trials:
        b = min(t["pre_fire_clutter"], 4)
        bybucket[b][1] += 1
        if t["hit"]:
            bybucket[b][0] += 1
    for b in sorted(bybucket):
        h, n = bybucket[b]
        label = f"{b}" if b < 4 else "4+"
        print(f"clutter={label}: {h}/{n} hit_rate={100*h/n:.0f}%")


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 900
    main(secs)
