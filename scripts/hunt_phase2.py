#!/usr/bin/env python3
"""Find the real phase-2 transition (bomber planes replace helicopters,
per user confirmation 2026-09-24) and characterize what a real bomb/plane
actually looks like, instead of continuing to chase the phase-1
doom-sequence artifact (diagnose_bomb_vs_doom.py, same day, confirmed
that shape's appearance/game-over is unaffected by shooting it and
doesn't correlate with visible walker counts reaching the doom
threshold -- it's very likely part of the phase-1 ending animation, not
phase 2).

Plays phase 1 normally (paratroopers only -- never engages the known
"fake bomb" shape, since engaging it does nothing per today's finding).
Tracks whether a helicopter (established signature: CYAN pixels in
HELI_Y0-HELI_Y1, see calibrate_helicopter.py) has been seen recently. If
helicopters go quiet for a sustained stretch (HELI_QUIET_S), that's the
user-confirmed signal that phase 2 may have begun -- start saving full
screenshots frequently from that point so we can see, for the first time
today, what a real bomber plane and a real bomb actually look like.
"""
import subprocess
import sys
import time
from collections import deque

import mss
import numpy as np
from PIL import Image

from actions import do_action
from read_score import is_done, read_score
from threats import CYAN, PIVOT, detect_barrel_angle, threat_points
from aim_solver import LATENCY_OFFSET_S, TICK_SAFETY_MARGIN_S, solve_intercept, valid_barrel_angle

TARGET_Y_MIN = 250
TARGET_Y_MAX = 540
MAX_BLOB_HEIGHT = 20

HELI_Y0, HELI_Y1 = 184, 250
HELI_QUIET_S = 15.0  # no helicopter seen for this long -> candidate phase 2

# 2026-09-24 correction: the first two "PHASE 2" flags both turned out to be
# the ordinary phase-1 doom/game-over cutscene (walkers frozen next to the
# turret, empty sky, no plane/bomb) -- helicopters also stop spawning during
# that sequence, so the quiet-timer alone can't tell "dying" from "phase 2
# arrived". Fix: treat crossing HELI_QUIET_S as only a CANDIDATE. Only
# declare real phase 2 if the episode keeps running (is_done() stays False)
# for another CONFIRM_SURVIVAL_S beyond that -- a doom sequence always ends
# in GAME OVER well before this extra window elapses.
CONFIRM_SURVIVAL_S = 10.0

OUT_DIR = "/captures/phase2_hunt"


def grab(sct, monitor):
    return np.array(sct.grab(monitor))[:, :, :3][:, :, ::-1]


def cluster(points, radius=20):
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


HELI_MIN_WIDTH = 15  # px -- wide enough to distinguish a real helicopter
# body from a narrow (~8px) freshly-released trooper briefly passing
# through the same y-band right after leaving the helicopter


def heli_seen(frame):
    mask = np.all(frame == CYAN, axis=-1)
    mask[:HELI_Y0, :] = False
    mask[HELI_Y1:, :] = False
    ys, xs = np.where(mask)
    if xs.size == 0:
        return False
    pts = list(zip(xs.tolist(), ys.tolist()))
    for b in cluster(pts, radius=20):
        bxs = [p[0] for p in b]
        if max(bxs) - min(bxs) >= HELI_MIN_WIDTH:
            return True
    return False


def main(max_seconds=1800):
    subprocess.run(["mkdir", "-p", OUT_DIR], check=True)
    t_start = time.time()
    last_heli_t = time.time()
    candidate = False       # crossed HELI_QUIET_S, not yet confirmed
    candidate_t = None      # when the candidate window started
    confirmed = False       # survived CONFIRM_SURVIVAL_S past candidate -> real phase 2
    n_saved = 0
    last_save_t = 0

    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        while time.time() - t_start < max_seconds:
            frame = grab(sct, monitor)
            now = time.time()
            if is_done(frame):
                score = read_score(frame)
                if candidate and not confirmed:
                    print(f"[{now-t_start:7.1f}s] GAME OVER score={score} -- candidate expired "
                          f"after {now-candidate_t:.1f}s, was doom not phase 2 "
                          f"(heli quiet for {now-last_heli_t:.1f}s)")
                else:
                    print(f"[{now-t_start:7.1f}s] GAME OVER score={score} "
                          f"(heli quiet for {now-last_heli_t:.1f}s, confirmed_phase2={confirmed})")
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.3)
                last_heli_t = time.time()
                candidate = False
                candidate_t = None
                confirmed = False
                continue

            if heli_seen(frame):
                if candidate or confirmed:
                    print(f"[{now-t_start:7.1f}s] helicopter reappeared -- back to phase 1 "
                          f"(was doom-quiet or a false alarm, not phase 2)")
                last_heli_t = now
                candidate = False
                candidate_t = None
                confirmed = False

            quiet_for = now - last_heli_t
            if quiet_for > HELI_QUIET_S and not candidate:
                candidate = True
                candidate_t = now
                print(f"[{now-t_start:7.1f}s] no helicopter for {quiet_for:.1f}s -- "
                      f"CANDIDATE phase 2, watching for survival past "
                      f"{CONFIRM_SURVIVAL_S:.0f}s more before confirming")

            if candidate and not confirmed and now - candidate_t > CONFIRM_SURVIVAL_S:
                confirmed = True
                print(f"[{now-t_start:7.1f}s] survived {now-candidate_t:.1f}s past candidate "
                      f"without GAME OVER -- CONFIRMED real PHASE 2, saving frequent screenshots")

            if confirmed and now - last_save_t > 0.5:
                path = f"{OUT_DIR}/frame_{n_saved:04d}_t{now-t_start:.1f}.png"
                Image.fromarray(frame).save(path)
                n_saved += 1
                last_save_t = now
                if n_saved % 10 == 0:
                    print(f"[{now-t_start:7.1f}s] saved {n_saved} phase-2-candidate frames")

            # defend paratroopers normally; never engage the known
            # phase-1 doom-sequence artifact
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

    print(f"\nsaved {n_saved} phase-2-candidate frames to {OUT_DIR}")


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 1800
    main(secs)
