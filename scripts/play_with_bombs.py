#!/usr/bin/env python3
"""Full play loop: paratroopers (priority, using the now-discrete-grid-aware
solve_intercept) plus bombs (using the new solve_bomb_intercept), with
nothing else -- no helicopter targeting yet. Point of this script is
survival/score over a long session now that per-shot accuracy is solid,
not another isolated accuracy measurement.

Bomb signature (hunt_bomb.py, 2026-09-22, one directly observed
trajectory): a small WHITE circular blob, bounding box ~7x7px, ~48 filled
pixels, moving at constant velocity in both x and y (~146, 151 px/s in
the one observed case) -- unlike a paratrooper (cyan, fixed x, vertical
fall) or a helicopter (much wider white+cyan sprite, fixed altitude,
horizontal only). No calibrated fixed bomb velocity yet, so this
estimates vx/vy fresh from two consecutive detections of the same blob
before attempting an intercept.

Runs until game-over the same way every other script here does (detect +
press space + keep going) and just keeps track of scores across restarts
-- no dedicated survival-extension logic, just point the same restart
loop at a longer session.
"""
import math
import subprocess
import sys
import time
from collections import deque

import mss
import numpy as np

from actions import do_action
from read_score import is_done, read_score
from threats import WHITE, PIVOT, detect_barrel_angle, threat_points
from aim_solver import (
    LATENCY_OFFSET_S,
    TICK_SAFETY_MARGIN_S,
    solve_intercept,
    solve_bomb_intercept,
    valid_barrel_angle,
)

TARGET_Y_MIN = 250
TARGET_Y_MAX = 540
MAX_BLOB_HEIGHT = 20

# Floor raised from 200 to 260 (2026-09-23) to exclude the helicopter
# altitude bands (219.2/243.2) -- the WHITE mask over that range is likely
# dominated by helicopter body pixels: excluding it cuts find_bomb()'s
# cost (measured ~71ms/call on this container -- a full-frame WHITE-mask
# scan is expensive here, same emulation overhead already found for
# full-frame CYAN scans, see aim_solver.py's fast_barrel_angle note) with
# zero effect on shot opportunities, since BOMB_MIN_FIRE_Y=380 means
# nothing above 260 would ever be fired on anyway.
BOMB_SKY_Y0, BOMB_SKY_Y1 = 260, 555
BOMB_W_RANGE = (6, 8)
BOMB_H_RANGE = (6, 8)
BOMB_NPX_RANGE = (38, 56)
BOMB_EDGE_MARGIN = 20  # exclude near-edge artifacts (helicopter spawn points observed there)
BOMB_MAX_TRACK_DIST = 150  # px a candidate may have moved between polls and still count as the same bomb
BOMB_MAX_TRACK_GAP_S = 0.4  # max real time between the two velocity-estimating observations
BOMB_MIN_AXIS_SPEED = 40  # px/s -- both |vx| and |vy| must clear this to count as genuinely diagonal

# Post-session analysis (2026-09-23, 161 logged bomb-intercept attempts
# across two live sessions) found two fixable problems, not "bombs are
# just hard to hit at some angles":
#
# 1. A recurring false positive: a fragment of the helicopter sprite near
#    its known altitude bands (219.2/243.2 -- calibrate_helicopter.py)
#    clears the shape+diagonal-motion filter, always with a SHALLOW
#    trajectory angle (observed max 36.8 deg from horizontal) and always
#    near y=224-268. It accounted for 61/161 (38%) of all "bomb" shots
#    and hit 1/61 (1.6%) -- because it was never a bomb. Real bombs were
#    never observed shallower than 45 deg. BOMB_MIN_TRAJ_ANGLE_DEG rejects
#    anything in between with a clean margin on both sides.
BOMB_MIN_TRAJ_ANGLE_DEG = 40

# 2. Among genuine (steep-trajectory) bomb shots, hit rate climbed
#    monotonically with how far the bomb had already fallen when fired at
#    (0% at y~250, 14% at y~300-350, 62% at y~400, 100% at y~500) while
#    showing almost no dependence on how much rotation was needed (27-40%
#    across small/big holds) -- i.e. the problem is lead-time/projection
#    error (a 2-frame velocity estimate's noise compounds over a longer
#    predicted flight time), not execution. Don't fire on an early
#    detection high in the sky -- keep tracking (refreshing the velocity
#    estimate from ever-more-recent, shorter-baseline pairs) until the
#    bomb has fallen past this altitude, which was where the hit rate
#    inflected sharply upward in the data.
BOMB_MIN_FIRE_Y = 380


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


# 2026-09-23 finding (diagnose_target_id.py): the old behavior -- reject
# any blob taller than MAX_BLOB_HEIGHT outright -- was the primary cause
# of "many paratroopers ignored entirely" reported watching over VNC.
# Real example: two clusters of 4-5 vertically-stacked troopers each
# (h=45 and h=41, well over the 20px limit) were both discarded whole,
# leaving nothing to shoot at despite several live, killable targets on
# screen -- and this gets worse the more clutter there is, exactly as
# reported. Splitting by y-gap instead of rejecting outright recovers the
# individual troopers a merged cluster's radius=15 cluster() call
# accidentally chained together.
SPLIT_Y_GAP = 12  # px -- smaller than the 15px cluster radius that
# caused the merge, larger than a single trooper's own internal point
# spread (~11px tall sprite, few-px internal gaps)


def split_oversized_blob(blob, gap=SPLIT_Y_GAP):
    """Re-segment a vertically-merged multi-trooper blob into individual
    groups by splitting wherever the y-gap between consecutive points
    (sorted by y) exceeds `gap`."""
    pts = sorted(blob, key=lambda p: p[1])
    groups = [[pts[0]]]
    for p in pts[1:]:
        if p[1] - groups[-1][-1][1] > gap:
            groups.append([p])
        else:
            groups[-1].append(p)
    return groups


# 2026-09-23 finding (diagnose_target_id.py): after a hit, a scattered
# burst of magenta (+ some cyan) debris pixels appears above the turret
# and lingers -- the cyan fraction of it clears the height filter and
# gets selected as a target, exactly matching "the gun is constantly
# shooting at the spray after a hit" reported watching over VNC. Measured
# directly: a real trooper's cyan footprint is dense (~55-65% of its own
# bounding box filled -- an 8x11-ish solid sprite), while debris
# fragments from the same screen measured 4-29% (scattered dots with
# gaps). MIN_BLOB_FILL_RATIO rejects the sparse case with a wide safety
# margin on both sides; MIN_BLOB_PIXELS additionally drops tiny 1-2px
# noise specks that could otherwise pass a fill-ratio check by chance.
MIN_BLOB_PIXELS = 15
MIN_BLOB_FILL_RATIO = 0.35


def is_real_trooper_shape(blob):
    xs = [p[0] for p in blob]
    ys = [p[1] for p in blob]
    if len(blob) < MIN_BLOB_PIXELS:
        return False
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    fill = len(blob) / ((w + 1) * (h + 1))
    return fill >= MIN_BLOB_FILL_RATIO


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


def find_bomb(frame):
    """Returns (cx, cy, w, h, n_px) of a bomb-shaped WHITE blob, or None.
    Shape only -- motion is checked by the caller across frames, since a
    single frame can't tell a bomb from a coincidentally similar-sized
    fragment of some other white sprite (a helicopter's tail/rotor was
    observed matching the size filter alone, moving purely horizontally --
    see plan doc 2026-09-22)."""
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


def main(max_seconds=1800, max_episodes=None):
    t_start = time.time()
    n_restarts = 0
    n_completed_episodes = 0
    best_score = 0
    last_score = 0
    bomb_shots = 0
    bomb_prev = None  # (t, x, y) of the most recent bomb observation

    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        last_status_t = time.time()
        while time.time() - t_start < max_seconds:
            if max_episodes is not None and n_completed_episodes >= max_episodes:
                print(f"\nreached {max_episodes} completed games, stopping")
                break
            frame = grab(sct, monitor)

            if is_done(frame):
                final_score = read_score(frame)
                best_score = max(best_score, final_score)
                n_restarts += 1
                # the very first is_done seen is just whatever state the
                # game happened to be in when this script started, not a
                # real completed game -- don't count it toward max_episodes
                if n_restarts > 1:
                    n_completed_episodes += 1
                print(f"[{time.time()-t_start:7.1f}s] GAME OVER final_score={final_score} "
                      f"best={best_score} restarts={n_restarts} games={n_completed_episodes}/{max_episodes}")
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.3)
                bomb_prev = None
                continue

            now = time.time()
            if now - last_status_t > 30:
                s = read_score(frame)
                print(f"[{now-t_start:7.1f}s] status: score={s} best={best_score} "
                      f"restarts={n_restarts} bomb_shots={bomb_shots}")
                last_status_t = now

            bomb = find_bomb(frame)
            if bomb is not None:
                bx, by, bw, bh, bn = bomb
                same_shape = (
                    bomb_prev is not None
                    and bomb_prev[3] == bw
                    and bomb_prev[4] == bh
                    and abs(bomb_prev[5] - bn) <= 4
                )
                if (
                    same_shape
                    and now - bomb_prev[0] <= BOMB_MAX_TRACK_GAP_S
                    and abs(bx - bomb_prev[1]) + abs(by - bomb_prev[2]) <= BOMB_MAX_TRACK_DIST
                ):
                    dt = now - bomb_prev[0]
                    vx = (bx - bomb_prev[1]) / dt
                    vy = (by - bomb_prev[2]) / dt
                    # Require genuinely diagonal motion in both axes --
                    # a helicopter fragment (pure horizontal) or a
                    # paratrooper free-fall body (pure vertical) can
                    # otherwise coincidentally match the shape filter.
                    diagonal = abs(vx) >= BOMB_MIN_AXIS_SPEED and abs(vy) >= BOMB_MIN_AXIS_SPEED
                    # Reject the known helicopter-fragment false positive:
                    # real bombs were never observed shallower than 45deg,
                    # that fragment never steeper than 36.8deg.
                    traj_angle = math.degrees(math.atan2(abs(vy), abs(vx))) if diagonal else 0.0
                    is_real_bomb = diagonal and traj_angle >= BOMB_MIN_TRAJ_ANGLE_DEG
                    if is_real_bomb and by >= BOMB_MIN_FIRE_Y:
                        barrel_angle = detect_barrel_angle(frame)
                        if valid_barrel_angle(barrel_angle):
                            # 2026-09-24: solve_bomb_intercept now uses a
                            # calibrated constant-vx/accelerating-vy model
                            # (diagnose_bomb_trajectory.py) instead of the
                            # raw 2-frame vx,vy estimate -- only the sign
                            # of vx (which side it's moving toward) still
                            # comes from live tracking, not the magnitude.
                            vx_sign = 1 if vx > 0 else -1
                            solution = solve_bomb_intercept(barrel_angle, bx, by, vx_sign)
                            if solution is not None:
                                direction, hold_s = solution
                                prev_score = read_score(frame)
                                if direction != 0:
                                    do_action(1 if direction > 0 else 2)
                                    sleep_s = max(0.0, hold_s - LATENCY_OFFSET_S + TICK_SAFETY_MARGIN_S)
                                else:
                                    sleep_s = 0.0
                                if sleep_s > 0:
                                    time.sleep(sleep_s)
                                do_action(3)
                                bomb_shots += 1
                                time.sleep(0.1)
                                new_score = read_score(grab(sct, monitor))
                                print(f"[{time.time()-t_start:7.1f}s] BOMB shot #{bomb_shots}: "
                                      f"pos=({bx:.0f},{by:.0f}) v=({vx:.0f},{vy:.0f}) "
                                      f"angle={traj_angle:.0f}deg dir={direction} hold={hold_s:.3f}s "
                                      f"score {prev_score}->{new_score}")
                        bomb_prev = None  # reset -- fired, need a fresh pair for the next bomb
                    elif is_real_bomb:
                        # genuine bomb, but still too high up -- don't fire
                        # yet (see BOMB_MIN_FIRE_Y above); keep tracking
                        # with this as the new, shorter-baseline reference
                        # point so the next estimate is fresher.
                        bomb_prev = (now, bx, by, bw, bh, bn)
                    else:
                        # not a real bomb (shallow angle, or not diagonal
                        # at all) -- drop it, don't keep chasing it
                        bomb_prev = None
                else:
                    bomb_prev = (now, bx, by, bw, bh, bn)
                time.sleep(0.02)
                continue
            else:
                # 2026-09-23 finding: a single frame with no bomb-shaped
                # blob (transient occlusion by another sprite, a one-frame
                # detection blip) used to reset tracking instantly, forcing
                # a full 2-point re-acquisition. Give it the same grace
                # period as the shape/gap check below instead of zero
                # tolerance.
                if bomb_prev is not None and now - bomb_prev[0] > BOMB_MAX_TRACK_GAP_S:
                    bomb_prev = None

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
            # 2026-09-23 finding: this used to be a single blind
            # time.sleep(sleep_s) -- a bomb appearing during a paratrooper
            # rotation (which can take over a second for a big sweep) went
            # completely unwatched until the rotation finished, by which
            # point it could be badly late or already past BOMB_MIN_FIRE_Y
            # with a stale tracking baseline. Poll for a bomb during the
            # wait and abandon this paratrooper shot (no explosion, just a
            # missed shot -- cheap) the moment one appears, so bomb
            # handling is never blocked behind a paratrooper engagement.
            aborted_for_bomb = False
            if sleep_s > 0:
                t_wait_end = time.time() + sleep_s
                while time.time() < t_wait_end:
                    if find_bomb(grab(sct, monitor)) is not None:
                        aborted_for_bomb = True
                        break
                    time.sleep(0.02)
            if aborted_for_bomb:
                print(f"[{time.time()-t_start:7.1f}s] aborted paratrooper shot for bomb "
                      f"(was {sleep_s:.2f}s into rotation)")
                continue
            do_action(3)
            time.sleep(0.05)

        best_score = max(best_score, read_score(grab(sct, monitor)))

    print(f"\nsession ended after {time.time()-t_start:.0f}s: best_score={best_score} "
          f"restarts={n_restarts} games={n_completed_episodes} bomb_shots={bomb_shots}")


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 1800
    n_games = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(secs, n_games)
