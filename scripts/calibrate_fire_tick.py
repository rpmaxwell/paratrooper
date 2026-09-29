#!/usr/bin/env python3
"""Stage A follow-up (plan_accuracy_and_priority.md, 2026-09-22): measure
the TRUE barrel angle and target position at the real fire instant by
directly observing them, instead of inferring/correcting for them.

Both the reverted discrete-tick solver and the fire-latency correction
tried against calibrate_hitbox_wide.py's data share the same flaw: they
adjust a *measurement* using a *formula* (a period estimate, a fixed
latency constant) rather than reading the real state off the screen. A
correction for a real effect should only ever reduce error once applied
correctly -- if it doesn't, the formula/proxy is wrong, not the effect.
So: stop correcting, start observing.

Method: `actions.py` documents that the fire key ("Up") both stops the
turret's rotation AND fires in one keypress. That means the barrel angle
sequence, sampled fast enough, must show a clear signature at the real
fire tick: a run of per-tick-changing readings (while still rotating)
followed abruptly by a run of *constant* readings (once stopped) -- the
transition IS the real fire tick, directly observable, no latency
constant needed. This script fires the key asynchronously (Popen, doesn't
block) and bursts-captures frames spanning the call at the highest rate
mss can sustain, then locates that transition per trial.

Cross-check: for trials that hit, the target blob should disappear at (or
immediately after) that same transition tick -- if the two signals
disagree by more than a tick or two, the barrel-stop signal isn't
actually locating the real fire tick and this method needs rethinking
before being trusted.
"""
import json
import subprocess
import sys
import time
from collections import deque

import mss
import numpy as np

from actions import do_action
from read_score import is_done, read_score
from threats import PIVOT, BARREL_RADIUS, CYAN, angle_to, detect_barrel_angle, threat_points
from aim_solver import (
    LATENCY_OFFSET_S,
    ROTATION_SPEED_DEG_S,
    TICK_SAFETY_MARGIN_S,
    _clip,
    solve_intercept,
    valid_barrel_angle,
)

TARGET_Y_MIN = 250
TARGET_Y_MAX = 540
OFFSETS_DEG = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
OUT_PATH = "/captures/fire_tick_trials.json"
PRE_FIRE_TAIL_S = 0.15  # fine-grained monitoring window before the intended fire moment
BURST_DURATION_S = 0.20  # fine-grained monitoring window after the real fire_async() call
MAX_BURST_SAMPLES = 500  # safety cap on total fine-grained samples per trial
KEY_DELAY_MS = 50


def grab(sct, monitor):
    return np.array(sct.grab(monitor))[:, :, :3][:, :, ::-1]


_BR = int(BARREL_RADIUS) + 2
_PX, _PY = PIVOT


def fast_barrel_angle(frame):
    """Equivalent to threats.detect_barrel_angle, but only scans a small
    box around the known pivot instead of a huge (1024x584) strip of the
    frame. Verified to return identical results to detect_barrel_angle
    over repeated live frames. Full-frame scanning measured at ~12.6
    ms/frame in this container (likely x86-on-arm emulation overhead
    scaling with array size, not real CPU cost) -- far too slow to
    resolve the game's ~50ms tick with more than 1-2 samples/tick. This
    crop gets it down to ~0.5ms/frame, needed for the burst capture below
    to actually have the resolution to find the fire-tick transition."""
    y0, y1 = int(_PY - _BR), int(_PY + _BR + 1)
    x0, x1 = int(_PX - _BR), int(_PX + _BR + 1)
    sub = frame[y0:y1, x0:x1]
    mask = np.all(sub == CYAN, axis=-1)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    xs = xs.astype(np.float64) + x0
    ys = ys.astype(np.float64) + y0
    dist = np.hypot(xs - _PX, ys - _PY)
    near = dist <= BARREL_RADIUS
    if not near.any():
        return None
    i = int(np.argmax(dist[near]))
    bx, by = xs[near][i], ys[near][i]
    return angle_to(PIVOT, (bx, by))


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


def single_target_or_none(frame):
    pts = [(x, y) for x, y in threat_points(frame) if TARGET_Y_MIN <= y <= TARGET_Y_MAX]
    if not pts:
        return None
    blobs = cluster(pts)
    if len(blobs) != 1:
        return None
    b = blobs[0]
    xs = [p[0] for p in b]
    ys = [p[1] for p in b]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


_TARGET_CROP_HALF_W = 100
_TARGET_CROP_MARGIN_Y = 20


def crop_target_region(frame, tx):
    """Small crop around the expected target x (no horizontal drift, per
    the fall-speed calibration) -- cheap enough to store one per burst
    sample without the memory cost of keeping full frames."""
    x0 = max(0, int(tx) - _TARGET_CROP_HALF_W)
    x1 = min(frame.shape[1], int(tx) + _TARGET_CROP_HALF_W)
    y0 = max(0, TARGET_Y_MIN - _TARGET_CROP_MARGIN_Y)
    y1 = min(frame.shape[0], TARGET_Y_MAX + _TARGET_CROP_MARGIN_Y)
    return frame[y0:y1, x0:x1].copy(), x0, y0


def single_target_in_crop(crop, x0, y0):
    """Same semantics as single_target_or_none (incl. the BARREL_RADIUS
    exclusion threat_points applies), operating on a crop_target_region
    crop instead of a full frame."""
    mask = np.all(crop == CYAN, axis=-1)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    xs = xs.astype(np.float64) + x0
    ys = ys.astype(np.float64) + y0
    px, py = PIVOT
    dist = np.hypot(xs - px, ys - py)
    keep = dist > BARREL_RADIUS
    xs, ys = xs[keep], ys[keep]
    if xs.size == 0:
        return None
    pts = list(zip(xs.tolist(), ys.tolist()))
    blobs = cluster(pts)
    if len(blobs) != 1:
        return None
    b = blobs[0]
    bxs = [p[0] for p in b]
    bys = [p[1] for p in b]
    return (sum(bxs) / len(bxs), sum(bys) / len(bys))


def biased_hold(barrel_angle, direction, hold_s, offset_deg):
    base_final_angle = _clip(barrel_angle + direction * ROTATION_SPEED_DEG_S * hold_s)
    desired_angle = _clip(base_final_angle + offset_deg)
    delta = desired_angle - barrel_angle
    if abs(delta) < 1e-6:
        return 0, 0.0
    new_direction = 1 if delta > 0 else -1
    new_hold = abs(delta) / ROTATION_SPEED_DEG_S
    return new_direction, new_hold


def fire_async():
    """Same key/delay as actions.do_action(3), but non-blocking so the
    caller can keep grabbing frames while it's in flight."""
    return subprocess.Popen(["xdotool", "key", "--delay", str(KEY_DELAY_MS), "Up"])


def find_fire_tick(samples):
    """samples: list of (t, barrel_angle_or_None, target_xy_or_None).
    Returns index of the first sample belonging to the final stable
    barrel-angle run (the real fire tick), or None if no transition is
    found (e.g. barrel was already stationary the whole burst, or too
    few valid angle readings to tell)."""
    angles = [s[1] for s in samples]
    valid_idx = [i for i, a in enumerate(angles) if a is not None]
    if len(valid_idx) < 2:
        return None
    final_value = angles[valid_idx[-1]]
    # walk backward through valid readings to find the last one that
    # still differs from the final resting value -- that's the last tick
    # of the "still rotating" run.
    boundary = None
    for j in reversed(valid_idx[:-1]):
        if abs(angles[j] - final_value) > 0.3:
            boundary = j
            break
    if boundary is None:
        return None  # constant for the whole burst -- no transition seen
    for k in valid_idx:
        if k > boundary:
            return k
    return None


def main(n_per_offset=6):
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

            target = single_target_or_none(frame)
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

            # Sleep through the bulk of the wait coarsely, then switch to
            # fine-grained monitoring for the tail of it -- this was added
            # after finding that a burst starting only at the fire call
            # often showed the barrel ALREADY stationary from sample 0,
            # meaning the real stop was happening BEFORE we ever called
            # fire_async(), not during its latency as assumed. Monitoring
            # through the tail of the pre-fire wait too lets us see
            # exactly when that really happens instead of guessing.
            coarse_sleep = max(0.0, sleep_s - PRE_FIRE_TAIL_S)
            if coarse_sleep > 0:
                time.sleep(coarse_sleep)
            target_fire_at = time.time() + max(0.0, sleep_s - coarse_sleep)

            # angle_history: full-resolution (t_abs, angle_or_None) --
            # cheap, kept in full. target_crops: small per-sample crops
            # (not full frames) for deferred target-position lookup --
            # cheap enough to keep one per sample too.
            angle_history = []
            target_crops = []
            fire_issued_t = None
            proc = None
            while True:
                now = time.time()
                if fire_issued_t is None and now >= target_fire_at:
                    proc = fire_async()
                    fire_issued_t = time.time()
                f = grab(sct, monitor)
                now = time.time()
                ba = fast_barrel_angle(f)
                angle_history.append((now, ba if valid_barrel_angle(ba) else None))
                target_crops.append(crop_target_region(f, tx))
                if fire_issued_t is not None and now - fire_issued_t >= BURST_DURATION_S:
                    break
                if len(angle_history) >= MAX_BURST_SAMPLES and fire_issued_t is not None:
                    break
            if proc is not None:
                proc.wait(timeout=1.0)

            time.sleep(0.05)
            frame2 = grab(sct, monitor)
            new_score = read_score(frame2)
            delta = new_score - prev_score
            fire_cost = -1.0 if prev_score > 0 else 0.0
            hit = (delta - fire_cost) > 0

            # Normalize to t=0 at the real fire_async() call -- negative t
            # is the observed tail of the pre-fire wait.
            raw = [(t - fire_issued_t, ba, crop) for (t, ba), crop in zip(angle_history, target_crops)]

            fire_idx = find_fire_tick([(t, ba, None) for t, ba, _ in raw])
            true_barrel_at_fire = raw[fire_idx][1] if fire_idx is not None else None
            true_target_at_fire = (
                single_target_in_crop(*raw[fire_idx][2]) if fire_idx is not None else None
            )
            true_miss_deg = None
            if true_barrel_at_fire is not None and true_target_at_fire is not None:
                true_required_angle = angle_to(PIVOT, true_target_at_fire)
                true_miss_deg = true_barrel_at_fire - true_required_angle

            # cross-check: when it's a hit, find the first frame from the
            # fire tick onward where the target blob is gone, and compare
            # its timing to the barrel-stop transition.
            blob_gone_idx = None
            if hit and fire_idx is not None:
                for i in range(fire_idx, len(raw)):
                    if single_target_in_crop(*raw[i][2]) is None:
                        blob_gone_idx = i
                        break
            fire_t = raw[fire_idx][0] if fire_idx is not None else None
            blob_gone_t = raw[blob_gone_idx][0] if blob_gone_idx is not None else None
            samples = raw  # for n_burst_samples below

            trials.append({
                "offset_deg": offset_deg,
                "barrel0": barrel0,
                "target0": [tx, ty],
                "debug_angles": [round(a, 2) if a is not None else None for _, a, _ in raw],
                "debug_times_ms": [round(t * 1000) for t, _, _ in raw],
                "n_burst_samples": len(samples),
                "fire_idx": fire_idx,
                "fire_t": fire_t,
                "true_barrel_at_fire": true_barrel_at_fire,
                "true_target_at_fire": list(true_target_at_fire) if true_target_at_fire else None,
                "true_miss_deg": true_miss_deg,
                "hit": hit,
                "blob_gone_idx": blob_gone_idx,
                "blob_gone_t": blob_gone_t,
            })
            n_done = len(trials)
            print(f"[{n_done:3d}/{total_needed}] offset={offset_deg:+3d}deg "
                  f"true_miss={true_miss_deg if true_miss_deg is None else round(true_miss_deg, 2)} "
                  f"hit={hit} fire_t={fire_t if fire_t is None else round(fire_t*1000)}ms "
                  f"blob_gone_t={blob_gone_t if blob_gone_t is None else round(blob_gone_t*1000)}ms "
                  f"{'(no transition seen)' if fire_idx is None else ''}")
            time.sleep(0.15)

    with open(OUT_PATH, "w") as f:
        json.dump(trials, f)
    print(f"\nsaved {len(trials)} trials to {OUT_PATH}")

    clean = [t for t in trials if t["true_miss_deg"] is not None]
    print(f"clean (fire tick located + target trackable there): {len(clean)}/{len(trials)}")
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

    cross = [t for t in trials if t["hit"] and t["fire_t"] is not None and t["blob_gone_t"] is not None]
    if cross:
        deltas_ms = [1000 * (t["blob_gone_t"] - t["fire_t"]) for t in cross]
        print(f"\ncross-check (hits only, n={len(cross)}): blob-disappear minus barrel-stop "
              f"timing, ms: mean={np.mean(deltas_ms):.1f} std={np.std(deltas_ms):.1f} "
              f"min={min(deltas_ms):.1f} max={max(deltas_ms):.1f}")
    else:
        print("\ncross-check: no hits with both signals available -- can't validate fire-tick detection this run")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    main(n)
