#!/usr/bin/env python3
"""Step 2 of the gun-only accuracy test (plan_accuracy_and_priority.md,
2026-09-22): with target detection removed entirely, test whether the gun
itself can reliably rotate from a known discrete angle A to a chosen
discrete angle B (from discover_firing_angles.py's measured list) and
fire exactly there.

Uses the SAME control path as real play (rotate, sleep for
hold_s - LATENCY_OFFSET_S, fire) so this measures the real gun's current
timing accuracy, not a new/different mechanism -- the only thing removed
is the target (no paratrooper detection, no fall-rate model, no
intercept solving). Ground truth for where it actually landed comes from
the same direct fire-tick burst-capture method validated in
calibrate_fire_tick.py (observe the barrel-angle transition across the
fire keypress), not from a formula.

No target on screen means score stays flat while paratroopers keep
landing -- the episode WILL end on its own before long. There's no known
reliable way to prevent that, so this just detects game-over (is_done)
and restarts (space), same as every other script here, and keeps going
across as many episodes as it takes to collect the sample.
"""
import json
import random
import subprocess
import sys
import time

import mss
import numpy as np

from actions import do_action
from read_score import is_done
from threats import PIVOT, BARREL_RADIUS, CYAN, angle_to
from aim_solver import (
    LATENCY_OFFSET_S,
    ROTATION_SPEED_DEG_S,
    TICK_SAFETY_MARGIN_S,
    valid_barrel_angle,
)

ANGLES_PATH = "/captures/firing_angles.json"
OUT_PATH = "/captures/gun_open_loop_trials.json"
PRE_FIRE_TAIL_S = 0.15
BURST_DURATION_S = 0.20
MAX_BURST_SAMPLES = 500
KEY_DELAY_MS = 50

# Tick-index deltas to cycle through, so trials cover a spread of short
# and long rotations in both directions rather than always adjacent
# ticks or always full-range sweeps.
DELTAS = [1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 12, -12, 17, -17]


def grab(sct, monitor):
    return np.array(sct.grab(monitor))[:, :, :3][:, :, ::-1]


_BR = int(BARREL_RADIUS) + 2
_PX, _PY = PIVOT


def fast_barrel_angle(frame):
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


def fire_async():
    return subprocess.Popen(["xdotool", "key", "--delay", str(KEY_DELAY_MS), "Up"])


def find_fire_tick(angle_seq):
    """angle_seq: list of angle_or_None, chronological. Returns index of
    the first sample of the final stable run, or None."""
    valid_idx = [i for i, a in enumerate(angle_seq) if a is not None]
    if len(valid_idx) < 2:
        return None
    final_value = angle_seq[valid_idx[-1]]
    boundary = None
    for j in reversed(valid_idx[:-1]):
        if abs(angle_seq[j] - final_value) > 0.3:
            boundary = j
            break
    if boundary is None:
        return None
    for k in valid_idx:
        if k > boundary:
            return k
    return None


def nearest_angle_index(angles, current):
    return min(range(len(angles)), key=lambda i: abs(angles[i] - current))


def rotate_and_fire(sct, monitor, current_angle, target_angle):
    """Rotate from current_angle to target_angle using the exact same
    control path as aim_solver.execute_intercept, then burst-capture
    across the fire call to find where it truly landed. Returns
    (true_final_angle_or_None, debug_dict)."""
    delta = target_angle - current_angle
    if abs(delta) < 1e-6:
        direction, hold_s = 0, 0.0
    else:
        direction = 1 if delta > 0 else -1
        hold_s = abs(delta) / ROTATION_SPEED_DEG_S

    if direction != 0:
        do_action(1 if direction > 0 else 2)
        sleep_s = max(0.0, hold_s - LATENCY_OFFSET_S + TICK_SAFETY_MARGIN_S)
    else:
        sleep_s = 0.0

    coarse_sleep = max(0.0, sleep_s - PRE_FIRE_TAIL_S)
    if coarse_sleep > 0:
        time.sleep(coarse_sleep)
    target_fire_at = time.time() + max(0.0, sleep_s - coarse_sleep)

    angle_history = []
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
        if fire_issued_t is not None and now - fire_issued_t >= BURST_DURATION_S:
            break
        if len(angle_history) >= MAX_BURST_SAMPLES and fire_issued_t is not None:
            break
    if proc is not None:
        proc.wait(timeout=1.0)

    raw = [(t - fire_issued_t, ba) for t, ba in angle_history]
    fire_idx = find_fire_tick([ba for _, ba in raw])
    true_final = raw[fire_idx][1] if fire_idx is not None else None
    debug = {
        "direction": direction,
        "hold_s": hold_s,
        "sleep_s": sleep_s,
        "fire_t": raw[fire_idx][0] if fire_idx is not None else None,
        "n_samples": len(raw),
        "debug_angles": [round(a, 2) if a is not None else None for _, a in raw],
    }
    return true_final, debug


def main(n_trials=80):
    with open(ANGLES_PATH) as f:
        angles = json.load(f)["angles"]
    n = len(angles)
    print(f"loaded {n} discrete angles from {ANGLES_PATH}")

    trials = []
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        di = 0
        cur_idx = None
        while len(trials) < n_trials:
            frame = grab(sct, monitor)
            if is_done(frame):
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.3)
                cur_idx = None  # unknown resting angle after a restart
                continue

            if cur_idx is None:
                a = fast_barrel_angle(frame)
                if not valid_barrel_angle(a):
                    time.sleep(0.03)
                    continue
                cur_idx = nearest_angle_index(angles, a)

            delta_idx = DELTAS[di % len(DELTAS)]
            di += 1
            target_idx = max(0, min(n - 1, cur_idx + delta_idx))
            if target_idx == cur_idx:
                continue
            current_angle = angles[cur_idx]
            target_angle = angles[target_idx]

            true_final, debug = rotate_and_fire(sct, monitor, current_angle, target_angle)

            error_deg = (true_final - target_angle) if true_final is not None else None
            landed_idx = nearest_angle_index(angles, true_final) if true_final is not None else None
            tick_offset = (landed_idx - target_idx) if landed_idx is not None else None

            trials.append({
                "start_idx": cur_idx,
                "target_idx": target_idx,
                "delta_idx": target_idx - cur_idx,
                "start_angle": current_angle,
                "target_angle": target_angle,
                "true_final_angle": true_final,
                "error_deg": error_deg,
                "landed_idx": landed_idx,
                "tick_offset": tick_offset,
                **debug,
            })
            n_done = len(trials)
            landed_str = "None" if true_final is None else f"{true_final:.3f}"
            err_str = "None" if error_deg is None else f"{error_deg:.3f}"
            print(f"[{n_done:3d}/{n_trials}] {cur_idx:2d}->{target_idx:2d} "
                  f"(delta_idx={target_idx-cur_idx:+3d}) target={target_angle:7.3f} "
                  f"landed={landed_str:>8} err={err_str:>7} tick_off={tick_offset}")

            cur_idx = landed_idx if landed_idx is not None else None
            time.sleep(0.1)

    with open(OUT_PATH, "w") as f:
        json.dump(trials, f)
    print(f"\nsaved {len(trials)} trials to {OUT_PATH}")

    clean = [t for t in trials if t["error_deg"] is not None]
    print(f"clean (fire tick located): {len(clean)}/{len(trials)}")
    errs = np.array([t["error_deg"] for t in clean])
    offs = np.array([t["tick_offset"] for t in clean])
    print(f"error_deg: mean={errs.mean():.3f} std={errs.std():.3f} min={errs.min():.3f} max={errs.max():.3f}")
    print(f"tick_offset: exact(0)={100*np.mean(offs==0):.0f}%  "
          f"+-1={100*np.mean(np.abs(offs)<=1):.0f}%  +-2={100*np.mean(np.abs(offs)<=2):.0f}%")
    from collections import Counter
    print("tick_offset distribution:", dict(sorted(Counter(offs.tolist()).items())))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    main(n)
