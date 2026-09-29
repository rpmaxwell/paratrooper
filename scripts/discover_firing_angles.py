#!/usr/bin/env python3
"""Step 1 of the gun-only accuracy test (plan_accuracy_and_priority.md,
2026-09-22): enumerate the turret's real discrete resting angles directly,
rather than computing them from ROTATION_SPEED_DEG_S/TICK_PERIOD_S. The
game updates the barrel position once per internal tick (established in
aim_solver.py's docstring -- rapid polling shows it holds exactly steady
for a run of frames, then jumps), so while rotating, the barrel can only
ever be observed at one of a finite set of angles between the two hard
limits. Sweep the full range with fast, pivot-cropped polling (same trick
as calibrate_fire_tick.py -- full-frame detect_barrel_angle measured at
~12.6 ms/frame in this container, almost certainly emulation overhead,
too slow to resolve a ~50ms tick) and record every distinct value
encountered, in order. That list *is* the answer -- no formula needed.
"""
import json
import subprocess
import sys
import time

import mss
import numpy as np

from actions import do_action
from read_score import is_done
from threats import PIVOT, BARREL_RADIUS, CYAN, angle_to
from aim_solver import valid_barrel_angle

OUT_PATH = "/captures/firing_angles.json"


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


def reset_if_needed(sct, monitor):
    frame = grab(sct, monitor)
    if is_done(frame):
        subprocess.run(["xdotool", "key", "space"], check=True)
        time.sleep(0.3)


def sweep_and_record(sct, monitor, direction, max_time):
    do_action(direction)
    t0 = time.time()
    samples = []
    while time.time() - t0 < max_time:
        f = grab(sct, monitor)
        t = time.time() - t0
        a = fast_barrel_angle(f)
        samples.append((t, a if valid_barrel_angle(a) else None))
    return samples


def distinct_values(samples, tol=0.15):
    """Collapse consecutive near-identical readings into the sequence of
    genuinely distinct resting angles the barrel passed through."""
    vals = []
    for _, a in samples:
        if a is None:
            continue
        if not vals or abs(a - vals[-1]) > tol:
            vals.append(a)
    return vals


def main(max_time=1.8):
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        reset_if_needed(sct, monitor)
        # Drive to a known limit first so the sweep below starts clean
        # regardless of wherever the barrel happened to be resting.
        sweep_and_record(sct, monitor, 2, max_time)  # rotate_right to the right limit
        reset_if_needed(sct, monitor)
        samples = sweep_and_record(sct, monitor, 1, max_time)  # rotate_left, full sweep

        vals = distinct_values(samples)
        if len(vals) < 3:
            print(f"too few distinct angles captured ({len(vals)}) -- try a longer max_time or check the game state")
            return
        steps = [round(vals[i + 1] - vals[i], 3) for i in range(len(vals) - 1)]

        print(f"n distinct angles: {len(vals)}")
        print("first 10:", [round(v, 3) for v in vals[:10]])
        print("last 10:", [round(v, 3) for v in vals[-10:]])
        print(f"range: {vals[0]:.3f} to {vals[-1]:.3f}")
        print(f"step sizes (deg): mean={np.mean(steps):.4f} std={np.std(steps):.4f} "
              f"min={np.min(steps):.4f} max={np.max(steps):.4f}")
        # flag any step that's a clear outlier (e.g. a truncated final
        # step against a limit, or a missed/merged tick from a detection
        # gap) rather than silently averaging over it
        med = np.median(steps)
        outliers = [(i, s) for i, s in enumerate(steps) if abs(s - med) > 0.5]
        if outliers:
            print(f"non-uniform steps (index, size) vs median {med:.4f}: {outliers}")

        with open(OUT_PATH, "w") as f:
            json.dump({"angles": vals, "steps": steps}, f)
        print(f"saved {len(vals)} angles to {OUT_PATH}")


if __name__ == "__main__":
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 1.8
    main(t)
