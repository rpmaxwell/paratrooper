#!/usr/bin/env python3
"""Stage 1.1: tight rotation-speed calibration. Repeatedly sweeps the
turret full-range in each direction, polling as fast as frame-grab allows,
and reports a per-sweep linear-fit speed plus a check for non-uniformity
(does it decelerate near the limits, or hold constant edge-to-edge?).

Run against the idle/dedicated instance only (sends real key input).
"""
import subprocess
import sys
import time

import mss
import numpy as np

from actions import do_action
from read_score import is_done
from threats import detect_barrel_angle


def grab(sct, monitor):
    return np.array(sct.grab(monitor))[:, :, :3][:, :, ::-1]


def reset_if_needed(sct, monitor):
    frame = grab(sct, monitor)
    if is_done(frame):
        subprocess.run(["xdotool", "key", "space"], check=True)
        time.sleep(0.3)
        frame = grab(sct, monitor)
    return frame


def sweep(sct, monitor, direction, sweep_time, poll):
    """direction: 1=left(rotate_left action), 2=right. Returns list of
    (t, angle) samples across the sweep."""
    do_action(direction)
    t0 = time.time()
    samples = []
    while time.time() - t0 < sweep_time:
        frame = grab(sct, monitor)
        t = time.time() - t0
        angle = detect_barrel_angle(frame)
        samples.append((t, angle))
        time.sleep(poll)
    return samples


def analyze(samples, label):
    valid = [(t, a) for t, a in samples if a is not None]
    if len(valid) < 5:
        print(f"{label}: not enough valid samples ({len(valid)})")
        return None
    final_angle = valid[-1][1]
    moving = [(t, a) for t, a in valid if abs(a - final_angle) > 1.0]
    if len(moving) < 5:
        print(f"{label}: not enough moving samples ({len(moving)})")
        return None
    ts = np.array([v[0] for v in moving])
    angs = np.array([v[1] for v in moving])
    slope, intercept = np.polyfit(ts, angs, 1)

    # Non-uniformity check: split the moving segment into first/middle/last
    # thirds and fit each separately -- a truly constant-speed motor should
    # give ~equal slopes across all three; deceleration near a limit would
    # show up as a shrinking |slope| in the last third.
    n = len(ts)
    third = max(n // 3, 2)
    segs = {
        "first_third": (ts[:third], angs[:third]),
        "middle_third": (ts[third : 2 * third], angs[third : 2 * third]),
        "last_third": (ts[2 * third :], angs[2 * third :]),
    }
    seg_slopes = {}
    for name, (st, sa) in segs.items():
        if len(st) >= 2:
            s, _ = np.polyfit(st, sa, 1)
            seg_slopes[name] = s

    span = moving[0][1] - moving[-1][1]
    dt = moving[-1][0] - moving[0][0]
    endpoint_speed = span / dt if dt > 0 else float("nan")
    print(
        f"{label}: linear-fit={slope:+.2f} deg/s  endpoint={endpoint_speed:+.2f} deg/s  "
        f"n={n}  segs={ {k: round(v, 1) for k, v in seg_slopes.items()} }"
    )
    return {"linear_fit": slope, "endpoint": endpoint_speed, "segs": seg_slopes}


def main(n_cycles=10, sweep_time=1.6, poll=0.02):
    right_results = []
    left_results = []
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        reset_if_needed(sct, monitor)
        for i in range(n_cycles):
            reset_if_needed(sct, monitor)
            r_samples = sweep(sct, monitor, 2, sweep_time, poll)  # rotate_right
            r = analyze(r_samples, f"cycle {i} RIGHT")
            if r:
                right_results.append(r)

            reset_if_needed(sct, monitor)
            l_samples = sweep(sct, monitor, 1, sweep_time, poll)  # rotate_left
            l = analyze(l_samples, f"cycle {i} LEFT ")
            if l:
                left_results.append(l)

    def summarize(results, label):
        if not results:
            print(f"{label}: no valid results")
            return
        speeds = [abs(r["linear_fit"]) for r in results]
        print(
            f"\n{label} summary (n={len(speeds)}): mean={np.mean(speeds):.2f} "
            f"std={np.std(speeds):.2f} min={np.min(speeds):.2f} max={np.max(speeds):.2f} "
            f"deg/sec"
        )

    summarize(right_results, "RIGHT")
    summarize(left_results, "LEFT")
    all_speeds = [abs(r["linear_fit"]) for r in right_results + left_results]
    if all_speeds:
        print(
            f"\nCOMBINED (n={len(all_speeds)}): mean={np.mean(all_speeds):.3f} "
            f"std={np.std(all_speeds):.3f} deg/sec"
        )


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(n)
