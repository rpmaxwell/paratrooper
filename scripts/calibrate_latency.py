#!/usr/bin/env python3
"""Stage 1.2: isolate LATENCY_OFFSET_S directly, instead of inferring it
from noisy hit/miss outcomes in practice runs.

For each trial: read the barrel angle while stationary (angle0), issue one
rotate action, then rapid-poll the barrel angle for a short window
immediately afterward. Because rotation is constant-velocity (Stage 1.1),
we can fit a line to the post-action angle samples and extrapolate it
backward to the exact instant we *called* do_action() -- the gap between
that extrapolated angle and the true stationary angle0 is rotation that
happened "for free" during the do_action() call itself (xdotool subprocess
spawn + key delay + X11 event delivery), expressed directly in seconds via
the known rotation speed.

Also supports overriding actions.py's xdotool --delay via
PARATROOPER_KEY_DELAY_MS, to test whether a shorter delay still reliably
registers with the game (reducing latency at the source, not just
compensating for it after the fact).
"""
import os
import subprocess
import sys
import time

import mss
import numpy as np

from actions import do_action
from read_score import is_done
from threats import detect_barrel_angle

ROTATION_SPEED_DEG_S = 128.23  # Stage 1.1 result


def grab(sct, monitor):
    return np.array(sct.grab(monitor))[:, :, :3][:, :, ::-1]


def reset_if_needed(sct, monitor):
    frame = grab(sct, monitor)
    if is_done(frame):
        subprocess.run(["xdotool", "key", "space"], check=True)
        time.sleep(0.3)
        frame = grab(sct, monitor)
    return frame


def settle_to_known_angle(sct, monitor, direction, settle_time=1.8):
    """Rotate hard into a limit so every trial starts from a known,
    stationary angle (avoids drift/limit-adjacency edge effects)."""
    do_action(direction)
    time.sleep(settle_time)


def one_trial(sct, monitor, direction, n_samples=12, sample_interval=0.02):
    frame = reset_if_needed(sct, monitor)
    angle0 = detect_barrel_angle(frame)
    if angle0 is None:
        return None

    t_call = time.time()
    do_action(direction)  # this call itself blocks for ~key-delay ms
    samples = []
    for _ in range(n_samples):
        f = grab(sct, monitor)
        t = time.time() - t_call
        a = detect_barrel_angle(f)
        samples.append((t, a))
        time.sleep(sample_interval)

    valid = [(t, a) for t, a in samples if a is not None and abs(a - angle0) > 0.3]
    if len(valid) < 4:
        return None
    ts = np.array([v[0] for v in valid])
    angs = np.array([v[1] for v in valid])
    slope, intercept = np.polyfit(ts, angs, 1)  # angle = slope*t + intercept
    virtual_angle_at_call = intercept  # t=0 is t_call
    sign = 1 if direction == 1 else -1
    # The fitted "motion" line necessarily undershoots angle0 when
    # extrapolated back to the call instant (real motion only starts after
    # a latency L, so the fitted line's t=0 value sits speed*L before
    # angle0) -- latency_s recovers L from that gap.
    angle_gained = sign * (virtual_angle_at_call - angle0)
    latency_s = -angle_gained / ROTATION_SPEED_DEG_S
    if not (0.0 <= latency_s <= 0.2):
        return None  # corrupted read (e.g. limit/reset glitch), discard
    return {
        "angle0": angle0,
        "slope": slope,
        "virtual_angle_at_call": virtual_angle_at_call,
        "angle_gained": angle_gained,
        "latency_s": latency_s,
        "n_valid": len(valid),
    }


def main(n_trials=20, key_delay_ms=None):
    if key_delay_ms is not None:
        os.environ["PARATROOPER_KEY_DELAY_MS"] = str(key_delay_ms)
    results = []
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        reset_if_needed(sct, monitor)
        for i in range(n_trials):
            direction = 1 if i % 2 == 0 else 2
            # settle away from the limit we're about to sweep toward so we
            # get a clean stationary read, not one contaminated by residual
            # motion from the previous trial
            settle_to_known_angle(sct, monitor, 2 if direction == 1 else 1, settle_time=1.2)
            r = one_trial(sct, monitor, direction)
            if r:
                results.append(r)
                print(
                    f"trial {i:2d} dir={direction}: angle0={r['angle0']:6.2f} "
                    f"slope={r['slope']:7.2f} virtual_angle0={r['virtual_angle_at_call']:6.2f} "
                    f"angle_gained={r['angle_gained']:+6.2f} latency={r['latency_s']*1000:6.1f}ms "
                    f"n={r['n_valid']}"
                )
            else:
                print(f"trial {i:2d} dir={direction}: SKIPPED (insufficient samples or corrupted read)")

    if results:
        lat = np.array([r["latency_s"] for r in results])
        print(f"\nn={len(lat)} latency: mean={lat.mean()*1000:.1f}ms std={lat.std()*1000:.1f}ms "
              f"min={lat.min()*1000:.1f}ms max={lat.max()*1000:.1f}ms")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    delay = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(n, delay)
