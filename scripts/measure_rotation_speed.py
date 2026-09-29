#!/usr/bin/env python3
"""Calibration: measure turret angular rotation speed (deg/sec) and the
angle limits it stops at. Sends ONE rotate action then polls barrel angle
over time without further input -- active (sends real key input), so run
this against an idle/dedicated instance, never the live training container.
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


def main(direction=1, poll=0.05, duration=6.0):
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        frame = grab(sct, monitor)
        if is_done(frame):
            subprocess.run(["xdotool", "key", "space"], check=True)
            time.sleep(0.3)
            frame = grab(sct, monitor)

        angle0 = detect_barrel_angle(frame)
        print(f"t=0.000 angle={angle0}")
        do_action(direction)  # 1=left, 2=right
        t0 = time.time()
        samples = [(0.0, angle0)]
        while time.time() - t0 < duration:
            frame = grab(sct, monitor)
            angle = detect_barrel_angle(frame)
            t = time.time() - t0
            samples.append((t, angle))
            print(f"t={t:.3f} angle={angle}")
            time.sleep(poll)

    # fit only the samples strictly before the plateau (find the last index
    # where angle is still meaningfully different from the final value)
    valid = [(t, a) for t, a in samples if a is not None]
    final_angle = valid[-1][1]
    moving = [(t, a) for t, a in valid if abs(a - final_angle) > 1.0]
    if len(moving) >= 3:
        ts = np.array([v[0] for v in moving])
        angs = np.array([v[1] for v in moving])
        slope, intercept = np.polyfit(ts, angs, 1)
        span = moving[0][1] - moving[-1][1]
        dt = moving[-1][0] - moving[0][0]
        print(f"\nlinear-fit angular speed: {slope:.2f} deg/sec")
        print(f"endpoint-to-endpoint: {span:.2f} deg over {dt:.3f}s = {span/dt:.2f} deg/sec")
    print(f"start angle: {valid[0][1]:.2f}  final (plateau) angle: {final_angle:.2f}")


def round_trip(poll=0.05, sweep_time=1.5):
    """Go to the right limit, confirm the round is still alive, then sweep
    back left and measure that direction too -- avoids the failure mode
    where a long-idle (never-firing) calibration round ends mid-measurement
    and later samples silently read a stray pixel on the title screen."""
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]

        def reset_if_needed():
            frame = grab(sct, monitor)
            if is_done(frame):
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.3)
                frame = grab(sct, monitor)
            return frame

        frame = reset_if_needed()
        print("start angle", detect_barrel_angle(frame))
        do_action(2)  # right
        time.sleep(sweep_time)

        frame = reset_if_needed()  # bail/restart cleanly if the round ended
        print("after right sweep, angle:", detect_barrel_angle(frame), "done:", is_done(frame))

        do_action(1)  # left
        t0 = time.time()
        samples = []
        while time.time() - t0 < sweep_time:
            frame = grab(sct, monitor)
            samples.append((time.time() - t0, detect_barrel_angle(frame), is_done(frame)))
            time.sleep(poll)

    for t, a, d in samples:
        print(f"t={t:.3f} angle={a} done={d}")
    valid = [(t, a) for t, a, d in samples if a is not None and not d]
    if len(valid) >= 3:
        final_angle = valid[-1][1]
        moving = [(t, a) for t, a in valid if abs(a - final_angle) > 1.0]
        if len(moving) >= 3:
            span = moving[-1][1] - moving[0][1]
            dt = moving[-1][0] - moving[0][0]
            print(f"\nleft-sweep endpoint-to-endpoint: {span:.2f} deg over {dt:.3f}s = {span/dt:.2f} deg/sec")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "roundtrip":
        round_trip()
    else:
        direction = int(sys.argv[1]) if len(sys.argv) > 1 else 1
        main(direction)
