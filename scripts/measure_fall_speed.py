#!/usr/bin/env python3
"""Calibration: track airborne cyan threat pixels over time to estimate
paratrooper fall speed (and horizontal drift, if any) in px/sec. Fires
occasionally (action 3) so a long-idle calibration round doesn't end via
accumulated landings mid-measurement -- run against a dedicated instance,
not the live training container.
"""
import subprocess
import time

import mss
import numpy as np

from actions import do_action
from read_score import is_done
from threats import threat_points

# Helicopters carry some cyan sprite detail up around y~227-245 (the
# helicopter band); exclude it so we only look at genuinely falling
# paratroopers, which release lower on the screen. Lowered from 300 to
# catch the pre-canopy-open free-fall segment right after release.
FALLING_Y_MIN = 250


def grab(sct, monitor):
    return np.array(sct.grab(monitor))[:, :, :3][:, :, ::-1]


def main(duration_s=25.0, poll=0.1):
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        frame = grab(sct, monitor)
        if is_done(frame):
            subprocess.run(["xdotool", "key", "space"], check=True)
            time.sleep(0.3)

        t0 = time.time()
        step = 0
        while time.time() - t0 < duration_s:
            frame = grab(sct, monitor)
            if is_done(frame):
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.3)
                continue
            pts = [(x, y) for x, y in threat_points(frame) if y >= FALLING_Y_MIN]
            t = time.time() - t0
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                print(f"t={t:.3f} n={len(pts)} y=[{min(ys):.0f},{max(ys):.0f}] "
                      f"mean=({sum(xs)/len(xs):.1f},{sum(ys)/len(ys):.1f})")
            else:
                print(f"t={t:.3f} n=0")
            if step % 8 == 0:
                do_action(3)  # occasional fire to prevent pile-up ending the round
            else:
                do_action(0)
            step += 1
            time.sleep(poll)


if __name__ == "__main__":
    main()
