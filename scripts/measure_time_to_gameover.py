#!/usr/bin/env python3
"""Stage 5.1: measure real wall-clock seconds from round-start to
game-over under a purely passive (no-op) agent. Used to compare across
dosbox-x cycles/turbo settings -- if raising cycles shortens this,
the game is cycle-bound and Stage 5.1 helps; if not, it's wall-clock
/timer-bound and we should rely on Stage 5.2 (parallel instances).
"""
import subprocess
import time

import mss
import numpy as np

from read_score import is_done


def grab(sct, monitor):
    img = np.array(sct.grab(monitor))
    return img[:, :, :3][:, :, ::-1]


def main(poll_interval=0.1, max_wait=120.0):
    subprocess.run(["xdotool", "key", "space"], check=True)
    time.sleep(0.3)

    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        frame = grab(sct, monitor)
        if is_done(frame):
            print("still at title after space -- retrying")
            subprocess.run(["xdotool", "key", "space"], check=True)
            time.sleep(0.3)

        t0 = time.time()
        while time.time() - t0 < max_wait:
            frame = grab(sct, monitor)
            if is_done(frame):
                elapsed = time.time() - t0
                print(f"game over after {elapsed:.2f}s (passive, no actions)")
                return elapsed
            time.sleep(poll_interval)
    print("timed out waiting for game over")
    return None


if __name__ == "__main__":
    main()
