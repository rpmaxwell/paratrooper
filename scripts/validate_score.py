#!/usr/bin/env python3
"""Stage 3 validation gate: play (scripted here; works the same for
manual play over VNC) while logging read_score()/is_done() every
frame. Prints a running log and a summary of score transitions and
where done fired, so it can be eyeballed against what actually
happened on screen.
"""
import random
import time

import numpy as np
import mss

from actions import do_action
from read_score import read_score, is_done


def grab_frame(sct, monitor) -> np.ndarray:
    img = np.array(sct.grab(monitor))
    return img[:, :, :3][:, :, ::-1]  # BGRA -> RGB


def main(duration_s: float = 90.0):
    random.seed(42)
    last_score = None
    transitions = []
    done_events = []

    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        t_start = time.time()
        step = 0
        while time.time() - t_start < duration_s:
            action = random.choice([1, 2, 3, 3, 3])
            do_action(action)
            time.sleep(0.1)

            frame = grab_frame(sct, monitor)
            score = read_score(frame)
            done = is_done(frame)

            if score != last_score:
                transitions.append((step, last_score, score))
                last_score = score
            if done:
                done_events.append(step)
                do_action(0)  # noop; will press space below to keep the loop moving
                import subprocess
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.5)

            step += 1

    print(f"ran {step} steps over {duration_s:.0f}s")
    print(f"score transitions ({len(transitions)}):")
    for s in transitions:
        print(" ", s)
    print(f"done fired at steps: {done_events}")


if __name__ == "__main__":
    main()
