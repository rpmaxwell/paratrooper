#!/usr/bin/env python3
"""Stage 2 validation gate: for each non-noop action, screenshot before,
send the action, screenshot after. Inspect the pairs in /captures manually
to confirm the turret actually responded as expected -- this is how we
catch a wrong key binding before it contaminates training data.
"""
import time

import mss
import mss.tools

from actions import ACTION_NAMES, do_action


def grab(path: str) -> None:
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        img = sct.grab(monitor)
        mss.tools.to_png(img.rgb, img.size, output=path)


for action in (1, 2, 3):
    name = ACTION_NAMES[action]
    grab(f"/captures/action_{action}_{name}_before.png")
    time.sleep(0.1)
    do_action(action)
    time.sleep(0.2)
    grab(f"/captures/action_{action}_{name}_after.png")
    print(f"action {action} ({name}): before/after pair saved")
