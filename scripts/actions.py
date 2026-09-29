#!/usr/bin/env python3
"""Stage 2: discrete action space for Paratrooper, injected via xdotool
against the Xvfb DISPLAY.

Key bindings below are taken verbatim from the game's own in-game
instructions screen (press 'I' at the title screen):
    Left / 4  -> counterclockwise (start turret moving)
    Right / 6 -> clockwise (start turret moving)
    Up / 8    -> STOPS the turret's movement AND fires

Note the last point: rotating and firing are mutually exclusive by
design -- firing requires the turret to stop turning first. This
answers the PRD's open question about simultaneous rotate+fire (it's
not possible; no combo actions needed).
"""
import os
import subprocess

KEYS = {
    1: "Left",   # rotate turret left (counterclockwise)
    2: "Right",  # rotate turret right (clockwise)
    3: "Up",     # stop + fire
}

ACTION_NAMES = {
    0: "noop",
    1: "rotate_left",
    2: "rotate_right",
    3: "fire",
}


def do_action(action: int, hold: float = None) -> None:
    """Send one discrete action. 0 is a no-op."""
    if action == 0:
        return
    if hold is None:
        # Overridable for latency experiments (see calibrate_latency.py)
        # without touching the default used by training/live-play
        # processes -- read fresh each call, not baked in at import time.
        hold = float(os.environ.get("PARATROOPER_KEY_DELAY_MS", "50")) / 1000.0
    key = KEYS[action]
    subprocess.run(
        ["xdotool", "key", "--delay", str(int(hold * 1000)), key],
        check=True,
    )


if __name__ == "__main__":
    import sys

    action = int(sys.argv[1])
    do_action(action)
    print(f"sent action {action} ({ACTION_NAMES[action]})")
