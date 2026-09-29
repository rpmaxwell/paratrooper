#!/usr/bin/env python3
"""Closed-loop turret control on the 19 discrete barrel positions
(bomb_model.BARREL). The turret steps ~one position per game tick while
rotating; Up stops it and fires. goto() watches the barrel every frame
and presses Up `lead` positions before the target to absorb key latency.

Run directly to calibrate `lead`: python3 turret.py [n_trials]
"""
import random
import subprocess
import sys
import time

import mss
import numpy as np

from bomb_model import BARREL, LIMIT_SIGHT_LANE, LIMIT_STATE, barrel_pos, to_index

N_POS = len(BARREL)
# positions of lead before pressing Up. Calibrated (python3 turret.py 60,
# 2026-09-25): lead 0 stopped exactly on target 16/17 times, lead 1 was
# always one short -- the stop takes effect before the next step.
GOTO_LEAD = 0


# Every Up press (each one fires a bullet), for offline shot audits:
# (time, context). Stops made by goto/clamp are kind "stop".
SHOTS = []


def key(k, hold_ms=20, ctx=None):
    if k == "Up":
        SHOTS.append((time.time(), ctx or {"kind": "stop"}))
    subprocess.run(["xdotool", "key", "--delay", str(hold_ms), k], check=True)


def grab_idx(sct, mon):
    return to_index(np.array(sct.grab(mon))[:, :, :3][:, :, ::-1])


def read_pos(sct, mon, tries=6):
    for _ in range(tries):
        p = barrel_pos(grab_idx(sct, mon))
        if p is not None:
            return p
    return None


def goto(sct, mon, target, lead=None, timeout=2.0, on_frame=None, corrections=1):
    """Rotate to `target` and stop there (the stop fires one bullet).
    on_frame(idx) is called for every frame grabbed while rotating, so a
    caller can keep tracking bombs; a truthy return stops the turret right
    there. Returns the final position read."""
    lead = GOTO_LEAD if lead is None else lead
    cur = read_pos(sct, mon)
    if cur is None or cur == target:
        return cur
    step = 1 if target > cur else -1
    key("Left" if step > 0 else "Right")  # Left = counterclockwise = higher angle
    t0 = time.time()
    while time.time() - t0 < timeout:
        idx = grab_idx(sct, mon)
        # check the barrel BEFORE the caller's per-frame work: the turret
        # steps every ~55ms, so any processing ahead of this check makes
        # the stop land a position late (seen live: 14->13 overshot to 12)
        p = barrel_pos(idx)
        if p is not None and (target - p) * step <= lead:
            break
        if on_frame and on_frame(idx):
            break  # caller asked to stop early
    key("Up")
    # stopped on sight: if this is a rotation limit, the turret is NOT
    # clamped into the stop, so it fires the steeper limit lane
    LIMIT_STATE["clamped"] = False
    time.sleep(0.08)
    final = read_pos(sct, mon)
    if final is not None and final != target and corrections > 0:
        return goto(sct, mon, target, lead, timeout, on_frame, corrections - 1)
    return final


CLAMP_HOLD_S = 0.2  # calibrate_limit_lanes.py: 0.2s into the stop -> flat lane 23/23
CLAMP_TICKS = 5    # planning cost of clamp() in game ticks (hold + key latency)


def clamp(sct, mon, limit):
    """At a rotation limit, push into the stop so the turret fires the
    flatter (+-20, -4) lane instead of the (+-20, -6) one. The closing Up
    fires a bullet along the clamped lane."""
    key("Left" if limit == len(BARREL) - 1 else "Right")
    time.sleep(CLAMP_HOLD_S)
    key("Up")
    LIMIT_STATE["clamped"] = True


def calibrate(n=40):
    from read_score import is_done
    results = {0: [], 1: [], 2: []}
    with mss.mss() as sct:
        mon = sct.monitors[1]
        for i in range(n):
            f = np.array(sct.grab(mon))[:, :, :3][:, :, ::-1]
            if is_done(f):
                key("space")
                time.sleep(1.0)
            lead = i % 3
            cur = read_pos(sct, mon)
            if cur is None:
                continue
            target = random.choice([p for p in range(N_POS) if abs(p - cur) >= 2])
            final = goto(sct, mon, target, lead=lead)
            err = None if final is None else final - target
            if err is not None:
                results[lead].append(err * (1 if target > cur else -1))  # + = overshoot
            print(f"lead={lead} {cur}->{target} final={final} overshoot={results[lead][-1] if err is not None else None}",
                  flush=True)
    for lead, errs in results.items():
        print(f"lead {lead}: overshoot counts {sorted((e, errs.count(e)) for e in set(errs))}")


if __name__ == "__main__":
    calibrate(int(sys.argv[1]) if len(sys.argv) > 1 else 45)
