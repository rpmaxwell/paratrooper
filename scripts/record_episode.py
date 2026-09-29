#!/usr/bin/env python3
"""Play one or more episodes with the existing phase-1 paratrooper defense
while a background thread records EVERY frame into a ring buffer. On game
over, dump the last BUFFER_S seconds to an .npz so the phase-2 (planes +
bombs) sequence can be studied offline, frame by frame, instead of
inferred from heuristics.

Frames are stored palette-indexed (one uint8 per pixel) over the game
area only, to keep the buffer small.

Usage: python3 record_episode.py [n_episodes] [out_dir]
"""
import os
import subprocess
import sys
import threading
import time
from collections import deque

import mss
import numpy as np

from actions import do_action
from aim_solver import LATENCY_OFFSET_S, TICK_SAFETY_MARGIN_S, solve_intercept, valid_barrel_angle
from play_one_game import find_paratrooper_target, grab
from read_score import is_done, read_score
from threats import detect_barrel_angle

# game area in the 1024x768 Xvfb frame (below the DOSBox-X menu bar)
GX0, GX1, GY0, GY1 = 192, 832, 200, 600
PALETTE = np.array([[0, 0, 0], [85, 255, 255], [255, 85, 255], [255, 255, 255]], dtype=np.uint8)
BUFFER_S = 40.0


def to_index(frame):
    crop = frame[GY0:GY1, GX0:GX1]
    idx = np.zeros(crop.shape[:2], dtype=np.uint8)
    for i, c in enumerate(PALETTE[1:], start=1):
        idx[np.all(crop == c, axis=-1)] = i
    return idx


class Recorder(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.buf = deque()
        self.lock = threading.Lock()
        self.stop_flag = False

    def run(self):
        with mss.mss() as sct:
            mon = sct.monitors[1]
            while not self.stop_flag:
                t = time.time()
                f = grab(sct, mon)
                idx = to_index(f)
                with self.lock:
                    self.buf.append((t, idx))
                    while self.buf and t - self.buf[0][0] > BUFFER_S:
                        self.buf.popleft()

    def snapshot(self):
        with self.lock:
            return list(self.buf)


def play_until_done(sct, mon):
    while True:
        frame = grab(sct, mon)
        if is_done(frame):
            return read_score(frame)
        target = find_paratrooper_target(frame)
        if target is None:
            time.sleep(0.05)
            continue
        barrel_angle = detect_barrel_angle(frame)
        if not valid_barrel_angle(barrel_angle):
            time.sleep(0.03)
            continue
        solution = solve_intercept(barrel_angle, *target)
        if solution is None:
            time.sleep(0.05)
            continue
        direction, hold_s = solution
        if direction != 0:
            do_action(1 if direction > 0 else 2)
            time.sleep(max(0.0, hold_s - LATENCY_OFFSET_S + TICK_SAFETY_MARGIN_S))
        do_action(3)
        time.sleep(0.05)


def main(n_episodes=1, out_dir="/captures/episodes"):
    os.makedirs(out_dir, exist_ok=True)
    rec = Recorder()
    rec.start()
    with mss.mss() as sct:
        mon = sct.monitors[1]
        for ep in range(n_episodes):
            if is_done(grab(sct, mon)):
                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.5)
            t0 = time.time()
            score = play_until_done(sct, mon)
            time.sleep(0.3)  # let a few game-over frames land in the buffer
            frames = rec.snapshot()
            ts = np.array([t for t, _ in frames]) - t0
            path = f"{out_dir}/ep_{int(t0)}.npz"
            np.savez_compressed(path, t=ts, frames=np.stack([f for _, f in frames]))
            fps = len(frames) / max(1e-6, ts[-1] - ts[0])
            print(f"episode {ep}: score={score} dur={time.time()-t0:.1f}s "
                  f"saved {len(frames)} frames ({fps:.1f} fps) -> {path}", flush=True)
    rec.stop_flag = True


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out = sys.argv[2] if len(sys.argv) > 2 else "/captures/episodes"
    main(n, out)
