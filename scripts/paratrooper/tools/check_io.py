"""Phase 1 gate: native capture parity + speed, and key latency
(XTEST vs xdotool). Run in the container with the game on screen:
    python3 -m paratrooper.tools.check_io
"""
import subprocess
import time

import mss
import numpy as np

from ..geometry import SCREEN_X0, SCREEN_Y0
from ..io.keys import Keys
from ..io.screen import Screen
from ..perception.barrel import barrel_pos


def old_to_index(full_rgb):
    crop = full_rgb[200:600, 192:832]
    idx = np.zeros(crop.shape[:2], np.uint8)
    for i, c in enumerate(([85, 255, 255], [255, 85, 255], [255, 255, 255]), start=1):
        idx[np.all(crop == c, axis=-1)] = i
    return idx


def parity(n=40):
    bad = 0
    with mss.mss() as sct:
        for _ in range(n):
            img = sct.grab(sct.monitors[1])
            bgra = np.frombuffer(img.raw, np.uint8).reshape(img.height, img.width, 4)
            full_rgb = bgra[..., [2, 1, 0]]
            old = old_to_index(full_rgb)[1::2, ::2]
            reg = bgra[SCREEN_Y0:SCREEN_Y0 + 400, SCREEN_X0:SCREEN_X0 + 640][::2, ::2]
            new = ((reg[..., 2] > 127).astype(np.uint8) << 1) | (reg[..., 1] > 127)
            bad += not np.array_equal(old, new)
            time.sleep(0.05)
    print(f"capture parity: {n - bad}/{n} frames identical to old path (downsampled)")


def speed(n=300):
    s = Screen()
    ts = []
    for _ in range(n):
        a = time.perf_counter()
        s.grab()
        ts.append(time.perf_counter() - a)
    ts = np.array(ts) * 1000
    print(f"grab+convert: median {np.median(ts):.2f} ms, p99 {np.percentile(ts, 99):.2f} ms")


def key_latency(keys, screen, sender, n=12):
    """time from sending Left/Right until the barrel sprite changes."""
    out = []
    for i in range(n):
        _, f, _ = screen.grab()
        p0 = barrel_pos(f)
        if p0 is None:
            sender("Up")
            time.sleep(0.2)
            continue
        k = "Left" if p0 < 9 else "Right"
        t0 = time.time()
        sender(k)
        while time.time() - t0 < 0.5:
            keys.pump()
            _, f, _ = screen.grab()
            if barrel_pos(f) not in (p0, None):
                out.append(time.time() - t0)
                break
        sender("Up")
        t1 = time.time()
        while time.time() - t1 < 0.15:
            keys.pump()
        time.sleep(0.25)
    return np.array(out) * 1000


def main():
    parity()
    speed()
    keys, screen = Keys(), Screen()
    keys.tap("space")  # start a game if on the title / game-over screen
    time.sleep(1.5)
    x = key_latency(keys, screen, lambda k: keys.press(k))
    d = key_latency(keys, screen, lambda k: subprocess.run(["xdotool", "key", "--delay", "20", k], check=True))
    for name, v in (("XTEST", x), ("xdotool", d)):
        if len(v):
            print(f"{name:8s} key -> barrel moved: n={len(v)} median {np.median(v):.0f} ms, "
                  f"min {v.min():.0f}, max {v.max():.0f}, std {v.std():.0f}")
    keys.release_all()


if __name__ == "__main__":
    main()
