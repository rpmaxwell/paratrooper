#!/usr/bin/env python3
"""Validation gate for threats.py, same pattern as validate_score.py: play
while logging every frame's detections, and dump annotated screenshots so
the pivot/barrel-angle/threat/walker detection can be eyeballed against what
actually happened on screen before it's trusted as a reward-shaping input.

Run this against the live game before wiring threats.py into training.
"""
import random
import time

import mss
import numpy as np
from PIL import Image, ImageDraw

from actions import do_action
from read_score import is_done, read_score
from threats import (
    PIVOT,
    count_walkers_per_side,
    detect_barrel_angle,
    nearest_threat_angle_gap,
    threat_points,
)

OUT_DIR = "/captures/validate_threats"
ANNOTATE_EVERY = 15  # dump an annotated frame this often (steps)


def grab_frame(sct, monitor) -> np.ndarray:
    img = np.array(sct.grab(monitor))
    return img[:, :, :3][:, :, ::-1]  # BGRA -> RGB


def annotate(frame: np.ndarray, barrel_angle, threats, left, right, gap) -> Image.Image:
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    px, py = PIVOT
    r = 3
    draw.ellipse([px - r, py - r, px + r, py + r], outline=(255, 0, 0), width=2)
    if barrel_angle is not None:
        import math

        length = 60
        ex = px + length * math.cos(math.radians(barrel_angle))
        ey = py - length * math.sin(math.radians(barrel_angle))
        draw.line([px, py, ex, ey], fill=(255, 0, 0), width=1)
    # raw threat pixels (not blob centroids -- see threats.py's v2 docstring)
    for tx, ty in threats:
        draw.point((tx, ty), fill=(255, 255, 0))
    draw.text((10, 620), f"barrel={barrel_angle} gap={gap} left={left} right={right}", fill=(255, 0, 0))
    return img


def main(duration_s: float = 90.0):
    import os

    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(42)
    last_left, last_right = 0, 0
    anomalies = []

    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        t_start = time.time()
        step = 0
        while time.time() - t_start < duration_s:
            action = random.choice([1, 2, 3, 3])
            do_action(action)
            time.sleep(0.1)

            frame = grab_frame(sct, monitor)
            barrel_angle = detect_barrel_angle(frame)
            gap = nearest_threat_angle_gap(frame, barrel_angle)
            threats = threat_points(frame)
            left, right = count_walkers_per_side(frame)
            score = read_score(frame)
            done = is_done(frame)

            if barrel_angle is None:
                anomalies.append((step, "no barrel detected"))
            if left < last_left or right < last_right:
                # expected sometimes (a walker is shot or captures the base
                # and disappears) -- just logged, not an error
                pass
            if left - last_left > 1 or right - last_right > 1:
                anomalies.append((step, f"landed count jumped >1 in a step: "
                                         f"({last_left},{last_right})->({left},{right})"))
            last_left, last_right = left, right

            print(f"step={step} score={score} done={done} barrel={barrel_angle} "
                  f"gap={gap} threats={len(threats)} left={left} right={right}")

            if step % ANNOTATE_EVERY == 0:
                annotate(frame, barrel_angle, threats, left, right, gap).save(
                    f"{OUT_DIR}/frame_{step:04d}.png"
                )

            if done:
                import subprocess

                subprocess.run(["xdotool", "key", "space"], check=True)
                time.sleep(0.5)
                last_left, last_right = 0, 0

            step += 1

    print(f"\nran {step} steps over {duration_s:.0f}s")
    print(f"anomalies ({len(anomalies)}):")
    for a in anomalies:
        print(" ", a)
    print(f"annotated frames written to {OUT_DIR}/ -- eyeball a few against the logged values")


if __name__ == "__main__":
    main()
