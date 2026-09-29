#!/usr/bin/env python3
"""Stage 1.3: measure real bullet travel speed by tracking the projectile
itself across rapid consecutive frames after firing at a clean screen (no
threats present, so nothing else could be mistaken for the bullet).

The bullet renders in the same cyan as the barrel/threats. Right after
firing, we rapid-poll frames and look for cyan pixels farther from the
pivot than the barrel tip, within a narrow angular band around the known
firing angle -- that's the bullet, nothing else should match. Track its
distance-from-pivot over time; the slope is bullet speed in px/sec.
"""
import subprocess
import sys
import time

import mss
import numpy as np

from actions import do_action
from read_score import is_done
from threats import CYAN, PIVOT, _mask_coords, detect_barrel_angle, nearest_threat_angle_gap
from aim_solver import valid_barrel_angle, ROTATE_LEFT_LIMIT_DEG, ROTATE_RIGHT_LIMIT_DEG

ANGLE_TOLERANCE_DEG = 6.0
CLEAR_CORRIDOR_DEG = 20.0  # require no other threat within this angular gap of the fire angle
MAX_TRACK_RADIUS = 400.0  # ignore anything absurdly far (screen-edge noise)


def grab(sct, monitor):
    return np.array(sct.grab(monitor))[:, :, :3][:, :, ::-1]


def reset_if_needed(sct, monitor):
    frame = grab(sct, monitor)
    if is_done(frame):
        subprocess.run(["xdotool", "key", "space"], check=True)
        time.sleep(0.3)
        frame = grab(sct, monitor)
    return frame


def corridor_is_clear(frame, angle):
    """True if no existing threat sits within CLEAR_CORRIDOR_DEG of angle --
    a full empty-screen requirement basically never happens under
    continuous play, but a clear angular corridor around our chosen fire
    angle is enough to isolate the bullet unambiguously."""
    gap = nearest_threat_angle_gap(frame, angle)
    return gap is None or gap > CLEAR_CORRIDOR_DEG


def bullet_distance(frame, fire_angle, barrel_radius_min=50.0):
    """Farthest cyan pixel beyond barrel_radius_min, within
    ANGLE_TOLERANCE_DEG of fire_angle -- the bullet, if present."""
    xs, ys = _mask_coords(frame, CYAN, y0=0)
    if xs.size == 0:
        return None
    px, py = PIVOT
    dist = np.hypot(xs - px, ys - py)
    angles = np.degrees(np.arctan2(py - ys, xs - px))
    gap = np.abs(((angles - fire_angle + 180) % 360) - 180)
    mask = (dist > barrel_radius_min) & (dist < MAX_TRACK_RADIUS) & (gap < ANGLE_TOLERANCE_DEG)
    if not mask.any():
        return None
    return float(dist[mask].max())


def one_trial(sct, monitor, target_angle_frac):
    """target_angle_frac in [0,1]: 0=right limit, 1=left limit -- picks a
    reachable angle away from both limits to fire at."""
    # settle at right limit then rotate left partway for a known-ish angle
    do_action(2)
    time.sleep(1.3)
    frame = reset_if_needed(sct, monitor)
    angle0 = detect_barrel_angle(frame)
    if not valid_barrel_angle(angle0):
        return None
    span = ROTATE_LEFT_LIMIT_DEG - ROTATE_RIGHT_LIMIT_DEG
    hold = (span * target_angle_frac) / 128.23
    do_action(1)
    time.sleep(hold)

    frame = grab(sct, monitor)
    if is_done(frame):
        return None  # round ended mid-trial (pileup elsewhere) -- skip, reset next trial
    fire_angle = detect_barrel_angle(frame)
    if not valid_barrel_angle(fire_angle):
        return None
    if not corridor_is_clear(frame, fire_angle):
        return None

    t0 = time.time()
    do_action(3)  # fire
    samples = []
    while time.time() - t0 < 0.6:
        f = grab(sct, monitor)
        t = time.time() - t0
        d = bullet_distance(f, fire_angle)
        samples.append((t, d))

    pts = [(t, d) for t, d in samples if d is not None]
    if len(pts) < 3:
        return None
    ts = np.array([p[0] for p in pts])
    ds = np.array([p[1] for p in pts])
    slope, intercept = np.polyfit(ts, ds, 1)
    resid = ds - (slope * ts + intercept)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return {
        "fire_angle": fire_angle,
        "n": len(pts),
        "speed": slope,
        "rmse": rmse,
        "d_range": (float(ds.min()), float(ds.max())),
        "raw": list(zip(ts.tolist(), ds.tolist())),
    }


def main(n_trials=15):
    results = []
    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        reset_if_needed(sct, monitor)
        for i in range(n_trials):
            frac = 0.2 + 0.6 * ((i % 5) / 4.0)  # vary angle across trials
            r = one_trial(sct, monitor, frac)
            if r:
                results.append(r)
                print(
                    f"trial {i:2d}: fire_angle={r['fire_angle']:6.1f} n={r['n']:2d} "
                    f"speed={r['speed']:7.1f} px/s rmse={r['rmse']:.2f} d_range={r['d_range']}"
                )
            else:
                print(f"trial {i:2d}: no clean bullet track (screen busy or nothing detected)")
                do_action(3)  # clear a bit of backlog before retrying
            time.sleep(0.3)

    if results:
        speeds = np.array([r["speed"] for r in results])
        print(f"\nn={len(speeds)} bullet speed: mean={speeds.mean():.1f} std={speeds.std():.1f} "
              f"min={speeds.min():.1f} max={speeds.max():.1f} px/s")
        clean = [r for r in results if r["rmse"] < 30]
        if clean:
            cs = np.array([r["speed"] for r in clean])
            print(f"clean (rmse<30, n={len(cs)}): mean={cs.mean():.1f} std={cs.std():.1f} px/s")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(n)
