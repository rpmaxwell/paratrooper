#!/usr/bin/env python3
"""Vision-only detection of turret aim angle, airborne-threat proximity, and
landed-walker counts -- used to build reward-shaping terms on top of the raw
score delta.

Paratrooper renders in a 4-color CGA palette (black/white/cyan/magenta), so
plain color masking is enough; no ML detector needed. All coordinates are in
the same 1024x768 Xvfb frame space as read_score.py.

Rewritten (v2) to be fully vectorized numpy, no Python-level blob grouping
in the hot path -- v1 used a per-pixel BFS connected-components pass twice
per env.step(), which measured out at up to ~20 sec/step on a crowded
screen (more sprites -> bigger cyan/white masks -> much slower BFS),
compounding badly since it got slowest exactly when the game was busiest.
None of the three questions this module actually needs to answer (barrel
angle, nearest-threat angle gap, walker count per side) require grouping
pixels into blobs first:
  - barrel angle: distance-from-pivot alone separates the barrel from
    everything else, so take the single farthest in-radius pixel directly.
  - nearest-threat angle gap: a plain per-pixel min over angles is exact,
    no need to reduce to blob centroids first.
  - walker count: a 1D cluster-by-gap over x-coordinates in the fixed torso
    y-band, which only loops over the resulting handful of clusters, not
    over pixels.

Calibration (pivot location, barrel-radius, ground-line/walker-band y
values) was measured directly off captures/*.png -- see that directory for
the source frames. As with every other vision component in this project
(read_score.py, actions.py), treat this as needing its own validation-gate
pass against the live game before trusting it in a training run (see
validate_threats.py).
"""
import math

import numpy as np

CYAN = np.array([85, 255, 255])
WHITE = np.array([255, 255, 255])

# Turret cab (magenta) centroid -- fixed at the same pixel location in every
# captured frame regardless of barrel angle, since only the barrel rotates.
PIVOT = (512.0, 520.0)

# Cyan pixels within this radius of the pivot are the barrel itself; farther
# cyan is a threat (parachute canopy / falling body), not part of the gun.
BARREL_RADIUS = 45.0

# Cyan at/below this y is the ground-line + walker-legs blob, not an airborne
# threat (the ground line sits at y~569-582 and walker legs merge into it).
THREAT_Y_MAX = 555

# A landed/walking soldier's white torso renders as a small blob in this y
# band before its legs merge into the ground-line cyan blob below it. This
# band overlaps the cannon base's own white rectangle (x:480-543,
# y:533-580), so BASE_X0/X1 (with margin) excludes it explicitly -- a
# walker torso is ~4-8px wide vs. the base's ~63px, but excluding by known
# position is simpler and more precise than a width-based cluster filter.
WALKER_Y0, WALKER_Y1 = 558, 572
BASE_X0, BASE_X1 = 478, 546
# x-gap (px) that separates two distinct walkers' torsos on the ground line.
WALKER_CLUSTER_GAP = 3


def _angular_gap(a, b):
    """Smallest angular distance between angles a and b (degrees), handling
    wraparound at +-180 -- a naive abs(a-b) can exceed 180 for angles on
    opposite sides of the +-180 seam (e.g. 179 vs -179 is a 2deg gap, not
    358)."""
    raw = np.abs(a - b)
    return np.minimum(raw, 360.0 - raw)


def angle_to(pivot, point) -> float:
    """Angle in degrees from pivot to point, math convention (0=right,
    90=up, 180=left)."""
    px, py = pivot
    x, y = point
    return math.degrees(math.atan2(py - y, x - px))


def _mask_coords(frame: np.ndarray, color: np.ndarray, y0: int = 0, y1: int = None):
    """xs, ys (float arrays) of pixels matching `color` within [y0, y1)."""
    mask = np.all(frame == color, axis=-1)
    mask[:y0, :] = False
    if y1 is not None:
        mask[y1:, :] = False
    ys, xs = np.where(mask)
    return xs.astype(np.float64), ys.astype(np.float64)


def detect_barrel_angle(frame: np.ndarray):
    """Aim direction in degrees, or None if no cyan is found near the pivot
    (shouldn't happen in practice -- frame capture can occasionally land
    mid-redraw)."""
    xs, ys = _mask_coords(frame, CYAN, y0=184)
    if xs.size == 0:
        return None
    px, py = PIVOT
    dist = np.hypot(xs - px, ys - py)
    near = dist <= BARREL_RADIUS
    if not near.any():
        return None
    i = int(np.argmax(dist[near]))
    bx, by = xs[near][i], ys[near][i]
    return angle_to(PIVOT, (bx, by))


def nearest_threat_angle_gap(frame: np.ndarray, barrel_angle: float = None):
    """Absolute angular gap in degrees between the current barrel aim and
    the nearest airborne threat pixel, or None if there's no barrel
    reading or nothing airborne right now. Pass a pre-computed
    barrel_angle to avoid detecting it twice in the same step."""
    xs, ys = _mask_coords(frame, CYAN, y0=184)
    if xs.size == 0:
        return None
    px, py = PIVOT
    dist = np.hypot(xs - px, ys - py)
    threat = (dist > BARREL_RADIUS) & (ys < THREAT_Y_MAX)
    if not threat.any():
        return None
    if barrel_angle is None:
        barrel_angle = detect_barrel_angle(frame)
        if barrel_angle is None:
            return None
    angles = np.degrees(np.arctan2(py - ys[threat], xs[threat] - px))
    return float(np.min(_angular_gap(angles, barrel_angle)))


def nearest_threat_angle_gap_by_side(frame: np.ndarray, barrel_angle: float = None, pivot_x: float = PIVOT[0]):
    """(left_gap, right_gap): absolute angular gap in degrees between the
    current barrel aim and the nearest airborne threat on each side
    separately, or None for a side with nothing airborne right now."""
    xs, ys = _mask_coords(frame, CYAN, y0=184)
    if xs.size == 0:
        return None, None
    px, py = PIVOT
    dist = np.hypot(xs - px, ys - py)
    threat = (dist > BARREL_RADIUS) & (ys < THREAT_Y_MAX)
    if not threat.any():
        return None, None
    if barrel_angle is None:
        barrel_angle = detect_barrel_angle(frame)
        if barrel_angle is None:
            return None, None
    txs, tys = xs[threat], ys[threat]
    angles = np.degrees(np.arctan2(py - tys, txs - px))
    gaps = _angular_gap(angles, barrel_angle)
    left_mask = txs < pivot_x
    right_mask = ~left_mask
    left_gap = float(np.min(gaps[left_mask])) if left_mask.any() else None
    right_gap = float(np.min(gaps[right_mask])) if right_mask.any() else None
    return left_gap, right_gap


def threat_points(frame: np.ndarray):
    """Raw airborne-threat pixel coordinates, for debug visualization only
    (validate_threats.py). Not used in the reward-computation hot path."""
    xs, ys = _mask_coords(frame, CYAN, y0=184)
    if xs.size == 0:
        return []
    px, py = PIVOT
    dist = np.hypot(xs - px, ys - py)
    threat = (dist > BARREL_RADIUS) & (ys < THREAT_Y_MAX)
    return list(zip(xs[threat].tolist(), ys[threat].tolist()))


def _cluster_1d(xs: np.ndarray, gap: float = WALKER_CLUSTER_GAP) -> np.ndarray:
    """Group sorted x-coords into clusters separated by more than `gap`
    pixels; returns each cluster's mean x. Loops over clusters (a handful),
    not pixels."""
    if xs.size == 0:
        return np.array([])
    xs = np.sort(xs)
    breaks = np.where(np.diff(xs) > gap)[0] + 1
    return np.array([g.mean() for g in np.split(xs, breaks)])


def count_walkers_per_side(frame: np.ndarray, pivot_x: float = PIVOT[0]):
    """(left_count, right_count) of landed/walking soldiers, split by side
    of the turret's x position."""
    xs, _ = _mask_coords(frame, WHITE, y0=WALKER_Y0, y1=WALKER_Y1 + 1)
    xs = xs[(xs < BASE_X0) | (xs > BASE_X1)]
    clusters = _cluster_1d(xs)
    left = int(np.sum(clusters < pivot_x))
    right = int(np.sum(clusters >= pivot_x))
    return left, right


if __name__ == "__main__":
    import mss

    with mss.mss() as sct:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        frame = np.array(sct.grab(monitor))[:, :, :3][:, :, ::-1]

    angle = detect_barrel_angle(frame)
    gap = nearest_threat_angle_gap(frame, angle)
    left, right = count_walkers_per_side(frame)
    print(f"barrel_angle={angle} nearest_threat_angle_gap={gap}")
    print(f"walkers left={left} right={right}")
