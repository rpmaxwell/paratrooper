#!/usr/bin/env python3
"""Deterministic aim-and-fire solver.

Paratrooper's turret rotation and paratrooper descent are both constant-
velocity (empirically confirmed, see calibrate_rotation.py and
calibrate_fall_speed.py/capture_fall_traces.py) -- this makes "will firing
now, or after rotating for T seconds, hit this target" a solvable
moving-target intercept problem, not something that needs to be learned by
trial and error.

Calibrated constants (Stage 1.1-1.3 of plan_accuracy_and_priority.md; see
the calibrate_*.py scripts for the raw trials):
  - Rotation speed: 128.23 deg/sec, std 1.07 across 18 clean full-range
    sweeps (2 detection-glitch outliers excluded) in both directions.
  - Rotation limits: 25.71 deg (full right) to 155.10 deg (full left),
    unchanged from earlier measurement -- the turret cannot reach fully
    horizontal in either direction.
  - Paratrooper fall speed is NOT a single constant -- it's two distinct
    phases, confirmed via piecewise-linear fits to 8 full release-to-
    landing traces (capture_fall_traces.py): a fast free-fall phase right
    after release at ~144.5 px/sec (std ~1.4 across 6 clean fits), then a
    much slower canopy-open descent at ~72.9 px/sec (std 0.36 across 17
    independent isolated-window fits -- the tightest measurement in this
    whole calibration pass), with the transition happening at a fairly
    consistent screen altitude around y=422 (std ~14, i.e. ~3% CV) rather
    than a fixed elapsed time since release. The old model used a single
    ~138 px/sec constant for the whole fall (apparently an average that
    happened to land near the free-fall number from a single early trace)
    -- since practice_deterministic_aim.py's target selection prefers the
    *freshest* (topmost, just-released) blob, most engagements were
    happening during or straddling the free-fall phase, where the old
    single-speed model was badly wrong for the back half of the flight.
    This was very likely the single largest source of the ~28% hit rate.
  - Helicopter horizontal speed: 144.78 px/sec (std 0.57, n=32), two fixed
    altitude bands at y=219.2 (moving left) and y=243.2 (moving right).
  - Bullet travel: confirmed hitscan, not merely assumed. Frame-diffed a
    30-frame rapid-capture burst (~3ms/frame) in a narrow angular wedge
    along the fire direction with a clear corridor (no other threats
    within 20deg) and found zero pixel change at any point after firing --
    no traveling projectile sprite is ever rendered. LEAD_TIME_S is
    dropped entirely (the earlier bundled fudge-factor grid search was
    inconclusive because there was nothing real for it to be compensating
    for beyond latency, which is now measured directly below).

LATENCY_OFFSET_S exists to absorb real round-trip latency (subprocess
spawn for xdotool, X server event delivery, frame-grab timing) between
"we decided to fire at time T" and "the fire key actually lands in-game."
Re-measured directly (calibrate_latency.py: extrapolate the post-action
angle-vs-time line back to the exact do_action() call instant and compare
to the true pre-action angle) rather than inferred from noisy practice-run
hit/miss outcomes: 52.7ms mean, std 18.3ms, n=31. Also confirmed reducing
actions.py's xdotool --delay from 50ms to 15ms does NOT reduce this
latency (52.5ms, n=25) -- the dominant cost is elsewhere (subprocess
spawn / X11 delivery), not the delay flag, so it's left at the proven-
reliable 50ms default.

THE GAME RUNS ON A DISCRETE TICK, NOT CONTINUOUS TIME -- confirmed, but
not yet successfully exploited. Rapid polling (70+Hz, well above the
game's own update rate) shows the barrel angle and the falling target's
y-position both hold *exactly* steady for a run of frames and then jump
all at once -- never smoothly interpolate. Measured tick period, from
independent barrel-rotation and target-fall traces: ~50ms (median),
consistent with a ~20Hz internal update loop (close to the classic
18.2Hz PC BIOS timer many 1982 DOS games synchronized to). This is a
real, solidly-evidenced structural fact about the emulated game.

A first attempt at exploiting it -- replacing the continuous root-find
below with a discrete search over tick indices, plus a
TICK_SAFETY_MARGIN_S fire-time buffer -- was tried and reverted: measured
against calibrate_isolated_miss.py's clean single-target trials, it made
the miss-angle std *worse* (6.67deg -> 12.76deg), not better. Most likely
the ~50ms tick-period estimate carries too much uncertainty (+-5-10ms
from our own polling resolution) to reliably snap to the *correct* tick,
and/or there's a phase offset between "we press rotate" and the game's
tick boundary that isn't just the fixed LATENCY_OFFSET_S constant --
forcing hold_s onto a rigid, possibly-mis-aligned grid was worse than
letting the continuous model float and average out. That attempt computed
a *uniform* grid from TICK_PERIOD_S; see below for why the real grid
isn't uniform and a second, more promising attempt using it directly.

THE TURRET CAN ONLY REST AT 19 FIXED ANGLES, NOT ANY CONTINUOUS VALUE
(2026-09-22, discover_firing_angles.py): sweeping the full range and
recording every genuinely distinct barrel-angle reading (collapsing
consecutive near-identical ones) gives the same 19 values, byte-for-byte,
across independent sweeps -- real, repeatable geometry (most likely
quantization of the rendered barrel-tip's integer pixel position at a
fixed radius from the pivot), not sampling noise. Spacing between
adjacent angles is NOT uniform -- 3.45 to 13.73 deg, mean 7.19 deg -- so
this is measured, not computed from ROTATION_SPEED_DEG_S/TICK_PERIOD_S.
This matters because the real hitbox was separately estimated at only
~1.7-3.4 deg wide (calibrate_isolated_miss.py) -- narrower than the
average gap between achievable angles. A target-free test
(test_gun_open_loop_accuracy.py: command the turret from angle A to a
chosen achievable angle B with no target on screen at all, using this
exact rotate/sleep/fire path) showed the execution itself lands on
exactly the intended discrete tick 73% of the time and within +-1 tick
97% of the time (std 3.11deg excluding two outlier trials, out of 100) --
an order of magnitude tighter than any target-based test in this file's
history. That means the old continuous solve below was never the
bottleneck it looked like: it solves for a target angle no real fire
could ever land on exactly, then whatever discrete tick the timing
happens to land nearest to is what actually fires -- an error up to
~half a tick (3.6 deg) baked in structurally, regardless of how tight
every other constant gets. solve_intercept below now searches the real
19-angle list directly instead of root-finding a continuous angle.
"""
import math
import time

import numpy as np

from actions import do_action
from threats import PIVOT, angle_to, detect_barrel_angle

ROTATION_SPEED_DEG_S = 128.23
ROTATE_RIGHT_LIMIT_DEG = 25.71  # minimum reachable angle
ROTATE_LEFT_LIMIT_DEG = 155.10  # maximum reachable angle
_ANGLE_TOLERANCE_DEG = 3.0  # small slop for measurement noise around the limits

# The turret's real, measured discrete resting angles (discover_firing_angles.py,
# 2026-09-22) -- see module docstring. Not uniform; do not replace with a
# computed arange over ROTATION_SPEED_DEG_S/TICK_PERIOD_S, that was tried
# and is a different (wrong) grid.
FIRING_ANGLES_DEG = [
    25.710, 34.216, 39.560, 50.440, 55.784, 64.290, 67.834, 76.430, 80.218,
    93.945, 101.689, 105.422, 113.962, 117.408, 125.754, 131.009, 141.633,
    146.821, 155.095,
]

FALL_SPEED_FREE_PX_S = 144.5  # pre-canopy free-fall
FALL_SPEED_CANOPY_PX_S = 72.9  # post-canopy-open descent
FALL_BREAK_Y = 422.0  # screen altitude where the canopy opens
GROUND_Y = 554.0  # beyond this the target has landed, no longer an airborne intercept target

LATENCY_OFFSET_S = 0.0527

# Separate from LATENCY_OFFSET_S (which absorbs rotation that happens
# "for free" inside the do_action() call itself): this is the Python-side
# processing time between grabbing the frame and issuing the *first*
# action for it -- target detection/clustering, barrel-angle detection,
# and solve_intercept's own root-finding all take real wall-clock time
# before any key is pressed, during which the target keeps falling and
# the model's "t=0" would otherwise silently be stale. Measured directly
# (time.time() before the frame grab vs. right before the first
# do_action() call, averaged over 30 real detect-and-solve cycles):
# 31.8ms mean, std 3.9ms.
DECISION_LATENCY_S = 0.0318

# Empirically measured discrete game-tick period (see module docstring) --
# kept for reference/future use, NOT currently applied: a discrete-tick
# reformulation of solve_intercept using this was tried and reverted
# (made the measured miss-angle std worse, not better).
TICK_PERIOD_S = 0.050
TICK_SAFETY_MARGIN_S = 0.0  # unused while the tick model is reverted


def valid_barrel_angle(angle) -> bool:
    """detect_barrel_angle occasionally picks up a stray threat pixel that
    happens to fall within BARREL_RADIUS of the pivot instead of the actual
    barrel, producing an angle outside the turret's physical range -- catch
    that here rather than acting on a bogus reading."""
    return angle is not None and (
        ROTATE_RIGHT_LIMIT_DEG - _ANGLE_TOLERANCE_DEG
        <= angle
        <= ROTATE_LEFT_LIMIT_DEG + _ANGLE_TOLERANCE_DEG
    )


def _fallen_y(y0: float, t: float) -> float:
    """Where a target starting at y0 will be after t more seconds of
    descent, accounting for the free-fall -> canopy-open speed change at
    FALL_BREAK_Y (see module docstring)."""
    if t <= 0:
        return y0
    if y0 >= FALL_BREAK_Y:
        return y0 + FALL_SPEED_CANOPY_PX_S * t
    time_to_break = (FALL_BREAK_Y - y0) / FALL_SPEED_FREE_PX_S
    if t <= time_to_break:
        return y0 + FALL_SPEED_FREE_PX_S * t
    return FALL_BREAK_Y + FALL_SPEED_CANOPY_PX_S * (t - time_to_break)


def _time_to_y(y0: float, y_target: float) -> float:
    """Inverse of _fallen_y: seconds for a target starting at y0 to reach
    y_target (y_target must be >= y0)."""
    if y0 >= FALL_BREAK_Y:
        return (y_target - y0) / FALL_SPEED_CANOPY_PX_S
    if y_target <= FALL_BREAK_Y:
        return (y_target - y0) / FALL_SPEED_FREE_PX_S
    t1 = (FALL_BREAK_Y - y0) / FALL_SPEED_FREE_PX_S
    t2 = (y_target - FALL_BREAK_Y) / FALL_SPEED_CANOPY_PX_S
    return t1 + t2


def _target_angle_at(tx: float, ty: float, t: float) -> float:
    return angle_to(PIVOT, (tx, _fallen_y(ty, t)))


def _clip(angle: float) -> float:
    return min(max(angle, ROTATE_RIGHT_LIMIT_DEG), ROTATE_LEFT_LIMIT_DEG)


def _max_reachable_t(target_x: float, target_y: float) -> float:
    """Seconds until the target either lands or its angle from the pivot
    exits the turret's reachable range, whichever comes first.

    The turret can't point near-horizontal in either direction (reachable
    range is [25.71, 155.10] degrees, not [0, 180]) -- a falling target
    positioned well off-center reaches that unreachable near-horizontal
    zone *before* it hits the ground, once it falls low enough relative to
    the pivot's own height. Capping only at GROUND_Y let the old solver
    keep searching for roots against a target angle that had already swept
    past the turret's physical limit, producing spurious "intercepts" at a
    barrel pinned at its limit against a target angle it could never
    really reach (large aim errors, tens of degrees, seen in practice).

    Worked in y-space (not angle-space) to sidestep atan2's wraparound at
    +-180 degrees: for a fixed horizontal offset dx from the pivot, the
    angle moves monotonically as y increases, so there's a single y beyond
    which the target has exited the reachable envelope on whichever side
    dx points -- solving for that y directly avoids ever comparing a
    wrapped angle value against the limits.
    """
    dx = target_x - PIVOT[0]
    if dx > 0:
        y_max = PIVOT[1] - dx * math.tan(math.radians(ROTATE_RIGHT_LIMIT_DEG))
    elif dx < 0:
        y_max = PIVOT[1] - dx * math.tan(math.radians(ROTATE_LEFT_LIMIT_DEG))
    else:
        y_max = GROUND_Y  # directly overhead, angle is always 90 -- reachable all the way down
    y_max = min(y_max, GROUND_Y)
    if y_max <= target_y:
        return 0.0  # already outside the reachable envelope
    return _time_to_y(target_y, y_max)


def solve_intercept(barrel_angle: float, target_x: float, target_y: float):
    """Returns (direction, T): direction is +1 (rotate left / increasing
    angle), -1 (rotate right / decreasing angle), or 0 (already aligned);
    T is seconds to hold that rotation before firing. Returns None if no
    achievable angle intercepts the target before it reaches the ground
    or leaves the turret's reachable range.

    target_x/target_y should be the position read directly off the
    captured frame -- DECISION_LATENCY_S is applied internally to advance
    the target to where it'll really be once we start acting on this
    solution, so callers don't need to account for it themselves.

    Rather than solving for the continuous angle that would exactly
    intercept the target (which the turret could never actually stop at
    -- see module docstring), this evaluates every one of the turret's 19
    real achievable angles (FIRING_ANGLES_DEG), computes how long it'd
    take to rotate there from barrel_angle and where the target would
    really be at that moment, and picks whichever achievable angle
    minimizes that miss. Only 19 candidates, so a direct scan is simpler
    and cheaper than root-finding ever was.
    """
    target_y = _fallen_y(target_y, DECISION_LATENCY_S)
    max_t = _max_reachable_t(target_x, target_y)
    if max_t <= 0:
        return None

    best = None  # (angle, hold_t, |miss|)
    for angle in FIRING_ANGLES_DEG:
        d = angle - barrel_angle
        hold_t = abs(d) / ROTATION_SPEED_DEG_S
        if hold_t > max_t:
            continue
        miss = abs(angle - _target_angle_at(target_x, target_y, hold_t))
        if best is None or miss < best[2]:
            best = (angle, hold_t, miss)

    if best is None:
        return None
    angle, hold_t, _ = best
    d = angle - barrel_angle
    direction = 0 if abs(d) < 1e-6 else (1 if d > 0 else -1)
    return (direction, hold_t)


# Bomb: a small WHITE circular blob (bounding box ~7x7px, ~48 filled
# pixels -- see hunt_bomb.py, 2026-09-22). Live sessions kept missing at
# almost exactly the same (position, "velocity") every time despite very
# different individual bombs (2026-09-23/24) -- the tell that it wasn't
# noise was that the original constant-velocity model was simply wrong,
# not that our estimate of it was noisy. diagnose_bomb_trajectory.py
# (6 tracks, 88 points, live) confirmed the real motion: horizontal speed
# is genuinely constant (132.8 px/s, std 5.3 -- tight), but vertical speed
# ACCELERATES continuously (least-squares fit: 142.6 px/s^2) rather than
# holding constant -- classic gravity-drop kinematics, not the ballistic
# constant-velocity model solve_bomb_intercept used to assume. Even more
# telling: the first trackable position was *exactly* y=276.0 every single
# time (std=0.0 across 6 independent bombs) -- this isn't "bombs behave
# similarly", it's the same scripted tick sequence every time, matching
# the discrete-tick reality already established elsewhere in this file.
# A naive 2-frame instantaneous-velocity estimate, projected forward
# assuming constant speed, systematically undershoots how far an
# accelerating target will have fallen by fire time -- worse the further
# ahead it has to predict -- which is exactly the bias that was making
# shots consistently land short.
#
# Calibrated directly (no live per-shot estimation needed at all now):
BOMB_VX_PX_S = 132.8  # magnitude; direction (sign) still read live, it's the one thing that varies
BOMB_Y_REF = 276.0
BOMB_VY_AT_REF_PX_S = 125.5
BOMB_ACCEL_PX_S2 = 142.6
BOMB_MAX_T_S = 3.0  # generous cap -- _bomb_max_reachable_t below stops earlier in practice


def _bomb_vy_at_y(y: float) -> float:
    """SUVAT: vy^2 = vy_ref^2 + 2*a*(y-y_ref) -- current vertical speed at
    a given height, given the calibrated constant acceleration. Avoids
    needing to know elapsed time since some absolute release moment;
    works from whatever height a bomb is first observed at."""
    v2 = BOMB_VY_AT_REF_PX_S ** 2 + 2 * BOMB_ACCEL_PX_S2 * (y - BOMB_Y_REF)
    return math.sqrt(max(v2, 0.0))


def _bomb_pos_at(bx: float, by: float, vx_sign: float, t: float):
    """Position t seconds after being observed at (bx, by), using the
    calibrated constant-horizontal-speed / constant-vertical-acceleration
    model. vx_sign is +1 or -1 (direction only -- magnitude is the
    calibrated BOMB_VX_PX_S, not measured live)."""
    vy0 = _bomb_vy_at_y(by)
    y = by + vy0 * t + 0.5 * BOMB_ACCEL_PX_S2 * t * t
    x = bx + vx_sign * BOMB_VX_PX_S * t
    return x, y


def _bomb_max_reachable_t(bx: float, by: float, vx_sign: float, dt: float = 0.02) -> float:
    """Seconds until the bomb passes the ground line or its angle from
    the pivot exits the turret's reachable range, whichever comes first.
    Unlike _max_reachable_t (paratrooper-only: fixed x, closed-form in
    y-space), the bomb moves in both x and y, so this just steps forward
    numerically -- cheap, since it only needs to bound the discrete-angle
    scan below, not resolve a root precisely."""
    t = 0.0
    while t <= BOMB_MAX_T_S:
        x, y = _bomb_pos_at(bx, by, vx_sign, t)
        if y >= GROUND_Y:
            return t
        angle = angle_to(PIVOT, (x, y))
        if not (ROTATE_RIGHT_LIMIT_DEG <= angle <= ROTATE_LEFT_LIMIT_DEG):
            return t
        t += dt
    return BOMB_MAX_T_S


def solve_bomb_intercept(barrel_angle: float, bx: float, by: float, vx_sign: float):
    """Same discrete-grid approach as solve_intercept (see its docstring),
    generalized to a target moving under the calibrated constant-vx /
    constant-vertical-acceleration bomb model instead of falling straight
    down from a fixed x. vx_sign is +1 (moving right) or -1 (moving left)
    -- the one thing about a given bomb that isn't fixed by calibration.
    Returns (direction, hold_t) or None, same convention as
    solve_intercept."""
    bx, by = _bomb_pos_at(bx, by, vx_sign, DECISION_LATENCY_S)
    max_t = _bomb_max_reachable_t(bx, by, vx_sign)
    if max_t <= 0:
        return None

    best = None  # (angle, hold_t, |miss|)
    for angle in FIRING_ANGLES_DEG:
        d = angle - barrel_angle
        hold_t = abs(d) / ROTATION_SPEED_DEG_S
        if hold_t > max_t:
            continue
        tx, ty = _bomb_pos_at(bx, by, vx_sign, hold_t)
        miss = abs(angle - angle_to(PIVOT, (tx, ty)))
        if best is None or miss < best[2]:
            best = (angle, hold_t, miss)

    if best is None:
        return None
    angle, hold_t, _ = best
    d = angle - barrel_angle
    direction = 0 if abs(d) < 1e-6 else (1 if d > 0 else -1)
    return (direction, hold_t)


def execute_intercept(direction: int, hold_s: float):
    """Rotates (if needed) then fires at the computed intercept time."""
    if direction != 0:
        do_action(1 if direction > 0 else 2)
    sleep_s = max(0.0, hold_s - LATENCY_OFFSET_S)
    if sleep_s > 0:
        time.sleep(sleep_s)
    do_action(3)
