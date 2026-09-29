"""Measured game physics, in GAME coordinates (the 640x400 game area; see
geometry.py). Ported from the validated scripts/bomb_model.py,
heli_model.py and trooper_model.py -- those stay the ground truth that
physics/tables.py is checked against. Every constant here was measured
from recordings or live calibration; the docstrings of the old modules
and plan_accuracy_and_priority.md hold the evidence.
"""
TICK_S = 1 / 18.2065

# ---- turret ---------------------------------------------------------------
# (angle, spawn, v) per barrel position 0 (25.7 deg, full right) .. 18 (155.1,
# full left). Right half measured (calibrate_bullets.py), left half mirrored
# (tip/spawn x -> 640-x, vx -> -vx), verified where measured.
_RIGHT = [
    (25.7, (360, 307), (20, -4)), (34.2, (356, 299), (18, -8)), (39.6, (352, 295), (16, -10)),
    (50.4, (352, 291), (16, -12)), (55.8, (344, 291), (12, -12)), (64.3, (340, 287), (10, -14)),
    (67.8, (336, 287), (8, -14)), (76.4, (332, 283), (6, -16)), (80.2, (324, 283), (2, -16)),
]
_CENTER = (93.9, (320, 283), (0, -16))
_LEFT_ANGLES = [155.1, 146.8, 141.6, 131.0, 125.8, 117.4, 114.0, 105.4, 101.7]
BARREL = sorted(_RIGHT + [_CENTER] + [(la, (640 - sp[0], sp[1]), (-v[0], v[1]))
                                      for la, (_, sp, v) in zip(_LEFT_ANGLES, _RIGHT)])
N_POS = len(BARREL)

# Each rotation limit is two lanes that look identical on screen
# (calibrate_limit_lanes.py): stopped on sight -> steeper lane (20/21 shots);
# held into the stop ("clamped") -> the table lane (23/23 shots).
LIMIT_SIGHT_LANE = {0: ((360, 303), (20, -6)), 18: ((280, 303), (-20, -6))}

# Lane ids: 0..18 = BARREL[pos] lanes; 19 / 20 = the sight lanes of pos 0 / 18.
LANES = [(sp, v) for _, sp, v in BARREL] + [LIMIT_SIGHT_LANE[0], LIMIT_SIGHT_LANE[18]]
N_LANES = len(LANES)
SIGHT_LANE_ID = {0: 19, 18: 20}


def lane_id(pos, cur_pos, clamped=False):
    """Lane a bullet from `pos` will use. Arriving at a limit (or sitting
    there unclamped) -> the steeper sight lane."""
    if pos in SIGHT_LANE_ID and (cur_pos != pos or not clamped):
        return SIGHT_LANE_ID[pos]
    return pos


# Key-press -> bullet at its spawn point, in ticks from the frame we reacted
# to (median first-visible latency 0.183 s with xdotool; re-measure for XTEST).
FIRE_LATENCY_TICKS = 3
TICKS_PER_POS = 1.1   # the turret steps ~one position per tick
SETTLE_TICKS = 2      # extra margin after a move
CLAMP_TICKS = 5       # cost of clamping into a rotation stop (0.2 s hold + latency)

# ---- bombs ------------------------------------------------------------------
# 8x8 white disc; x moves 8 px/tick (the plane's speed), y follows this fixed
# discrete-gravity sequence; lands on the turret ~33 ticks after release.
BOMB_YS = [17, 19, 21, 23, 25, 29, 33, 37, 41, 47, 53, 59, 65, 73, 81, 89, 97, 107, 117,
           127, 137, 149, 161, 173, 185, 199, 213, 227, 241, 257, 273, 289, 305, 323]
BOMB_TICK_OF_Y = {y: k for k, y in enumerate(BOMB_YS)}
BOMB_VX = 8
BOMB_LANDED_K = 31


def bomb_at(x_release, direction, k):
    if not 0 <= k < len(BOMB_YS):
        return None
    return x_release + BOMB_VX * direction * k, BOMB_YS[k]


# ---- aircraft -----------------------------------------------------------------
HELI_VX = 8   # helicopters and planes: 8 px/tick, direction measured from motion


def heli_box(x0, y0, d):
    """Helicopter from its 32x4 skids' top-left; the tail side depends on
    direction. 48 px wide: re-measured 2026-09-27 from full sprites (the old
    heli_model box was 44 wide -- its measuring window clipped the tail)."""
    if d < 0:
        return x0, y0 - 16, x0 + 47, y0 + 3
    return x0 - 16, y0 - 16, x0 + 31, y0 + 3


def plane_box(x0, y0, d):
    """48x20 hollow outline, always at the top."""
    return x0, 1, x0 + 47, 20


# ---- troopers -------------------------------------------------------------------
FREE_VY, CANOPY_VY = 8, 4
GROUND_BODY_Y = 369
FREE_MAX_MEET_TICKS = 4     # free-fallers: point-blank only (canopy may open mid-flight)
CANOPY_OPEN_FALL_PX = 172   # median free fall before the canopy opens


def trooper_box(x, y, part):
    """Two solid, adjacent hitboxes (per-shot audit, 2026-09-27): canopy =
    dome + the whole string area down to the head; trooper = head + body."""
    if part == "canopy":
        return x - 8, y - 32, x + 15, y - 5
    return x, y - 4, x + 7, y + 11


# ---- the simulator (ground truth for the tables) ------------------------------------
def _overlap(box, px, py):
    x0, y0, x1, y1 = box
    return px + 1 >= x0 and px <= x1 and py + 1 >= y0 and py <= y1


def simulate(lane, box_at, spawn_tick, max_k=10 ** 9):
    """Tick at which a bullet on `lane` = (spawn, v), at its spawn point on
    `spawn_tick`, meets the target whose box at tick k is box_at(k) (None =
    target gone). Checks each tick plus the swept path between ticks.
    Identical semantics to the validated old simulators."""
    (sx, sy), (vx, vy) = lane
    for j in range(0, 40):
        k = spawn_tick + j
        if k > max_k:
            return None
        ux, uy = sx + vx * j, sy + vy * j
        if uy < -2 or not -2 <= ux <= 640:
            return None
        for s in ((1.0,) if j == 0 else (0.25, 0.5, 0.75, 1.0)):
            b = box_at(k - 1 + s)
            if b is None:
                return None
            if _overlap(b, ux - vx * (1 - s), uy - vy * (1 - s)):
                return k
    return None


# ---- targeting area -------------------------------------------------------------
def _reach_floor(x):
    """Largest body-top y (lowest point on screen) at which any lane's bullet
    path crosses a trooper in column x -- below it no shot exists at all."""
    best = -10 ** 9
    for (sx, sy), (vx, vy) in LANES:
        for j10 in range(0, 400):
            j = j10 / 10
            px, py = sx + vx * j, sy + vy * j
            if py < -2 or not -2 <= px <= 640:
                break
            if x - 8 <= px <= x + 15:
                best = max(best, py + 4)
    return best


# lowest hittable body top per column (even x): ~258 at the screen edges,
# ~303-311 next to the turret. Troopers below this can never be hit.
REACH_FLOOR = {x: _reach_floor(x) for x in range(0, 640, 2)}
