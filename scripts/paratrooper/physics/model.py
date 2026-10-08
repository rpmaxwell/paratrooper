"""Measured game physics, in GAME coordinates (the 640x400 game area; see
geometry.py). Ported from the validated scripts/bomb_model.py,
heli_model.py and trooper_model.py -- those stay the ground truth that
physics/tables.py is checked against. Every constant here was measured
from recordings or live calibration; the docstrings of the old modules
and plan_accuracy_and_priority.md hold the evidence.
"""
import os

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


# Bomb hitbox variant, per process (A/B, like HELI_BOX / TROOPER_BOX):
#   "model" -- the 8x8 sprite, swept bullet path (tables._bomb_table)
#   "code"  -- the game's own test, read from the disassembly (collision
#              routine, CS:11D4; tools/disasm.py): checked once per tick at
#              the bullet's drawn position, against the bomb where it was
#              BEFORE its move that tick; hit when the bullet's CGA column
#              (game x // 8) is within 1 of the bomb's and its native row is
#              5 above .. 8 below the bomb's top row: ~24 x 28 game px.
#              Explains 99.8% of 1.6k credited bomb kills (the 8x8 box: 22%).
BOMB_BOX = os.environ.get("PARATROOPER_BOMB_BOX", "model")
if BOMB_BOX not in ("model", "code"):
    raise ValueError(f"PARATROOPER_BOMB_BOX={BOMB_BOX!r}: expected 'model' or 'code'")


def bomb_hit_code(ux, uy, bx, by):
    """The game's bullet-vs-bomb test. Bullet drawn at (ux, uy), bomb
    top-left (bx, by), game coordinates."""
    return abs((ux >> 3) - (bx >> 3)) <= 1 and -8 <= ((by - 1) >> 1) - ((uy - 1) >> 1) <= 5


def bomb_at(x_release, direction, k):
    if not 0 <= k < len(BOMB_YS):
        return None
    return x_release + BOMB_VX * direction * k, BOMB_YS[k]


# ---- aircraft -----------------------------------------------------------------
HELI_VX = 8   # helicopters and planes: 8 px/tick, direction measured from motion

# Helicopter hitbox variant, per process so two bots can A/B it:
#   "model"  -- the sprite box below, swept bullet path (the validated default)
#   "fitted" -- the same box +8 px in x for both directions, bullet tested at
#               its drawn position only. Frame-level fit, 2026-10-06: 96.9% of
#               ~28k bullet outcomes exact vs 94.0% (drawn) / 88.8% (swept)
#               for "model"; see sim/collision.py (FITTED_BOX) and section 9
#               of notebooks/sim_spawn_drop.ipynb.
HELI_BOX = os.environ.get("PARATROOPER_HELI_BOX", "model")
if HELI_BOX not in ("model", "fitted"):
    raise ValueError(f"PARATROOPER_HELI_BOX={HELI_BOX!r}: expected 'model' or 'fitted'")
HELI_BOX_DX = 8 if HELI_BOX == "fitted" else 0
HELI_SWEPT = HELI_BOX == "model"


def heli_box(x0, y0, d):
    """Helicopter from its 32x4 skids' top-left; the tail side depends on
    direction. 48 px wide: re-measured 2026-09-27 from full sprites (the old
    heli_model box was 44 wide -- its measuring window clipped the tail).
    Shifted by HELI_BOX_DX under the "fitted" variant."""
    if d < 0:
        return x0 + HELI_BOX_DX, y0 - 16, x0 + 47 + HELI_BOX_DX, y0 + 3
    return x0 - 16 + HELI_BOX_DX, y0 - 16, x0 + 31 + HELI_BOX_DX, y0 + 3


def plane_box(x0, y0, d):
    """48x20 hollow outline, always at the top."""
    return x0, 1, x0 + 47, 20


# ---- troopers -------------------------------------------------------------------
FREE_VY, CANOPY_VY = 8, 4
GROUND_BODY_Y = 369
FREE_MAX_MEET_TICKS = 4     # free-fallers: point-blank only (canopy may open mid-flight)
CANOPY_OPEN_FALL_PX = 172   # median free fall before the canopy opens


# Trooper hitbox variant, per process (A/B, like HELI_BOX):
#   "model"  -- two solid, adjacent boxes (per-shot audit, 2026-09-27):
#               canopy = dome + the string area down to the head, body =
#               head + body (8 px wide)
#   "fitted" -- frame-level fit, 2026-10-07 (sim/collision.py, ~9.9k bullet
#               outcomes near troopers, 235 recorded games): NOT shifted
#               (+8 px like the helicopters made it worse, 82.0% vs 86.6%)
#               but bigger -- the body is as wide as the canopy (24 px) and
#               taller, the canopy reaches down to the body. 91.7% of
#               outcomes exact vs 86.4% for "model" on held-out games.
# Offsets (dx0, dy0, dx1, dy1) from the body's top-left (x, y), inclusive.
#   "fitted_parts" -- the same overall extent as "fitted", with the canopy /
#               body split taken from WHICH part died (sim/collision.py
#               kill_part_events, 1,536 kills next to canopy troopers): canopy
#               down to y-5 and body from y-4, as in "model", but both 24 px
#               wide; a free-faller's body ("body_free", arms up) reaches up to
#               y-12. Right part 94.9% vs 86.0% ("fitted") / 81.8% ("model");
#               absorption outcomes as "fitted" (91.7%). The narrow "model" body
#               is why crush shots kill the body: the planner thinks a bullet
#               can pass beside the body up to the canopy.
TROOPER_BOXES = {
    "model": {"canopy": (-8, -32, 15, -5), "body": (0, -4, 7, 11)},
    "fitted": {"canopy": (-8, -32, 15, 1), "body": (-8, -12, 15, 13)},
    "fitted_parts": {"canopy": (-8, -32, 15, -5), "body": (-8, -4, 15, 13), "body_free": (-8, -12, 15, 13)},
}
TROOPER_BOX = os.environ.get("PARATROOPER_TROOPER_BOX", "model")
if TROOPER_BOX not in TROOPER_BOXES:
    raise ValueError(f"PARATROOPER_TROOPER_BOX={TROOPER_BOX!r}: expected one of {sorted(TROOPER_BOXES)}")
TROOPER_BOX_OFFSETS = TROOPER_BOXES[TROOPER_BOX]


def trooper_offsets(part, free=False):
    """(dx0, dy0, dx1, dy1) of a part under the TROOPER_BOX variant; free=True
    = the body of a trooper still in free fall (own box if the variant has one)."""
    if part == "body" and free and "body_free" in TROOPER_BOX_OFFSETS:
        return TROOPER_BOX_OFFSETS["body_free"]
    return TROOPER_BOX_OFFSETS[part]


def trooper_box(x, y, part, free=False):
    """Hitbox of a trooper part ("canopy" | "body"), body top-left (x, y),
    under the TROOPER_BOX variant."""
    dx0, dy0, dx1, dy1 = trooper_offsets(part, free)
    return x + dx0, y + dy0, x + dx1, y + dy1


# ---- the simulator (ground truth for the tables) ------------------------------------
def _overlap(box, px, py):
    x0, y0, x1, y1 = box
    return px + 1 >= x0 and px <= x1 and py + 1 >= y0 and py <= y1


def simulate(lane, box_at, spawn_tick, max_k=10 ** 9, swept=True):
    """Tick at which a bullet on `lane` = (spawn, v), at its spawn point on
    `spawn_tick`, meets the target whose box at tick k is box_at(k) (None =
    target gone). Checks each tick plus (swept=True) the path between ticks
    -- the old simulators' semantics; troopers are hit only at drawn
    positions (swept=False), see tables.TROOPER_SUBSTEPS."""
    (sx, sy), (vx, vy) = lane
    for j in range(0, 40):
        k = spawn_tick + j
        if k > max_k:
            return None
        ux, uy = sx + vx * j, sy + vy * j
        if uy < -2 or not -2 <= ux <= 640:
            return None
        for s in ((1.0,) if j == 0 or not swept else (0.25, 0.5, 0.75, 1.0)):
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
