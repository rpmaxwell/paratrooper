#!/usr/bin/env python3
"""Paratrooper tracking + intercept planning, same tick-simulation approach
as bomb_model.py / heli_model.py (game-area coordinates, palette-indexed
frames). Measured from 10 recorded episodes (2026-09-26):

  * Body = cyan 8x12 (56 px) with a white 4x4 head directly above it.
  * Canopy = cyan 24x12 (232 px) dome at exactly (body.x - 8, body.y - 32),
    magenta strings between. Canopy detection flickers frame to frame, so
    state needs hysteresis.
  * Troopers never drift: x is constant for life -> x is the identity key.
  * Fall per tick: 8 px free-falling, 4 px under canopy. A trooper whose
    canopy is shot falls at 8 px/tick again and dies on landing (taking a
    landed trooper under it with it) -- "doomed", never worth a bullet.
  * Canopies open at any altitude (y 73..349 seen), so a free-faller's
    future is only predictable over a short horizon.
  * Landed trooper = white 4x4 head at ground level (y0 365; 349 when
    standing on another trooper's head).
"""
import numpy as np

from bomb_model import BARREL, FIRE_LATENCY_TICKS, LIMIT_SIGHT_LANE, TICK_S, components, lane

FREE_VY, CANOPY_VY = 8, 4
GROUND_BODY_Y = 369  # body top of a trooper standing on the ground
LANDED_HEAD_Y_MIN = 330
TURRET_X0, TURRET_X1 = 280, 360  # turret/barrel region: its cyan isn't troopers
# Canopy may open any time and halve the fall speed; the shot audit
# (audit_trooper_shots.py, 2026-09-27) found only 3/15 free-faller body
# shots hit, 5 of the misses because the canopy opened before the bullet
# arrived. So: point-blank only, otherwise wait for the canopy.
FREE_MAX_MEET_TICKS = 4


def detect(idx):
    """-> bodies [(x, y)], canopies [(body_x, body_y_implied)], landed_x [x]"""
    bodies, canopies, landed = [], [], []
    for xs, ys in components(idx[40:372] == 1):
        w, h, n = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, xs.size
        x0, y0 = int(xs.min()), int(ys.min()) + 40
        if (w, h, n) == (8, 12, 56):
            bodies.append((x0, y0))
        elif (w, h, n) == (24, 12, 232):
            canopies.append((x0 + 8, y0 + 32))
    for xs, ys in components(idx[LANDED_HEAD_Y_MIN:372] == 3):
        if xs.size == 16 and xs.max() - xs.min() == 3 and ys.max() - ys.min() == 3:
            x = int(xs.min()) - 2
            if not TURRET_X0 <= x <= TURRET_X1:
                landed.append(x)
    return bodies, canopies, landed


class Trooper:
    def __init__(self, x, y, t):
        self.x, self.y, self.t = x, y, t
        self.first_t = t
        self.canopy_hits = 0      # frames with a canopy seen over us
        self.canopy_last_t = None
        self.state = "free"       # free | canopy | doomed | dead
        self.pending = None       # engagement awaiting its meeting tick
        self.hist = [(t, y)]

    @property
    def vy(self):
        return CANOPY_VY if self.state == "canopy" else FREE_VY

    def y_at(self, t):
        return self.y + self.vy * (t - self.t) / TICK_S


class TrooperTracker:
    def __init__(self, log=print, on_miss=None):
        self.on_miss = on_miss
        self.troopers = []
        self.landed = []
        self.log = log
        self.stats = dict(engaged=0, shots=0, body_kills=0, chute_kills=0, misses=0,
                          doomed_seen=0, bonus_candidates=0)

    def update(self, idx, t):
        bodies, canopies, self.landed = detect(idx)
        seen = set()
        for x, y in bodies:
            if TURRET_X0 <= x <= TURRET_X1 and y > 250:
                continue
            tr = next((tr for tr in self.troopers if tr.x == x and tr.state != "dead"
                       and tr.y - 2 <= y <= tr.y + FREE_VY * max(1, (t - tr.t) / TICK_S) + 8), None)
            if tr is None:
                tr = Trooper(x, y, t)
                self.troopers.append(tr)
            elif y != tr.y:
                tr.hist.append((t, y))
                tr.y, tr.t = y, t
            else:
                tr.t = t
            seen.add(id(tr))
        for bx, by in canopies:
            for tr in self.troopers:
                if tr.x == bx and abs(tr.y - by) <= 12 and tr.state in ("free", "canopy"):
                    tr.canopy_hits += 1
                    tr.canopy_last_t = t
                    if tr.state == "free" and tr.canopy_hits >= 2:
                        tr.state = "canopy"
        for tr in self.troopers:
            if tr.state == "canopy" and tr.canopy_last_t is not None and t - tr.canopy_last_t > 0.15:
                # canopy gone: confirmed by the body falling at free-fall speed
                recent = [h for h in tr.hist if h[0] >= tr.canopy_last_t - 0.02]
                if len(recent) >= 2 and (recent[-1][1] - recent[0][1]) / max(1e-3, (recent[-1][0] - recent[0][0]) / TICK_S) > 6:
                    tr.state = "doomed"
                    self.stats["doomed_seen"] += 1
                    over = any(abs(lx - tr.x) <= 6 for lx in self.landed)
                    self.log(f"trooper x={tr.x} CHUTE LOST at y={tr.y} -> doomed"
                             f"{' (lands on a trooper!)' if over else ''}")
                    if tr.pending:
                        self.stats["chute_kills"] += 1
                        tr.pending = None
        # vanished troopers
        for tr in list(self.troopers):
            gone = t - tr.t
            if tr.y >= GROUND_BODY_Y - 20 and gone > 0.3:
                self.troopers.remove(tr)  # landed (or died on landing)
            elif gone > 0.3:
                if tr.pending and t >= tr.pending["t_meet"]:
                    self.stats["body_kills"] += 1
                    self.log(f"trooper x={tr.x} KILLED (body) at y~{tr.y}")
                self.troopers.remove(tr)
        # resolve engagements whose meeting tick passed with the target intact
        for tr in self.troopers:
            p = tr.pending
            # a trooper still alive and not doomed after the meeting tick was
            # missed -- including one whose canopy opened meanwhile (that
            # state change used to leave it "pending" and untargetable)
            if p and t > p["t_meet"] + 3 * TICK_S and id(tr) in seen and tr.state in ("free", "canopy"):
                self.stats["misses"] += 1
                self.log(f"trooper x={tr.x} MISS ({p['part']}, pos {p['pos']}, shots {p['shots']})")
                tr.pending = None
                if self.on_miss:
                    self.on_miss()

    def candidates(self):
        return [tr for tr in self.troopers if tr.state in ("free", "canopy") and tr.pending is None]


# Exact sprite pixels (from recordings) -- used by audit_trooper_shots.py to
# describe where bullets went. NOT the game's collision: the audit showed
# collisions are rectangular (see _box), and _mask_hit only applies to
# 5-tuple boxes, which _box no longer returns.
_CANOPY_ROWS = ["........########........"] * 2 + ["....################...."] * 2 + \
               ["..####################.."] * 2 + ["#" * 24] * 6
_BODY_ROWS = ["..####.."] * 4 + ["########"] * 2 + ["..####.."] * 4 + ["##....##"] * 6
MASKS = {"canopy": np.array([[c == "#" for c in r] for r in _CANOPY_ROWS]),
         "body": np.array([[c == "#" for c in r] for r in _BODY_ROWS])}


def _mask_hit(box, px, py):
    """Does a 2x2 bullet at (px, py) touch an actual sprite pixel?"""
    x0, y0, x1, y1, part = box
    m = MASKS[part]
    px, py = int(round(px)), int(round(py))
    for by in (py, py + 1):
        for bx in (px, px + 1):
            r, c = by - int(round(y0)), bx - int(round(x0))
            if 0 <= r < m.shape[0] and 0 <= c < m.shape[1] and m[r, c]:
                return True
    return False


def _box(tr, part, k, t_obs):
    """Target box (x0, y0, x1, y1) k ticks after t_obs."""
    y = tr.y_at(t_obs) + tr.vy * k
    # Two solid, adjacent hitboxes (shot audit, 2026-09-27): the canopy box
    # covers the dome AND the whole string area down to the head -- 12 of
    # 14 bullets that reached the black gap between the strings still
    # killed the canopy; the trooper box (head + body) sits directly below.
    if part == "canopy":
        return tr.x - 8, y - 32, tr.x + 15, y - 5
    return tr.x, y - 4, tr.x + 7, y + 11


def bullet_hits_box(box_at, pos, spawn_tick, max_k=60, lane_=None):
    (sx, sy), (vx, vy) = lane_ or (BARREL[pos][2], BARREL[pos][3])
    for j in range(0, 40):
        k = spawn_tick + j
        if k > max_k:
            return None
        ux, uy = sx + vx * j, sy + vy * j
        if uy < -2 or not -2 <= ux <= 640:
            return None
        for s in ((1.0,) if j == 0 else (0.25, 0.5, 0.75, 1.0)):
            b0 = box_at(k - 1 + s)
            if b0 is None:
                return None
            x0, y0, x1, y1 = b0[:4]
            px, py = ux - vx * (1 - s), uy - vy * (1 - s)
            if px + 1 >= x0 and px <= x1 and py + 1 >= y0 and py <= y1 and \
                    (len(b0) < 5 or _mask_hit(b0, px, py)):
                return k
    return None


CLAMP_TICKS = 5  # keep in sync with turret.CLAMP_TICKS (no mss import here)


def plan_trooper(tr, part, cur_pos, t_obs, ticks_per_pos=1.1, settle_ticks=2):
    """Best barrel position for this trooper part. Returns
    (pos, spawn_ticks, meet_tick, clamp) or None. Earliest meet wins; ties
    go to the wider window. At the two rotation limits both lanes are
    options: stop on sight (steeper lane, no delay) or clamp into the stop
    (flatter lane that reaches lower side targets, CLAMP_TICKS slower)."""
    ticks_to_ground = (GROUND_BODY_Y - tr.y_at(t_obs)) / tr.vy
    max_k = ticks_to_ground - 1
    if part == "body" and tr.state == "free":
        max_k = min(max_k, FREE_MAX_MEET_TICKS)

    def box_at(k):
        return _box(tr, part, k, t_obs) if k <= max_k else None

    options = [(pos, False) for pos in range(len(BARREL))] + [(p, True) for p in LIMIT_SIGHT_LANE]
    best = None
    for pos, clamp in options:
        moves = 0 if cur_pos is None else abs(pos - cur_pos)
        earliest = (int(np.ceil(moves * ticks_per_pos)) + (settle_ticks if moves else 0)
                    + FIRE_LATENCY_TICKS + (CLAMP_TICKS if clamp else 0))
        lane_ = (BARREL[pos][2], BARREL[pos][3]) if clamp else lane(pos, cur_pos)
        run = []
        for f in range(earliest, earliest + 25):
            m = bullet_hits_box(box_at, pos, f, max_k, lane_)
            if m is not None:
                run.append((f, m))
            elif run:
                break
        if not run:
            continue
        key = (run[len(run) // 2][1], -len(run), moves)
        if best is None or key < best[0]:
            best = (key, pos, [f for f, _ in run], run[len(run) // 2][1], clamp)
    return None if best is None else best[1:]


# Free fall before the canopy opens, measured from first sighting near the
# helicopters (10 recordings, n=89): 10/25/50/75/90th pct = 100/132/172/
# 196/204 px. Only 1 of 90 tracks reached the ground without a canopy.
CANOPY_OPEN_FALL_PX = 172


def predicted_canopy_trooper(tr, t):
    """Hypothetical canopied version of a free-faller, at its most likely
    canopy-opening point, for pre-aiming."""
    y_open = max(tr.y_at(t), tr.hist[0][1] + CANOPY_OPEN_FALL_PX)
    ghost = Trooper(tr.x, int(y_open), t)
    ghost.state = "canopy"
    return ghost


def urgency(tr, landed, t):
    """Lower = shoot first: ticks until it lands, heavily discounted when a
    chute shot would drop it onto an already-landed trooper (double kill),
    and when its side is already close to the 4-landed doom threshold."""
    ticks = (GROUND_BODY_Y - tr.y_at(t)) / tr.vy
    over_landed = any(abs(lx - tr.x) <= 6 for lx in landed)
    side = [lx for lx in landed if (lx < 320) == (tr.x < 320)]
    score = ticks - 6 * len(side)
    if over_landed and tr.state == "canopy":
        score -= 100
    return score, over_landed
