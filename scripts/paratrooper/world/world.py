"""World model + verified telemetry.

update(t, frame) turns one native frame into tracked entities -- bombs,
helicopters, planes, troopers (with lifecycle), landed troopers, OUR
bullets, the turret, score, phase -- all in game coordinates and game
ticks, and emits telemetry records as things resolve.

Telemetry principle: an outcome is logged only when the frames show it.
Kills are attributed to a bullet only when that bullet vanished where its
next position overlapped the entity (canopy box, body box, aircraft,
bomb) within a few ticks of the entity disappearing / losing its canopy;
otherwise the fate is logged with evidence "unexplained".
"""
from dataclasses import dataclass, field
from itertools import count

from ..perception.barrel import barrel_pos
from ..perception.hud import is_done, read_score
from ..perception.sprites import detect
from ..physics import model as M
from .clock import GameClock

TRACK_KEEP_S = 0.6        # bombs: keep a track through short occlusions
TROOPER_GONE_S = 0.3
AIR_GONE_S = 0.3
BULLET_GONE_TICKS = 1.6   # not seen at its next position this long -> ended
EVIDENCE_TICKS = 4        # bullet end <-> entity change must be this close
STILL_TICKS = 5           # a trooper body that hasn't moved this long has landed
LANDED_STILL_TICKS = 6    # a head counts as landed once it hasn't moved for this long
_ids = count(1)


@dataclass
class Bomb:
    id: int
    dir: int
    x_release: int
    k: int
    first_k: int
    t: float
    first_tick: int
    plan: tuple = None
    fired: list = field(default_factory=list)

    def k_est(self, t):
        return self.k + int(round((t - self.t) / M.TICK_S))


@dataclass
class Trooper:
    id: int
    x: int
    y: int
    t: float
    first_tick: int
    first_y: int
    state: str = "free"            # free | canopy | doomed
    canopy_hits: int = 0
    canopy_last_t: float = None
    canopy_open: tuple = None      # (tick, y)
    canopy_lost: tuple = None      # (tick, y)
    y_by_tick: dict = field(default_factory=dict)
    engagements: list = field(default_factory=list)  # shot ids aimed at it
    pending: dict = None           # engagement awaiting its meeting tick
    feasible_ever: bool = False    # some plan existed at some point (set by the policy)
    busy_skips: int = 0            # times it was feasible but the turret was busy
    hist: list = field(default_factory=list)

    @property
    def vy(self):
        return M.CANOPY_VY if self.state == "canopy" else M.FREE_VY


@dataclass
class Aircraft:
    id: int
    kind: str      # heli | plane
    y0: int        # skid y (heli) / 1 (plane)
    x: int
    d: int         # -1 left, +1 right, 0 unknown yet
    t: float
    first_tick: int
    pending: list = field(default_factory=list)  # shot ids aimed at it
    busy_until: float = 0.0  # our bullets are en route until then


@dataclass
class Bullet:
    id: int
    press_t: float
    lane: int
    ctx: dict
    pos: tuple = None
    t: float = None
    first_tick: int = None
    path: list = field(default_factory=list)   # (tick, x, y)
    v_measured: tuple = None
    end: dict = None


class World:
    def __init__(self, t0, emit):
        self.emit = emit
        self.clock = GameClock(t0)
        self.t0 = t0
        self.phase = 1
        self.last_heli_t = t0
        self.bombs, self.troopers, self.aircraft = [], [], []
        self.bullets, self.expected, self.orphans = [], [], []
        self.recent_ends = []          # ended bullets, for kill attribution
        self.landed = []
        self.landed_hist = []          # (tick, landed heads) -- to confirm crushes after the fact
        self._head_since = {}          # head position -> first tick seen there
        self.standing = []             # (x, body y) of troopers that stopped on a stack
        self.lane_dir = {}
        self.entry_side = None
        self.barrel = None
        self.clamped = False           # set by control
        self.score = 0
        self.done = False
        self._prev_moving = None
        self.stats = dict(shots=0, bullets_found=0, bullets_not_found=0, bullets_unattributed=0,
                          hits=0, left_screen=0,
                          unexplained_ends=0, trooper_body_killed=0, trooper_chute_killed=0,
                          trooper_landed=0, trooper_fell_dead=0, crushed=0, aircraft_down=0,
                          bombs_shot=0, bombs_landed=0, unexplained_fates=0)

    # ------------------------------------------------------------------ utils
    def tick(self, t):
        return self.clock.tick(t)

    def note_press(self, t, ctx, pos=None):
        """Control pressed Up (every press fires a bullet) with the barrel at
        pos (default: the last position read)."""
        pos = self.barrel if pos is None else pos
        lane = M.lane_id(pos, pos, self.clamped) if pos is not None else None
        b = Bullet(next(_ids), t, lane, ctx)
        self.expected.append(b)
        self.stats["shots"] += 1
        self.emit(dict(type="press", tick=self.tick(t), phase=round(((t - self.clock.t0) % M.TICK_S) / M.TICK_S, 3),
                       shot=b.id, lane=lane, barrel=pos,
                       clamped=self.clamped, ctx=ctx))
        return b.id

    # ----------------------------------------------------------------- update
    def update(self, t, frame):
        s = detect(frame)
        self.barrel = barrel_pos(frame)
        self.done = is_done(frame)
        tk = self.tick(t)
        sc = read_score(frame)
        if sc != self.score and not self.done:
            self.emit(dict(type="score", tick=tk, old=self.score, new=sc, delta=sc - self.score))
            self.score = sc
        moving = (tuple(s.helis), tuple(s.planes_full), tuple(s.bombs), tuple(s.dots))
        if self._prev_moving is not None and moving != self._prev_moving:
            self.clock.observe_move(t)
        self._prev_moving = moving
        self._phase(s, t, tk)
        self._aircraft(s, t, tk)
        self._bombs(s, t, tk)
        self._bullets(s, t, tk)
        self._troopers(s, t, tk)
        # landed = a head that has stayed put: a falling trooper's own head
        # passes through the same band on its way down (and dies on impact),
        # which otherwise looks like a landed trooper that then "vanishes"
        for h in s.landed:
            self._head_since.setdefault(h, tk)
        self._head_since = {h: t0 for h, t0 in self._head_since.items() if h in set(s.landed)}
        self.landed = [h for h, t0 in self._head_since.items() if tk - t0 >= LANDED_STILL_TICKS]
        self.landed_hist.append((tk, list(self.landed)))
        if len(self.landed_hist) > 120:
            self.landed_hist = self.landed_hist[-80:]
        return s

    # ------------------------------------------------------------------ phase
    def _phase(self, s, t, tk):
        prev = self.phase
        if s.heli_any:
            self.last_heli_t = t
            if self.phase != 1:
                self.phase = 1
        if s.planes and self.phase != 2:
            self.phase = 2
        elif self.phase == 1 and not s.heli_any and t - self.last_heli_t > 0.5:
            self.phase = "gap"
        if self.phase != prev:
            self.emit(dict(type="phase", tick=tk, old=prev, new=self.phase))

    # --------------------------------------------------------------- aircraft
    def _aircraft(self, s, t, tk):
        seen = set()
        obs = [("heli", y0, x) for x, y0 in s.helis] + [("plane", 1, x) for x in s.planes_full]
        for kind, y0, x in obs:
            a = None
            for c in self.aircraft:
                if c.kind != kind or c.y0 != y0 or id(c) in seen:
                    continue
                dt = (t - c.t) / M.TICK_S
                if c.d and abs(x - (c.x + M.HELI_VX * c.d * dt)) <= 16 + 4 * dt:
                    a = c
                    break
                if not c.d and 0 <= abs(x - c.x) <= 16 + 8 * dt:
                    a = c
                    break
            if a is None:
                a = Aircraft(next(_ids), kind, y0, x, 0, t, tk)
                self.aircraft.append(a)
            elif x != a.x:
                a.d = 1 if x > a.x else -1
                if kind == "heli":
                    self.lane_dir[y0] = a.d
            a.x, a.t = x, t
            seen.add(id(a))
        # a plane entering at an edge (not one leaving: no tracked full plane
        # near that edge yet) sets the side its bomb will come from
        for x0, x1 in s.planes:
            if x0 == 0 and not any(c.kind == "plane" and c.x <= 64 for c in self.aircraft):
                self.entry_side = 1
            elif x1 == 639 and not any(c.kind == "plane" and c.x >= 528 for c in self.aircraft):
                self.entry_side = -1
        for a in list(self.aircraft):
            if id(a) in seen or t - a.t < AIR_GONE_S:
                continue
            self.aircraft.remove(a)
            px = a.x + M.HELI_VX * a.d * round((t - a.t) / M.TICK_S)
            at_edge = a.x <= 8 or a.x >= 640 - (44 if a.kind == "heli" else 48) - 8
            ev = self._evidence(a.id, self.tick(a.t))  # around its last sighting
            if ev:
                fate = "shot_down"
                self.stats["aircraft_down"] += 1
                self.stats["hits"] += 1
            else:
                fate = "left_screen" if at_edge or not 0 <= px <= 600 else "vanished_unexplained"
                if fate == "vanished_unexplained":
                    self.stats["unexplained_fates"] += 1
            self.emit(dict(type="aircraft", id=a.id, kind=a.kind, lane=a.y0, dir=a.d,
                           first_tick=a.first_tick, last_tick=self.tick(a.t), last_x=a.x,
                           fate=fate, evidence=ev, shots=a.pending))

    def aircraft_box(self, a, ticks_ahead=0):
        x = a.x + M.HELI_VX * a.d * ticks_ahead
        return M.heli_box(x, a.y0, a.d) if a.kind == "heli" else M.plane_box(x, 1, a.d)

    # ------------------------------------------------------------------ bombs
    def _bombs(self, s, t, tk):
        for x, y in s.bombs:
            k = M.BOMB_TICK_OF_Y.get(y)
            if not k:  # k=0 frames sit at an irregular x
                continue
            b = next((b for b in self.bombs
                      if abs((x - M.BOMB_VX * b.dir * k) - b.x_release) <= 8 and k >= b.k), None)
            if b is None:
                d = 1 if x < 320 else -1
                xr = x - M.BOMB_VX * d * k
                if not (-24 <= xr <= 200 if d > 0 else 432 <= xr <= 664):
                    continue
                b = Bomb(next(_ids), d, xr, k, k, t, tk)
                self.bombs.append(b)
                self.emit(dict(type="bomb_seen", tick=tk, id=b.id, dir=d, x_release=xr, k=k))
            b.k, b.t = k, t
        for b in list(self.bombs):
            if t - b.t > TRACK_KEEP_S or b.k_est(t) > len(M.BOMB_YS):
                self.bombs.remove(b)
                ev = self._evidence(b.id, self.tick(b.t))
                self.stats["hits"] += bool(ev)
                fate = "shot" if ev else ("landed" if b.k >= M.BOMB_LANDED_K - 1 else "vanished_unexplained")
                self.stats[{"shot": "bombs_shot", "landed": "bombs_landed",
                            "vanished_unexplained": "unexplained_fates"}[fate]] += 1
                self.emit(dict(type="bomb", id=b.id, dir=b.dir, x_release=b.x_release,
                               first_k=b.first_k, last_k=b.k, plan=b.plan, fired=b.fired,
                               fate=fate, evidence=ev))

    def bomb_box(self, b, ticks_ahead=0):
        p = M.bomb_at(b.x_release, b.dir, b.k + ticks_ahead)
        return None if p is None else (p[0], p[1], p[0] + 7, p[1] + 7)

    # ---------------------------------------------------------------- bullets
    def _bullets(self, s, t, tk):
        dots = set(s.dots)
        claimed = set()
        # A bullet moves exactly one lane step per game tick, so on a frame k
        # ticks after its last sighting it must be k steps on -- never "stayed
        # put" once a tick has passed (that let a bullet grab the dot of a
        # second bullet spawned right behind it on the same lane, swapping
        # their identities). +-1 step absorbs clock jitter at tick boundaries.
        for b in sorted(self.bullets, key=lambda b: b.first_tick or 0):
            (vx, vy) = M.LANES[b.lane][1]
            m_exp = max(0, tk - self.tick(b.t))
            order = [m_exp] + [m for m in (m_exp + 1, m_exp - 1) if m >= (1 if m_exp >= 1 else 0)]
            seen_at = None
            for m in order:
                p = (b.pos[0] + vx * m, b.pos[1] + vy * m)
                if p in dots and p not in claimed:
                    seen_at = (m, p)
                    break
            if seen_at:
                m, p = seen_at
                claimed.add(p)
                if m:
                    b.pos, b.t = p, t
                    b.path.append((tk, p[0], p[1]))
                continue
            if (t - b.t) / M.TICK_S > BULLET_GONE_TICKS:
                self._end_bullet(b, tk)
        # orphan candidates (dots at a lane's spawn with no press to explain
        # them): keep the lanes whose next position shows up, promote once
        # exactly one lane remains
        for o in list(self.orphans):
            keep = []
            for lane in o["lanes"]:
                (vx, vy) = M.LANES[lane][1]
                nxt = (o["pos"][0] + vx, o["pos"][1] + vy)
                if nxt in dots and nxt not in claimed:
                    keep.append((lane, nxt))
            if len(keep) == 1:
                lane, nxt = keep[0]
                claimed.add(nxt)
                b = Bullet(next(_ids), o["t"], lane, {"kind": "unattributed"}, pos=nxt, t=t,
                           first_tick=o["tick"], path=[(o["tick"],) + o["pos"], (tk,) + nxt])
                self.bullets.append(b)
                self.orphans.remove(o)
                self.stats["bullets_unattributed"] += 1
            elif not keep and t - o["t"] > 2 * M.TICK_S:
                self.orphans.remove(o)
            elif keep:
                o["lanes"] = [lane for lane, _ in keep]
        # new bullets: a dot on the lane of an expected (pressed) shot
        for e in sorted(self.expected, key=lambda e: e.press_t):
            if t - e.press_t > 0.6:
                self.expected.remove(e)
                self.stats["bullets_not_found"] += 1
                self.emit(dict(type="bullet", shot=e.id, ctx=e.ctx, found=False))
                continue
            if e.lane is None:
                continue
            (sx, sy), (vx, vy) = M.LANES[e.lane]
            # a press at tick T spawns its bullet at T+3 (+-1, measured): only
            # a dot whose inferred spawn tick fits may be this press's bullet,
            # nearest T+3 first -- otherwise an older bullet on the same lane,
            # or the one fired a tick later, gets claimed
            T = self.tick(e.press_t)
            js = [tk - s for s in (T + M.FIRE_LATENCY_TICKS, T + M.FIRE_LATENCY_TICKS - 1,
                                   T + M.FIRE_LATENCY_TICKS + 1) if 0 <= tk - s <= 6]
            for j in js:
                p = (sx + vx * j, sy + vy * j)
                if p in dots and p not in claimed:
                    claimed.add(p)
                    e.pos, e.t, e.first_tick = p, t, tk
                    e.path.append((tk, p[0], p[1]))
                    self.expected.remove(e)
                    self.bullets.append(e)
                    self.stats["bullets_found"] += 1
                    break
        self._find_orphans(dots, claimed, t, tk)

    def _find_orphans(self, dots, claimed, t, tk):
        for p in dots - claimed:
            lanes = [li for li, ((sx, sy), (vx, vy)) in enumerate(M.LANES)
                     if any(p == (sx + vx * j, sy + vy * j) for j in range(0, 6))]
            if lanes and not any(o["pos"] == p for o in self.orphans):
                self.orphans.append(dict(pos=p, t=t, tick=tk, lanes=lanes))

    def _end_bullet(self, b, tk):
        self.bullets.remove(b)
        (vx, vy) = M.LANES[b.lane][1]
        # lane check independent of the clock: every observed step must be a
        # whole number of the lane's per-tick steps
        steps = [(x2 - x1, y2 - y1) for (_, x1, y1), (_, x2, y2) in zip(b.path, b.path[1:])]
        lane_ok = True
        for dx, dy in steps:
            m = round(dx / vx) if vx else round(dy / vy)
            if m < 1 or (dx, dy) != (vx * m, vy * m):
                lane_ok = False
            elif m == 1:
                b.v_measured = (dx, dy)
        hits = self._what_it_hit(b, vx, vy)
        cp = (b.pos[0] + vx, b.pos[1] + vy)
        if hits:
            ent, region, point, _ = hits[0]
            b.end = dict(kind="hit_candidate", tick=tk, point=point, entity=ent, region=region,
                         candidates=[(e, r) for e, r, _, _ in hits])
            for e, r, pt, htk in hits:
                self.recent_ends.append((htk, b.id, e, r, pt))
        elif not (0 <= cp[0] <= 639 and 0 <= cp[1] <= 399):
            b.end = dict(kind="left_screen", tick=tk, point=cp)
            self.stats["left_screen"] += 1
        else:
            b.end = dict(kind="vanished", tick=tk, point=cp)
            self.stats["unexplained_ends"] += 1
        self.recent_ends = [e for e in self.recent_ends if e[0] >= tk - 40]
        rec = dict(type="bullet", shot=b.id, ctx=b.ctx, found=True, lane=b.lane,
                   lane_v=M.LANES[b.lane][1], v_measured=b.v_measured,
                   lane_ok=lane_ok,
                   first_tick=b.first_tick, path=b.path, end=b.end)
        self.emit(rec)

    def _entity_boxes(self, T):
        """Every entity's hitbox(es) at game tick T (float), stepped from the
        tick it was last seen -- bullets and targets on one tick numbering."""
        out = []
        for tr in self.troopers:
            y = tr.y + tr.vy * (T - self.tick(tr.t))
            if tr.state == "canopy":
                out.append((tr.id, "canopy", M.trooper_box(tr.x, y, "canopy")))
            out.append((tr.id, "body", M.trooper_box(tr.x, y, "body")))
        for a in self.aircraft:
            if a.d:
                x = a.x + M.HELI_VX * a.d * (T - self.tick(a.t))
                box = M.heli_box(x, a.y0, a.d) if a.kind == "heli" else M.plane_box(x, 1, a.d)
                out.append((a.id, a.kind, box))
        for b in self.bombs:
            kf = b.k + (T - self.tick(b.t))
            k0 = int(kf // 1)
            p0, p1 = M.bomb_at(b.x_release, b.dir, k0), M.bomb_at(b.x_release, b.dir, k0 + 1)
            if p0 and p1:
                fr = kf - k0
                x, y = p0[0] + (p1[0] - p0[0]) * fr, p0[1] + (p1[1] - p0[1]) * fr
                out.append((b.id, "bomb", (x - 1, y - 1, x + 8, y + 8)))
        return out

    def _what_it_hit(self, b, vx, vy):
        """Walk the bullet's path up to two ticks past its last sighting (a
        bullet merges into a sprite's pixels just before hitting it, so its
        last *visible* position can be a tick early), sub-stepped like the
        planner, against every entity at the same game tick.
        -> (entity id, region, point, tick) or None."""
        tb = self.tick(b.t)
        found = []  # every entity on the path: stacked lanes make the first one ambiguous
        for m in (1, 2):
            for s in (0.25, 0.5, 0.75, 1.0):
                f = m - 1 + s
                p = (b.pos[0] + vx * f, b.pos[1] + vy * f)
                for ent, region, box in self._entity_boxes(tb + f):
                    if _in(box, p) and all(e != ent for e, *_ in found):
                        found.append((ent, region, (round(p[0]), round(p[1])), tb + m))
        return found or None

    def _evidence(self, entity_id, tk, region=None):
        """Shot id of a bullet that ended inside this entity within
        EVIDENCE_TICKS of now -- the only way a kill gets credited."""
        for e_tk, shot, ent, reg, _ in reversed(self.recent_ends):
            if ent == entity_id and abs(tk - e_tk) <= EVIDENCE_TICKS and (region is None or reg == region):
                return shot
        return None

    # --------------------------------------------------------------- troopers
    def _troopers(self, s, t, tk):
        seen = set()
        for x, y in s.bodies:
            tr = next((tr for tr in self.troopers if tr.x == x and id(tr) not in seen
                       and tr.y - 2 <= y <= tr.y + M.FREE_VY * max(1, (t - tr.t) / M.TICK_S) + 8), None)
            if tr is None:
                if any(sx == x and abs(sy - y) <= 2 for sx, sy in self.standing):
                    continue  # a trooper already standing here, not a new one
                tr = Trooper(next(_ids), x, y, t, tk, y)
                tr.y_since = tk
                self.troopers.append(tr)
            elif y != tr.y:
                tr.hist.append((t, y))
                tr.y, tr.t = y, t
                tr.y_since = tk
            else:
                tr.t = t
            tr.y_by_tick[tk] = y
            seen.add(id(tr))
        for bx, by in s.canopies:
            for tr in self.troopers:
                if tr.x == bx and abs(tr.y - by) <= 12 and tr.state in ("free", "canopy"):
                    tr.canopy_hits += 1
                    tr.canopy_last_t = t
                    if tr.state == "free" and tr.canopy_hits >= 2:
                        tr.state = "canopy"
                        tr.canopy_open = (tk, tr.y)
        for tr in self.troopers:
            if tr.state == "canopy" and tr.canopy_last_t is not None and t - tr.canopy_last_t > 0.15:
                recent = [h for h in tr.hist if h[0] >= tr.canopy_last_t - 0.02]
                if len(recent) >= 2 and (recent[-1][1] - recent[0][1]) / max(
                        1e-3, (recent[-1][0] - recent[0][0]) / M.TICK_S) > 6:
                    tr.state = "doomed"
                    tr.canopy_lost = (tk, tr.y)
                    tr.chute_evidence = self._evidence(tr.id, self.tick(tr.canopy_last_t), "canopy")
                    if tr.pending:
                        tr.pending = None
            p = tr.pending
            if p and t > p["t_meet"] + 3 * M.TICK_S and id(tr) in seen and tr.state in ("free", "canopy"):
                tr.pending = None  # missed (the ledger has the bullet's own record)
        for tr in list(self.troopers):
            # stopped moving (it falls 4-8 px every tick) -> it has landed, on
            # the ground or on top of a stack; it is no longer a target
            if id(tr) in seen and tr.state != "doomed" and tk - getattr(tr, "y_since", tk) >= STILL_TICKS:
                self.troopers.remove(tr)
                self.standing.append((tr.x, tr.y))  # body top where it stopped
                self._close_trooper(tr, self.tick(tr.t), landed=True)
                continue
            gone = t - tr.t
            if gone <= TROOPER_GONE_S:
                continue
            self.troopers.remove(tr)
            self._close_trooper(tr, self.tick(tr.t))  # evidence around its last sighting
        # forget standing troopers that are gone (crushed, or the stack cleared)
        bodies_now = set(s.bodies) | set(s.grounded)
        self.standing = [(x, y) for x, y in self.standing if (x, y) in bodies_now]

    def ground_y(self, x):
        """Body top of a trooper when it lands in column x: the ground, or
        16 px higher per trooper already stacked there."""
        stack = sum(1 for lx, _ in self.landed if abs(lx - x) <= 6)
        return M.GROUND_BODY_Y - 16 * stack

    def _close_trooper(self, tr, tk, landed=False):
        near_ground = landed or tr.y >= M.GROUND_BODY_Y - 40
        rec = dict(type="trooper", id=tr.id, x=tr.x, first_tick=tr.first_tick, first_y=tr.first_y,
                   canopy_open=tr.canopy_open, canopy_lost=tr.canopy_lost,
                   y_by_tick=sorted(tr.y_by_tick.items()), engagements=tr.engagements,
                   feasible_ever=tr.feasible_ever, busy_skips=tr.busy_skips, last_y=tr.y)
        if tr.state == "doomed":
            rec["fate"] = "chute_killed"
            rec["fate_evidence"] = getattr(tr, "chute_evidence", None) or "unexplained"
            self.stats["hits"] += rec["fate_evidence"] != "unexplained"
            rec["fell_dead"] = near_ground
            self.stats["trooper_chute_killed"] += 1
            if near_ground:
                self.stats["trooper_fell_dead"] += 1
            # crush: a landed trooper in this column just before impact that is
            # gone now (checking only *now* misses it -- a crush removes it)
            last_tk = self.tick(tr.t)
            before = [h for tk2, h in self.landed_hist if last_tk - 6 <= tk2 <= last_tk - 1]
            under_before = {lx for heads in before for lx, ly in heads if abs(lx - tr.x) <= 6}
            if under_before:
                still = any(abs(lx - tr.x) <= 6 for lx, ly in self.landed)
                rec["over_landed_at"] = sorted(under_before)[0]
                rec["crushed_landed_at"] = None if still else sorted(under_before)[0]
                if not still:
                    self.stats["crushed"] += 1
        elif near_ground:
            rec["fate"] = "landed"
            rec["side"] = "left" if tr.x < 320 else "right"
            # 1 + troopers already standing below it in this column
            rec["stack_height"] = 1 + sum(1 for lx, ly in self.landed if abs(lx - tr.x) <= 6 and ly > tr.y)
            # note: "not reachable from the turret" is judged from wherever the
            # turret actually was -- tools/audit_misses.py separates truly
            # unreachable from out-of-position by replaying the trajectory
            rec["missed_reason"] = ("never_reachable_from_turret" if not tr.feasible_ever and not tr.engagements
                                    else "reachable_but_turret_busy" if not tr.engagements
                                    else "engaged_and_missed")
            self.stats["trooper_landed"] += 1
        else:
            ev = self._evidence(tr.id, tk, "body") or self._evidence(tr.id, tk)
            rec["fate"] = "body_killed" if ev else "lost"
            rec["fate_evidence"] = ev or "unexplained"
            if ev:
                self.stats["trooper_body_killed"] += 1
                self.stats["hits"] += 1
            else:
                self.stats["unexplained_fates"] += 1
        if rec.get("fate_evidence") == "unexplained" and rec["fate"] != "lost":
            self.stats["unexplained_fates"] += 1
        self.emit(rec)

    def candidates(self):
        """Targetable troopers: falling, not already being shot at, and still
        above the lowest point the gun can reach in their column (below it no
        shot exists -- and they only go lower)."""
        return [tr for tr in self.troopers if tr.state in ("free", "canopy") and tr.pending is None
                and tr.y <= M.REACH_FLOOR.get(tr.x, M.GROUND_BODY_Y)]

    def close_all(self, t):
        tk = self.tick(t)
        for tr in list(self.troopers):
            self.troopers.remove(tr)
            rec_fate = tr.state
            self.emit(dict(type="trooper", id=tr.id, x=tr.x, first_tick=tr.first_tick, first_y=tr.first_y,
                           canopy_open=tr.canopy_open, fate="open_at_game_over", state=rec_fate,
                           engagements=tr.engagements, last_y=tr.y))
        for b in list(self.bullets):
            self._end_bullet(b, tk)


def _in(box, p, pad=0):
    x0, y0, x1, y1 = box
    return x0 - pad <= p[0] + 1 and p[0] <= x1 + pad and y0 - pad <= p[1] + 1 and p[1] <= y1 + pad
