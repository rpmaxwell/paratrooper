"""Heuristic policy: today's behavior (scripts/bomb_defense.py) on the new
world model, table planners and non-blocking turret. Baseline for RL and
the parity target for the refactor.

Priorities, checked every frame:
  1. bombs -- a reflex that preempts anything (widest reachable window,
     <=3 bullets in flight, stop once a bullet's meeting tick has passed
     and the bomb is gone)
  2. phase 2: planes, only from positions near the bomb park position and
     only with no bomb in the air; otherwise park for the entry side
  3. troopers: most urgent first; canopy (bigger, slower, can crush a
     landed trooper) else point-blank body; one bullet when the window
     absorbs +-1 tick, else two; then leave it alone until its meeting tick
  4. pre-aim at a free-faller's predicted canopy point
  5. helicopters (yield to any new trooper / state change)
  6. late in phase 1 with nothing to do: park left for the planes

A Job is one engagement: move -> (clamp) -> fire at planned ticks, timed
off the target's own tick counter. Everything is non-blocking: step()
does a little work per frame and returns.
"""
import os
import time

from ..physics import model as M
from ..physics import plan as P
from ..physics import prob_plan as PP
from ..physics import tables as TB

DEFAULT_PARK = {1: 14, -1: 4}   # 125.8 deg for bombs from the left, 55.8 from the right
PLANE_MAX_PARK_DIST = 5
MAX_BOMB_IN_FLIGHT = 3
# Round-4 bombers drop bombs in pairs 2-4 ticks apart; planning the first
# alone leaves the second out of reach most of the time. "1": plan a pair
# from one position (physics.plan.plan_bomb_pair).
BOMB_PAIRS = os.environ.get("PARATROOPER_BOMB_PAIRS") == "1"
MAX_PAIR_GAP = 6       # release ticks apart to count as a pair
LATE_GAP_S = 30
P_MIN = 0.5            # engage a free-faller when the best shot hits with at least this probability
MAX_FREE_PLANS = 2     # probability plans per step (2 ms each): most urgent free-fallers only


def urgency(tr, landed):
    ticks = (M.GROUND_BODY_Y - tr.y) / tr.vy
    over = any(abs(lx - tr.x) <= 6 for lx, _ in landed)
    side = sum(1 for lx, _ in landed if (lx < 320) == (tr.x < 320))
    return ticks - 6 * side - (100 if over and tr.state == "canopy" else 0), over


class Job:
    def __init__(self, kind, target, pos, fire, meet, clamp=False, tick_of=None, abort=None, info=None):
        self.kind, self.target, self.pos, self.fire_ticks = kind, target, pos, list(fire)
        self.meet, self.clamp = meet, clamp
        self.tick_of = tick_of      # () -> target's current tick relative to the plan
        self.abort = abort          # () -> reason or None
        self.info = info or {}
        self.stage = "move"
        self.tick0 = None           # game tick of the plan's tick 0
        self.shots = []             # shot ids fired by this job
        self.fired = []
        self.t_start = time.time()


class HeuristicPolicy:
    def __init__(self, world, turret, log=print):
        self.w, self.turret, self.log = world, turret, log
        self.job = None
        self.parked_for = None
        self.pre_aimed = None
        self.t_start = time.time()
        self.stats = dict(jobs={}, aborted={}, fired={})

    # ----------------------------------------------------------------- driver
    def _free_plan(self, tr):
        """Hit-probability plan for a free-faller, cached per tick / turret state."""
        w = self.w
        key = (tr.id, tr.y, w.barrel, w.clamped)
        cache = getattr(self, "_pp_cache", {})
        if key not in cache:
            drop_y = tr.first_y if tr.first_y <= 60 else 33
            s_now = max(0, (tr.y - drop_y) // M.FREE_VY)  # ticks since the drop (8 px/tick)
            cache = {k: v for k, v in cache.items() if k[0] != tr.id}
            cache[key] = PP.plan_free(tr.x, tr.y, s_now, w.barrel, w.clamped, ground_y=w.ground_y(tr.x))
            self._pp_cache = cache
        return cache[key]

    def _crush_plan(self, tr):
        """Chute-only shot for a trooper over a landed one -> (pos, fire,
        meet, clamp, p_hit) or None. Cached per tick / turret state."""
        w = self.w
        key = ("crush", tr.id, tr.y, tr.state, w.barrel, w.clamped)
        cache = getattr(self, "_crush_cache", {})
        if key not in cache:
            cache = {k: v for k, v in cache.items() if k[1] != tr.id}
            cache[key] = self._crush_plan_uncached(tr)
            self._crush_cache = cache
        return cache[key]

    def _crush_plan_uncached(self, tr):
        w = self.w
        if tr.state == "canopy":
            plan = P.plan_trooper(tr.x, tr.y, tr.vy, "canopy", w.barrel, w.clamped, chute_only=True,
                                  ground_y=w.ground_y(tr.x))
            if not plan:
                return None
            pos, window, meet, clamp = plan
            mid = len(window) // 2
            fire = [window[mid]] if len(window) >= 4 else window[max(0, mid - 1):mid + 1]
            return pos, fire, meet, clamp, 1.0
        drop_y = tr.first_y if tr.first_y <= 60 else 33
        pp = PP.plan_free(tr.x, tr.y, max(0, (tr.y - drop_y) // M.FREE_VY), w.barrel, w.clamped,
                          chute_only=True, ground_y=w.ground_y(tr.x))
        if not pp or pp["p_hit"] < P_MIN:
            return None
        return pp["pos"], pp["spawns"], pp["meet_last"], pp["clamp"], pp["p_hit"]

    def _track_feasibility(self):
        """Mark troopers some plan could reach (from wherever the turret is),
        even while busy -- so a landed trooper's missed_reason can say
        'turret busy' vs 'unreachable' truthfully."""
        w = self.w
        for tr in w.candidates():
            if tr.feasible_ever or tr.state != "canopy":
                continue  # free-fallers: set when their probability plan clears P_MIN
            if P.plan_trooper(tr.x, tr.y, tr.vy, "canopy", w.barrel, w.clamped, swept=True):
                tr.feasible_ever = True

    def step(self, t):
        w = self.w
        self._frame = getattr(self, "_frame", 0) + 1
        if self._frame % 4 == 0:
            self._track_feasibility()
        # 1. bomb reflex
        unplanned = [b for b in w.bombs if b.plan is None]
        if unplanned and not (self.job and self.job.kind == "bomb"):
            if self.job:
                self._abort("bomb")
            self._start_bomb(max(unplanned, key=lambda b: b.k), t)
        if self.job:
            self._run_job(t)
            return
        if self.turret.busy():
            return
        # idle planning at most once per game tick: nothing moves in between
        tk = w.tick(t)
        if tk == getattr(self, "_idle_tick", None) and not w.bombs:
            return
        self._idle_tick = tk
        if w.bombs:
            unplanned = [b for b in w.bombs if b.plan is None]
            if unplanned:
                self._start_bomb(max(unplanned, key=lambda b: b.k), t)
            return
        if w.phase == 2:
            if self._start_plane(t):
                return
            if w.entry_side is not None:
                self._park(w.entry_side)
            return
        self.parked_for = None
        if self._start_trooper(t) or self._pre_aim(t) or self._start_heli(t):
            return
        if w.phase == "gap" and t - self.t_start > LATE_GAP_S:
            self._park(1)

    def _park(self, side):
        pos = DEFAULT_PARK[side]
        if self.parked_for == side and not self.turret.busy():
            return
        self.parked_for = side
        if self.turret.goto(pos, {"kind": "stop", "why": "park"}) is None:
            self.parked_for = None  # keys not settled: retry next frame

    def _replan_bomb_if_unfired(self, j):
        if j.kind == "bomb" and not j.fired and j.target in self.w.bombs:
            j.target.plan = None
        if j.kind == "bomb" and "pair" in j.info:
            b2, _, fire2, _ = j.info["pair"]
            if b2 in self.w.bombs and not any(f in fire2 for f in j.fired):
                b2.plan = None

    def _abort(self, why):
        j = self.job
        self._replan_bomb_if_unfired(j)
        self.stats["aborted"][j.kind] = self.stats["aborted"].get(j.kind, 0) + 1
        self.turret.stop(why)
        self.job = None

    def _run_job(self, t):
        j = self.job
        why = j.abort() if j.abort else None
        if not why and j.stage == "move" and t - j.t_start > 3.0:
            why = "move_timeout"
        if why:
            self._abort(why)
            return
        if j.stage == "move":
            if self.turret.busy():
                return
            # the stop press fires a bullet down the planned lane a few ticks
            # early -- it often makes the kill, so it is tagged with the target
            started = None
            if self.w.barrel != j.pos:
                started = self.turret.goto(j.pos, {"kind": "stop", "why": j.kind, "target": j.target.id})
                if started is None:
                    return  # keys not settled yet: try again next frame
                if not started and self.w.barrel is None:
                    return
            if self.turret.busy():
                return
            if self.w.barrel is not None and self.w.barrel != j.pos:
                self._abort("goto_missed")
                return
            if j.clamp:
                self.turret.clamp(j.pos)
                self.turret._ctx = {"kind": "clamp", "why": j.kind, "target": j.target.id}
                j.stage = "clamp"
                return
            j.stage = "fire"
        if j.stage == "clamp":
            if self.turret.busy():
                return
            j.stage = "fire"
        if j.stage == "fire":
            k = j.tick_of()
            if k is None:
                self._finish(j)
                return
            spawn = k + M.FIRE_LATENCY_TICKS
            if spawn > j.fire_ticks[-1]:
                self._finish(j)
                return
            if spawn in j.fire_ticks and spawn not in j.fired and self._may_fire(j, k):
                j.fired.append(spawn)
                ctx = dict(kind=j.kind, target=j.target.id, pos=j.pos, spawn=spawn, meet=j.meet,
                           plan_spawn_tick=(j.tick0 + spawn) if j.tick0 is not None else None, **j.info)
                shot = self.turret.fire(ctx, j.pos)
                j.shots.append(shot)
                if j.kind == "trooper":
                    j.target.engagements.append(shot)
                elif j.kind in ("heli", "plane"):
                    j.target.pending.append(shot)
                if len(j.fired) == len(j.fire_ticks) and j.kind != "bomb":
                    self._finish(j)

    def _finish(self, j):
        self._replan_bomb_if_unfired(j)
        self.stats["fired"][j.kind] = self.stats["fired"].get(j.kind, 0) + len(j.fired)
        if j.fired and j.kind == "trooper":
            # wait for these bullets' outcome: cleared early by the world when
            # they all end without a kill, else resolved after the meeting tick
            j.target.pending = dict(tick_meet=j.tick0 + j.meet, shots=set(j.shots))
        if j.fired and j.kind in ("heli", "plane"):
            j.target.busy_until = self.w.clock.t0 + (j.tick0 + j.meet + 2 + 1) * M.TICK_S
        self.job = None

    def _may_fire(self, j, k):
        if j.kind != "bomb":
            return True
        if "pair" in j.info:
            return self._may_fire_pair(j, k)
        b = j.target
        meets = [P.bomb_meet(b.x_release, b.dir, M.lane_id(j.pos, j.pos, self.w.clamped), f) or f + 8
                 for f in j.fired]
        return sum(1 for m in meets if m >= k) < MAX_BOMB_IN_FLIGHT

    def _new_job(self, job):
        self.job = job
        self.stats["jobs"][job.kind] = self.stats["jobs"].get(job.kind, 0) + 1

    # ------------------------------------------------------------------ bombs
    def _bomb_partner(self, b):
        """An unplanned bomb released just after b by the same plane -> (bomb, gap) or None."""
        rel = self.w.tick(b.t) - b.k
        best = None
        for b2 in self.w.bombs:
            if b2 is b or b2.plan is not None or b2.dir != b.dir:
                continue
            gap = (self.w.tick(b2.t) - b2.k) - rel
            if 0 < gap <= MAX_PAIR_GAP and (best is None or gap < best[1]):
                best = (b2, gap)
        return best

    def _bomb_done(self, j, b, fire):
        """b is dead or gone: missing for 2 ticks after one of its bullets."""
        return b not in self.w.bombs or (any(f in fire for f in j.fired) and (time.time() - b.t) / M.TICK_S >= 2)

    def _may_fire_pair(self, j, k):
        b, (b2, fire1, fire2, gap) = j.target, j.info["pair"]
        spawn = k + M.FIRE_LATENCY_TICKS
        mine = []
        if spawn in fire1 and not self._bomb_done(j, b, fire1):
            mine.append((b, fire1, 0))
        if spawn in fire2 and not self._bomb_done(j, b2, fire2):
            mine.append((b2, fire2, gap))
        if not mine:
            return False   # that bomb is already down: don't spray its run
        lane = M.lane_id(j.pos, j.pos, self.w.clamped)
        x, fire, g = mine[0]
        meets = [P.bomb_meet(x.x_release, x.dir, lane, f - g) or f + 8 for f in j.fired if f in fire]
        return sum(1 for m in meets if m >= k) < MAX_BOMB_IN_FLIGHT

    def _start_bomb_pair(self, b, b2, gap):
        plan = P.plan_bomb_pair(b.x_release, b2.x_release, b.dir, gap, b.k, self.w.barrel, self.w.clamped)
        if plan is None:
            return False
        pos, run1, run2 = plan
        fire1 = list(range(run1[0] - 1, run1[-1] + 1))
        fire2 = list(range(run2[0] - 1, run2[-1] + 1))
        b.plan, b2.plan = (pos, run1), (pos, [f - gap for f in run2])
        tick0 = self.w.tick(b.t) - b.k
        job = None

        def tick_of():
            if self._bomb_done(job, b, fire1) and self._bomb_done(job, b2, fire2):
                return None
            return self.w.tick(time.time()) - tick0

        job = Job("bomb", b, pos, sorted(set(fire1) | set(fire2)), None, tick_of=tick_of,
                  info=dict(x_release=b.x_release, pair=(b2, fire1, fire2, gap)))
        job.tick0 = tick0
        job.fired = b.fired  # shared: the first bomb's record logs the pair's shots (its frame)
        self._new_job(job)
        self.log(f"BOMB PAIR {b.id}+{b2.id} gap {gap} k={b.k}: pos {pos} spawn {run1} + {run2} (barrel {self.w.barrel})")
        return True

    def _start_bomb(self, b, t):
        partner = self._bomb_partner(b) if BOMB_PAIRS else None
        if partner and self._start_bomb_pair(b, *partner):
            return
        plan = P.plan_bomb(b.x_release, b.dir, b.k, self.w.barrel, self.w.clamped)
        if plan is None:
            b.plan = (None, [])
            self.log(f"bomb {b.id}: no reachable intercept (k={b.k}, barrel {self.w.barrel})")
            return
        pos, run = plan
        b.plan = plan
        fire = list(range(run[0] - 1, run[-1] + 1))

        tick0 = self.w.tick(b.t) - b.k

        def tick_of():
            if b not in self.w.bombs:
                return None
            # game clock, not an extrapolation: the press then lands early in
            # the tick, where press->spawn latency is a reliable 3 ticks
            k = self.w.tick(time.time()) - tick0
            # stop as soon as the bomb has been missing for 2 ticks: an early
            # bullet often kills it before the planned meeting tick, and
            # waiting for that tick sent up to 9 more shots into the spray
            # (45 of 88 bomb kills). Bombs aren't hidden in open sky; if one
            # does reappear, the world clears its plan and it is re-planned.
            if b.fired and (time.time() - b.t) / M.TICK_S >= 2:
                return None
            return k

        job = Job("bomb", b, pos, fire, None, tick_of=tick_of, info=dict(x_release=b.x_release))
        job.tick0 = tick0
        job.fired = b.fired  # shared: the bomb record logs what was fired
        self._new_job(job)
        self.log(f"BOMB {b.id} k={b.k} x_release={b.x_release}: pos {pos} spawn {run} (barrel {self.w.barrel})")

    # ----------------------------------------------------------------- planes
    def _start_plane(self, t):
        w = self.w
        if w.bombs or w.entry_side is None:
            return False
        park = DEFAULT_PARK[w.entry_side]
        allowed = range(max(0, park - PLANE_MAX_PARK_DIST), min(M.N_POS - 1, park + PLANE_MAX_PARK_DIST) + 1)
        best = None
        for a in w.aircraft:
            if a.kind != "plane" or not a.d or a.busy_until > t:
                continue
            plan = P.plan_air("plane", a.x, 1, a.d, w.barrel, w.clamped, allowed=allowed)
            if plan and (best is None or plan[2] < best[1][2]):
                best = (a, plan)
        if best is None:
            return False
        a, (pos, window, meet) = best
        self.parked_for = None
        self._new_job(self._air_job("plane", a, pos, window, meet, t,
                                    abort=lambda: "bomb" if self.w.bombs else None, shots_if_wide=1))
        return True

    # ------------------------------------------------------------ helicopters
    def _start_heli(self, t):
        w = self.w
        best = None
        for a in w.aircraft:
            if a.kind != "heli" or not a.d or a.busy_until > t:
                continue
            plan = P.plan_air("heli", a.x, a.y0, a.d, w.barrel, w.clamped)
            if plan and (best is None or plan[2] < best[1][2]):
                best = (a, plan)
        if best is None:
            return False
        a, (pos, window, meet) = best
        known = {(tr.id, tr.state) for tr in w.candidates()}

        def abort():
            if self.w.bombs or self.w.phase == 2:
                return "bomb_or_planes"
            if {(tr.id, tr.state) for tr in self.w.candidates()} - known:
                return "new_trooper"
            return None

        self._new_job(self._air_job("heli", a, pos, window, meet, t, abort=abort, shots_if_wide=2))
        return True

    def _air_job(self, kind, a, pos, window, meet, t, abort, shots_if_wide):
        mid = len(window) // 2
        fire = [window[mid]] if (len(window) >= 3 and shots_if_wide == 1) else window[max(0, mid - 1):mid + 1]
        tick0 = self.w.tick(a.t_x)  # tick the observed x belongs to

        def tick_of():
            if a not in self.w.aircraft:
                return None
            return self.w.tick(time.time()) - tick0

        job = Job(kind, a, pos, fire, meet, tick_of=tick_of, abort=abort, info=dict(t_obs=t, lane=a.y0))
        job.tick0 = tick0
        return job

    # --------------------------------------------------------------- troopers
    def _start_trooper(self, t):
        w = self.w
        cands = w.candidates()
        if not cands:
            return False
        ranked = sorted(cands, key=lambda tr: urgency(tr, w.landed)[0])
        free_planned = 0
        for tr in ranked[:4]:
            over = urgency(tr, w.landed)[1]
            # above a landed trooper: a chute-only shot (it falls onto the one
            # below and both are removed) if one is likely enough, else kill it
            crush = self._crush_plan(tr) if over else None
            if crush:
                # directly above a landed trooper: shoot only the chute -- it
                # falls onto the one below and both are removed
                pos, fire, meet, clamp, p_hit = crush
                part = "chute_only"
            elif tr.state == "canopy":
                # prefer a shot that overlaps at a drawn position (kills ~91%);
                # else one whose only contact is between ticks (~61%)
                part, p_hit = "canopy", TB.P_DRAWN
                plan = P.plan_trooper(tr.x, tr.y, tr.vy, part, w.barrel, w.clamped, ground_y=w.ground_y(tr.x))
                if not plan:
                    plan = P.plan_trooper(tr.x, tr.y, tr.vy, part, w.barrel, w.clamped,
                                          ground_y=w.ground_y(tr.x), swept=True)
                    p_hit = TB.P_SWEPT_ONLY
                if not plan:
                    continue
                pos, window, meet, clamp = plan
                mid = len(window) // 2
                fire = [window[mid]] if len(window) >= 4 else window[max(0, mid - 1):mid + 1]
            elif os.environ.get("PARATROOPER_FREE") == "pointblank":
                # previous rule (point-blank body shots only), kept for A/B tests
                plan = P.plan_trooper(tr.x, tr.y, tr.vy, "body", w.barrel, w.clamped, free=True)
                if not plan:
                    continue
                pos, window, meet, clamp = plan
                mid = len(window) // 2
                fire = [window[mid]] if len(window) >= 4 else window[max(0, mid - 1):mid + 1]
                part, p_hit = "body", 1.0
            else:
                # free-faller: best shot over every chute-opening scenario
                if free_planned >= MAX_FREE_PLANS:
                    continue
                free_planned += 1
                pp = self._free_plan(tr)
                if not pp or pp["p_hit"] < P_MIN:
                    continue
                part, p_hit = "free", pp["p_hit"]
                pos, fire, meet, clamp = pp["pos"], pp["spawns"], pp["meet_last"], pp["clamp"]
            tr.feasible_ever = True
            y_obs, state = tr.y, tr.state
            over = urgency(tr, w.landed)[1]

            tick0 = self.w.tick(tr.t_y)  # tick the observed y belongs to

            def tick_of(tr=tr, tick0=tick0):
                if tr not in self.w.troopers:
                    return None
                return self.w.tick(time.time()) - tick0

            def abort(tr=tr, state=state):
                if self.w.bombs:
                    return "bomb"
                if self.w.phase == 2:
                    return "planes"
                if tr.state != state and not (state == "free" and tr.state == "canopy"):
                    return "target_changed"  # (a free-faller opening is covered by its plan)
                return None

            self.pre_aimed = None
            job = Job("trooper", tr, pos, fire, meet, clamp=clamp, tick_of=tick_of, abort=abort,
                      info=dict(t_obs=t, part=part, state=state, y_obs=y_obs, clamp=clamp, over_landed=over,
                                p_hit=round(p_hit, 3)))
            job.tick0 = tick0
            self._new_job(job)
            return True
        return False

    def _pre_aim(self, t):
        w = self.w
        for tr in sorted(w.candidates(), key=lambda tr: urgency(tr, w.landed)[0]):
            if tr.state != "free":
                continue
            y_open = max(tr.y, tr.first_y + M.CANOPY_OPEN_FALL_PX)
            plan = P.plan_trooper(tr.x, y_open, M.CANOPY_VY, "canopy", None)
            if not plan:
                continue
            pos = plan[0]
            if pos == w.barrel or self.pre_aimed == (tr.id, pos):
                return False
            if self.turret.goto(pos, {"kind": "stop", "why": "pre_aim"}) is None:
                return True  # keys not settled: retry next frame
            self.pre_aimed = (tr.id, pos)
            return True
        return False
