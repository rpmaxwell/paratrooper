#!/usr/bin/env python3
"""Full player: phase-1 paratrooper defense (existing solver) + phase-2
bomb interception built on the measured tick model in bomb_model.py.

Phase 2 policy:
  * On the helicopter->plane transition (or any plane entering), park the
    barrel on the default intercept position for that side, so a bomb
    finds us already (nearly) aimed.
  * Bomb detection only accepts the exact 8x8/48px white disc whose y is
    on the known fall sequence and whose implied release x is plausible.
    Explosion shrapnel (turret or bomb) never matches that, so the
    targeting can't lock onto debris.
  * Each bomb's whole future path is known from ONE detection (tick index
    = position of its y in BOMB_YS). plan_intercept simulates every barrel
    position tick by tick and picks the reachable one with the widest
    window of hitting spawn ticks; we go there and fire exactly when the
    bomb's own tick counter says the bullet will spawn inside that window.

Every episode is recorded (record_episode.Recorder) and dumped on game
over for offline analysis.

Usage: python3 bomb_defense.py [n_episodes] [out_dir]
"""
import json
import os
import subprocess
import sys
import time

import mss
import numpy as np

from actions import do_action
from aim_solver import LATENCY_OFFSET_S, TICK_SAFETY_MARGIN_S, solve_intercept, valid_barrel_angle
from bomb_model import (BARREL, BOMB_TICK_OF_Y, BOMB_YS, FIRE_LATENCY_TICKS, TICK_S, PhaseTracker,
                        barrel_pos, bullet_hits, lane, plan_intercept, sky_sprites, to_index)
from play_one_game import find_paratrooper_target, grab
from read_score import is_done, read_score
from record_episode import Recorder
from threats import detect_barrel_angle
from trooper_model import TrooperTracker, plan_trooper, predicted_canopy_trooper, urgency
from heli_model import HELI_VX, find_helis, plan_heli, plane_box
import turret
from turret import clamp, goto, key

DEFAULT_POS = {1: 14, -1: 4}  # 125.8 deg for bombs from the left, 55.8 from the right
FIRE_MARGIN_TICKS = 1  # extra shot before the planned window (latency slop)
MAX_IN_FLIGHT = 3  # bullets allowed en route to a bomb before waiting to see if they hit
LANDED_K = 31  # a bomb still tracked at this tick is hitting the turret
# A bomb can drop out of detection for a few ticks when it overlaps another
# sprite. Keep its track (and plan) alive through gaps this long, firing on
# the tick predicted from elapsed time, instead of retiring and re-planning.
TRACK_KEEP_S = 0.6
HELI_MAX_SHOTS = 2  # helicopters are lowest priority: a short, well-timed pair
# Planes: only from barrel positions this close to the bomb-ready park
# position, so the turret can always get back in time for a bomb.
PLANE_MAX_PARK_DIST = 5


class Bomb:
    def __init__(self, x, y, k, t):
        self.dir = 1 if x < 320 else -1
        self.x_release = x - 8 * self.dir * k
        self.k = k
        self.first_k = k
        self.t = t
        self.plan = None
        self.fired = []
        self.goto_result = None

    def k_est(self, now):
        """Current tick: observed, or extrapolated through a detection gap."""
        return self.k + int(round((now - self.t) / TICK_S))

    def matches(self, x, k):
        return abs((x - 8 * self.dir * k) - self.x_release) <= 8 and k >= self.k

    def log(self):
        return dict(dir=self.dir, x_release=self.x_release, first_k=self.first_k, last_k=self.k,
                    plan_pos=None if not self.plan else self.plan[0],
                    plan_angle=None if not self.plan or self.plan[0] is None else BARREL[self.plan[0]][0],
                    window=None if not self.plan else self.plan[1],
                    goto_result=self.goto_result, fired_spawn_ticks=self.fired)


def plausible_release(x_release, direction):
    return -24 <= x_release <= 200 if direction > 0 else 432 <= x_release <= 664


class Player:
    def __init__(self, sct, mon, log):
        self.sct, self.mon, self.log = sct, mon, log
        self.reset()

    rec = None  # Recorder, set by main(); used for evidence clips
    evidence_dir = None
    evidence_left = {}

    evidence_queue = None

    def evidence(self, tag, seconds=1.5, delay=0.15):
        """Mark an event for a frame clip (checked frame by frame later).
        Only QUEUES it: saving blocks for ~0.3s, which once cost a bomb
        mid-firing-window. flush_evidence() writes clips when idle."""
        if self.rec is None or self.evidence_left.get(tag, 0) <= 0:
            return
        self.evidence_left[tag] -= 1
        if self.evidence_queue is None:
            self.evidence_queue = []
        self.evidence_queue.append((tag, time.time() - seconds, time.time() + delay))

    def flush_evidence(self):
        """Write queued clips -- call only when nothing is in flight."""
        while self.evidence_queue and self.evidence_queue[0][2] <= time.time():
            tag, t_from, t_to = self.evidence_queue.pop(0)
            frames = [f for f in self.rec.snapshot() if t_from <= f[0] <= t_to]
            if not frames:
                continue
            path = f"{self.evidence_dir}/evidence_{tag}_{int(self.t0)}_{int(t_to * 1000)}.npz"
            np.savez_compressed(path, t=np.array([f[0] for f in frames]) - self.t0,
                                frames=np.stack([f[1] for f in frames]))
            self.log(f"  evidence saved: {path}")

    def reset(self):
        self.phase = PhaseTracker()
        self.bombs = []
        self.done_bombs = []
        self.parked_for = None
        self.last_planes = []
        self.entry_side = None
        self.pre_aimed = None
        self.heli_pending = []  # engagements awaiting their meeting tick
        self.lane_dir = {}  # skid y0 -> -1/+1, measured from motion
        self.prev_helis = []
        self.planes_full = []  # [(x0, dir or None)] fully visible planes this frame
        self.plane_pending = []
        self.plane_stats = dict(engaged=0, shots=0, kills=0, misses=0, unresolved=0, aborted=0)
        self.heli_stats = dict(engaged=0, shots=0, kills=0, misses=0, unresolved=0, aborted=0)
        self.troopers = TrooperTracker(log=lambda m: self.log(m),
                                       on_miss=lambda: self.evidence("trooper_miss", seconds=2.0, delay=0))
        self.t0 = time.time()

    def observe(self, idx, t=None):
        """Update phase + bomb tracks from one palette-indexed frame."""
        t = time.time() if t is None else t
        s = sky_sprites(idx)
        prev_phase = self.phase.phase
        ph = self.phase.update(s, t)
        if ph != prev_phase:
            self.log(f"phase {prev_phase} -> {ph}  planes={s['planes']}")
        for x0, x1 in s["planes"]:
            if (x0 == 0 or x1 == 639) and not any(abs(x0 - a) <= 24 for a, _ in self.last_planes):
                self.entry_side = 1 if x0 == 0 else -1
        # fully-visible planes, direction measured from motion
        full = []
        for x0, x1 in s["planes"]:
            if x1 - x0 != 47:
                continue
            d = None
            for px, _ in self.planes_full:
                if 0 < abs(x0 - px) <= 16:
                    d = 1 if x0 > px else -1
            full.append((x0, d))
        self.planes_full = full
        self._resolve_planes(t)
        self.last_planes = s["planes"]
        for x, y in s["bombs"]:
            k = BOMB_TICK_OF_Y.get(y)
            # k=0 (y=17) frames sit at an irregular x (the bomb then jumps
            # 16px into tick 1), so they'd give a wrong release x
            if not k:
                continue
            for b in self.bombs:
                if b.matches(x, k):
                    b.k, b.t = k, t
                    break
            else:
                b = Bomb(x, y, k, t)
                if plausible_release(b.x_release, b.dir):
                    self.bombs.append(b)
                    self.log(f"BOMB seen k={k} at ({x},{y}) dir={b.dir:+d} x_release={b.x_release}")
        self._learn_lane_dirs(idx)
        self._resolve_helis(idx, t)
        self.troopers.update(idx, t)
        # retire bombs that have been gone too long to be an occlusion, or
        # that must have landed by now
        for b in list(self.bombs):
            if t - b.t > TRACK_KEEP_S or b.k_est(t) > len(BOMB_YS):
                self.bombs.remove(b)
                outcome = "LANDED" if b.k >= LANDED_K - 1 else "GONE(hit?)"
                if outcome == "LANDED":
                    self.evidence("bomb_landed", seconds=2.5, delay=0)
                self.log(f"bomb x_release={b.x_release} ended at k={b.k}: {outcome}")
                d = b.log()
                d["outcome"] = outcome
                self.done_bombs.append(d)
        return s

    def grab_observe(self):
        frame = grab(self.sct, self.mon)
        idx = to_index(frame)
        self.observe(idx)
        return frame, idx

    # ---- phase 2 -------------------------------------------------------
    def engage(self, b):
        cur = barrel_pos(to_index(grab(self.sct, self.mon)))
        plan = plan_intercept(b.x_release, b.dir, b.k, cur)
        if plan is None:
            self.log(f"  no reachable intercept for bomb k={b.k} x_release={b.x_release} cur={cur}")
            b.plan = (None, [])
            return
        pos, window = plan
        b.plan = plan
        self.log(f"  plan: pos {pos} ({BARREL[pos][0]} deg) spawn ticks {window} (cur pos {cur}, bomb k={b.k})")
        if cur != pos:
            def watch(idx):
                self.observe(idx)  # keep tracking; never abort this rotation
            b.goto_result = goto(self.sct, self.mon, pos, on_frame=watch)
            self.log(f"  goto {pos} -> {b.goto_result} (bomb k={b.k})")
        # Fire from the start of the window, a few bullets at a time. Each
        # bullet has a known tick at which it meets the bomb; once one of
        # those has passed and the bomb is no longer seen, it's dead --
        # stop, instead of emptying the rest of a long window into the
        # debris (which looks exactly like "shooting at shrapnel").
        fire_ticks = range(window[0] - FIRE_MARGIN_TICKS, window[-1] + 1)
        meets = {}
        while b in self.bombs:
            frame, idx = self.grab_observe()
            if is_done(frame):
                return
            now = time.time()
            k = b.k_est(now)
            unseen_ticks = (now - b.t) / TICK_S
            if meets and min(meets.values()) <= k and unseen_ticks >= 1.5:
                b.killed_by_us = True
                break
            spawn = k + FIRE_LATENCY_TICKS
            if spawn > fire_ticks[-1]:
                break
            in_flight = sum(1 for m in meets.values() if m >= k)
            if spawn in fire_ticks and spawn not in meets and in_flight < MAX_IN_FLIGHT:
                key("Up", ctx=dict(kind="bomb", x_release=b.x_release, dir=b.dir, pos=pos, spawn=spawn, k=b.k))
                b.fired.append(spawn)
                meets[spawn] = bullet_hits(b.x_release, b.dir, pos, spawn, lane(pos, pos)) or spawn + 8
        self.log(f"  fired at spawn ticks {b.fired}")

    def park(self, direction):
        pos = DEFAULT_POS[direction]
        if self.parked_for == direction:
            return
        self.parked_for = direction
        cur = barrel_pos(to_index(grab(self.sct, self.mon)))
        if cur != pos:
            def watch(idx):
                self.observe(idx)
                return bool(self.bombs)  # a bomb takes priority over parking
            r = goto(self.sct, self.mon, pos, on_frame=watch, corrections=0)
            self.log(f"park for dir {direction:+d}: {cur} -> {r}")

    # ---- helicopters (lowest priority) --------------------------------
    def _learn_lane_dirs(self, idx):
        """Measure each lane's flight direction from frame-to-frame motion.
        Lanes change every wave (17/41, 29/65, 41/89, ...), and the same y
        can be the right-flying lane in one wave and the left-flying lane
        in a later one, so a lookup table gets it wrong."""
        helis = find_helis(idx)
        for x, y in helis:
            for px, py in self.prev_helis:
                if py == y and 0 < abs(x - px) <= 16:
                    d = 1 if x > px else -1
                    if self.lane_dir.get(y) != d:
                        self.lane_dir[y] = d
                        self.log(f"lane y0={y} flies {'right' if d > 0 else 'left'}")
        self.prev_helis = helis

    def _resolve_helis(self, idx, t):
        """Once an engaged helicopter's meeting tick has passed (+2 ticks
        for the explosion to replace the sprite), score kill or miss by
        whether it is still where it should be."""
        if not self.heli_pending:
            return
        helis = None
        for e in list(self.heli_pending):
            if t < e["t_meet"] + 2 * TICK_S:
                continue
            if helis is None:
                helis = find_helis(idx)
            k = int(round((t - e["t_obs"]) / TICK_S))
            px = e["x0"] + HELI_VX * e["d"] * k
            if not (0 <= px <= 640 - 44):
                outcome = "unresolved"  # left the screen: can't tell kill from exit
            else:
                # generous: anything still in that lane near the prediction
                # means the helicopter survived
                alive = any(y == e["y0"] and abs(x - px) <= 48 for x, y in helis)
                outcome = "misses" if alive else "kills"
            self.heli_stats[outcome] += 1
            self.log(f"heli lane {e['y0']} {outcome.upper()} (pos {e['pos']}, shots {e['shots']})")
            self.heli_pending.remove(e)

    def _trooper_appeared(self, idx):
        """Cheap per-frame check for a paratrooper in the targeting band --
        cyan below the helicopter lanes, away from the barrel."""
        band = idx[80:340] == 1
        band[260 - 80 - 60:, 320 - 60:320 + 60] = False  # barrel region
        return band.sum() >= 15

    def heli_step(self, idx):
        helis = find_helis(idx)
        if not helis:
            return False
        dirs = self.lane_dir
        busy = {(e["y0"], e["d"]) for e in self.heli_pending}
        cur = barrel_pos(idx)
        best = None
        for x0, y0 in helis:
            if y0 not in dirs:
                continue  # direction not measured yet (needs two frames)
            d = dirs[y0]
            if any(e["y0"] == y0 and abs(e["x0"] + HELI_VX * d * round((time.time() - e["t_obs"]) / TICK_S) - x0) <= 16
                   for e in self.heli_pending):
                continue  # bullets already on their way to this one
            plan = plan_heli(x0, y0, d, cur)
            if plan and (best is None or plan[2] < best[1][2]):
                best = ((x0, y0, d), plan)
        if best is None:
            return False
        (x0, y0, d), (pos, window, meet) = best
        t_obs = time.time()
        self.heli_stats["engaged"] += 1
        self.log(f"HELI lane {y0} x={x0} dir={d:+d}: pos {pos} ({BARREL[pos][0]} deg) "
                 f"spawn ticks {window} meet tick {meet} (cur {cur})")

        known = {(id(tr), tr.state) for tr in self.troopers.candidates()}

        def watch(idx):
            self.observe(idx)
            # yield to a trooper that is new or has changed state (e.g. its
            # canopy opened); ones already known were unreachable
            fresh = {(id(tr), tr.state) for tr in self.troopers.candidates()} - known
            return bool(self.bombs) or self.phase.phase == 2 or bool(fresh)

        if cur != pos:
            r = goto(self.sct, self.mon, pos, on_frame=watch, corrections=0)
            if r != pos:
                self.heli_stats["aborted"] += 1
                self.log(f"  heli engagement aborted (turret at {r})")
                return True
        # fire the middle ticks of the window, timed off the heli's own x
        mid = len(window) // 2
        fire = window[max(0, mid - 1):mid - 1 + HELI_MAX_SHOTS] or window[:HELI_MAX_SHOTS]
        shots = 0
        while True:
            frame, idx = self.grab_observe()
            if is_done(frame) or self.bombs or self.phase.phase == 2:
                break
            now = time.time()
            track = [x for x, y in find_helis(idx) if y == y0]
            pred = x0 + HELI_VX * d * ((now - t_obs) / TICK_S)
            near = [x for x in track if abs(x - pred) <= 16]
            k = (near[0] - x0) // (HELI_VX * d) if near else int(round((now - t_obs) / TICK_S))
            spawn = k + FIRE_LATENCY_TICKS
            if spawn > fire[-1]:
                break
            if spawn in fire and spawn not in getattr(self, "_heli_fired", ()):
                key("Up", ctx=dict(kind="heli", lane=y0, x0=x0, dir=d, pos=pos, spawn=spawn, t_obs=t_obs - self.t0))
                shots += 1
                self._heli_fired = getattr(self, "_heli_fired", set()) | {spawn}
                if shots >= HELI_MAX_SHOTS:
                    break
        self._heli_fired = set()
        self.heli_stats["shots"] += shots
        if shots:
            self.heli_pending.append(dict(x0=x0, y0=y0, d=d, t_obs=t_obs, pos=pos, shots=shots,
                                          t_meet=t_obs + meet * TICK_S))
        return True

    # ---- planes (lowest priority, phase 2 only) ------------------------
    def _resolve_planes(self, t):
        for e in list(self.plane_pending):
            if t < e["t_meet"] + 2 * TICK_S:
                continue
            k = int(round((t - e["t_obs"]) / TICK_S))
            px = e["x0"] + HELI_VX * e["d"] * k
            if not 0 <= px <= 640 - 48:
                outcome = "unresolved"
            else:
                alive = any(abs(x - px) <= 48 for x, _ in self.planes_full)
                outcome = "misses" if alive else "kills"
            self.plane_stats[outcome] += 1
            self.log(f"plane {outcome.upper()} (pos {e['pos']}, shots {e['shots']})")
            self.plane_pending.remove(e)
            self.evidence(f"plane_{outcome}", seconds=2.5)

    def plane_step(self, idx):
        """Shoot a plane only when it can't cost us a bomb: no bomb in the
        air, from a position near the bomb-ready park position, aborting
        the moment a bomb shows up."""
        if self.bombs or self.entry_side is None:
            return False
        park = DEFAULT_POS[self.entry_side]
        allowed = range(max(0, park - PLANE_MAX_PARK_DIST), min(len(BARREL) - 1, park + PLANE_MAX_PARK_DIST) + 1)
        cur = barrel_pos(idx)
        now = time.time()
        best = None
        for x0, d in self.planes_full:
            if d is None:
                continue
            if any(abs(e["x0"] + HELI_VX * e["d"] * round((now - e["t_obs"]) / TICK_S) - x0) <= 24
                   for e in self.plane_pending):
                continue
            plan = plan_heli(x0, 1, d, cur, box=plane_box, allowed=allowed)
            if plan and (best is None or plan[2] < best[1][2]):
                best = ((x0, d), plan)
        if best is None:
            return False
        (x0, d), (pos, window, meet) = best
        t_obs = now
        self.plane_stats["engaged"] += 1
        self.parked_for = None  # re-park after this
        self.log(f"PLANE x={x0} dir={d:+d}: pos {pos} ({BARREL[pos][0]} deg) spawn {window} meet {meet} (cur {cur})")

        def watch(idx):
            self.observe(idx)
            return bool(self.bombs)

        if cur != pos:
            r = goto(self.sct, self.mon, pos, on_frame=watch, corrections=0)
            if r != pos:
                self.plane_stats["aborted"] += 1
                self.log(f"  plane engagement aborted (turret at {r})")
                return True
        # one shot when the window absorbs +-1 tick of timing error
        mid = len(window) // 2
        fire = [window[mid]] if len(window) >= 3 else window[:2]
        fired = []
        while True:
            frame, idx = self.grab_observe()
            if is_done(frame) or self.bombs:
                break
            k = int(round((time.time() - t_obs) / TICK_S))
            near = [x for x, _ in self.planes_full if abs(x - (x0 + HELI_VX * d * k)) <= 12]
            if near:
                k = (near[0] - x0) // (HELI_VX * d)
            spawn = k + FIRE_LATENCY_TICKS
            if spawn > fire[-1]:
                break
            if spawn in fire and spawn not in fired:
                key("Up", ctx=dict(kind="plane", x0=x0, dir=d, pos=pos, spawn=spawn, t_obs=t_obs - self.t0))
                fired.append(spawn)
                if len(fired) == len(fire):
                    break
        self.plane_stats["shots"] += len(fired)
        if fired:
            self.plane_pending.append(dict(x0=x0, d=d, t_obs=t_obs, pos=pos, shots=len(fired),
                                           t_meet=t_obs + meet * TICK_S))
        return True

    # ---- paratroopers ----------------------------------------------------
    def trooper_step(self, idx):
        """Engage the most urgent reachable trooper: canopy if it has one
        (3x the target area; a chute kill can also crush a landed trooper),
        else the body. Fire, then leave it alone until the bullets' meeting
        tick says whether it was hit -- no follow-up spray."""
        now = time.time()
        cands = self.troopers.candidates()
        if not cands:
            return False
        cur = barrel_pos(idx)
        ranked = sorted(cands, key=lambda tr: urgency(tr, self.troopers.landed, now)[0])
        choice = None
        for tr in ranked[:4]:
            part = "canopy" if tr.state == "canopy" else "body"
            plan = plan_trooper(tr, part, cur, now)
            if plan:
                choice = (tr, part, plan)
                break
        if choice is None:
            return self.pre_aim(ranked, cur, now)
        self.pre_aimed = None
        tr, part, (pos, window, meet, clamp_it) = choice
        t_obs, y_obs, state = now, tr.y_at(now), tr.state
        over = urgency(tr, self.troopers.landed, now)[1]
        st = self.troopers.stats
        st["engaged"] += 1
        st["bonus_candidates"] += over and part == "canopy"
        self.log(f"TROOPER x={tr.x} y={tr.y} {state} -> {part}{' OVER LANDED' if over else ''}: "
                 f"pos {pos}{' CLAMPED' if clamp_it else ''} ({BARREL[pos][0]} deg) spawn {window} "
                 f"meet {meet} (cur {cur})")

        def watch(idx):
            self.observe(idx)
            return bool(self.bombs) or self.phase.phase == 2 or tr.state != state

        if cur != pos:
            r = goto(self.sct, self.mon, pos, on_frame=watch, corrections=0)
            if r != pos:
                return True
        if clamp_it:
            clamp(self.sct, self.mon, pos)
        # one bullet when the window is wide enough to absorb +-1 tick of
        # timing error, otherwise a pair
        mid = len(window) // 2
        fire = [window[mid]] if len(window) >= 4 else window[max(0, mid - 1):mid + 1]
        fired = []
        while tr.state == state and tr in self.troopers.troopers:
            frame, idx = self.grab_observe()
            if is_done(frame) or self.bombs or self.phase.phase == 2:
                break
            k = int(round((tr.y_at(time.time()) - y_obs) / tr.vy))
            spawn = k + FIRE_LATENCY_TICKS
            if spawn > fire[-1]:
                break
            if spawn in fire and spawn not in fired:
                self.evidence("trooper_shot", seconds=0.2, delay=1.6)
                key("Up", ctx=dict(kind="trooper", x=tr.x, y_obs=y_obs, state=state, part=part, pos=pos, clamp=clamp_it, spawn=spawn, meet=meet, t_obs=t_obs - self.t0))
                fired.append(spawn)
                if len(fired) == len(fire):
                    break
        st["shots"] += len(fired)
        if fired:
            tr.pending = dict(t_meet=t_obs + meet * TICK_S, part=part, pos=pos,
                              shots=len(fired), state=state)
        return True

    def pre_aim(self, ranked, cur, now):
        """Nothing shootable yet: turn toward where the most urgent
        free-faller's canopy will most likely open, so the canopy shot is
        a short move when it comes. Returns True if it moved."""
        for tr in ranked:
            if tr.state != "free":
                continue
            ghost = predicted_canopy_trooper(tr, now)
            plan = plan_trooper(ghost, "canopy", None, now)
            if not plan:
                continue
            pos = plan[0]
            if pos == cur or self.pre_aimed == (id(tr), pos):
                return False
            self.pre_aimed = (id(tr), pos)
            self.log(f"pre-aim for trooper x={tr.x} (canopy expected ~y{ghost.y}): pos {cur} -> {pos}")
            known = {(id(t), t.state) for t in self.troopers.candidates()}

            def watch(idx):
                self.observe(idx)
                fresh = {(id(t), t.state) for t in self.troopers.candidates()} - known
                return bool(self.bombs) or self.phase.phase == 2 or bool(fresh)
            goto(self.sct, self.mon, pos, on_frame=watch, corrections=0)
            return True
        return False

    # ---- phase 1 -------------------------------------------------------
    def paratrooper_step(self, frame):
        target = find_paratrooper_target(frame)
        if target is None:
            # nothing threatening: take a shot at a helicopter if one is
            # reachable (lowest priority -- aborts if a trooper shows up)
            if not self.heli_step(to_index(frame)):
                time.sleep(0.02)
            return
        barrel_angle = detect_barrel_angle(frame)
        if not valid_barrel_angle(barrel_angle):
            return
        solution = solve_intercept(barrel_angle, *target)
        if solution is None:
            return
        direction, hold_s = solution
        if direction != 0:
            do_action(1 if direction > 0 else 2)
            wait = max(0.0, hold_s - LATENCY_OFFSET_S + TICK_SAFETY_MARGIN_S)
            t_end = time.time() + wait
            while time.time() < t_end:  # keep watching the sky while rotating
                _, idx = self.grab_observe()
                if self.phase.phase == 2 or self.bombs:
                    key("Up")
                    return
        do_action(3)

    def step(self):
        frame, idx = self.grab_observe()
        if is_done(frame):
            return "done"
        unplanned = [b for b in self.bombs if b.plan is None]
        if unplanned:
            self.engage(max(unplanned, key=lambda b: b.k))  # closest to impact first
            return
        if self.bombs:
            return  # engaged bombs still in flight (or briefly occluded)
        if self.phase.phase == 2:
            if self.plane_step(idx):
                return
            if not self.bombs and not self.plane_pending:
                self.flush_evidence()
            # bombs are released just after a plane enters, so aim for the
            # side the most recent plane came in from
            if self.entry_side is not None:
                self.park(self.entry_side)
            return
        self.parked_for = None
        if self.trooper_step(idx) or self.heli_step(idx):
            return
        if self.phase.phase == "gap" and time.time() - self.t0 > 30:
            # nothing to shoot during a lull after the first 30s: planes may
            # be next (wave 1 opens at ~38.2s, from the left every time), so
            # park for them -- troopers above take precedence
            self.park(1)
            return
        self.flush_evidence()  # idle: nothing to shoot
        time.sleep(0.01)


def main(n_episodes=5, out_dir="/captures/bomb_defense", keep="all"):
    """keep='all' saves every episode's frame recording; keep='notable'
    only saves a new best score or an episode where a bomb landed (for
    long unattended runs -- each recording is ~3.5 MB)."""
    os.makedirs(out_dir, exist_ok=True)
    rec = Recorder()
    rec.start()
    summary = []
    with mss.mss() as sct:
        mon = sct.monitors[1]

        def log(msg):
            print(f"[{time.time() - player.t0:6.2f}] {msg}", flush=True)

        player = Player(sct, mon, log)
        player.rec = rec
        player.evidence_dir = out_dir
        player.evidence_left = dict(plane_kills=6, plane_misses=6, trooper_miss=12, bomb_landed=12,
                                    trooper_shot=int(os.environ.get("AUDIT_TROOPER_SHOTS", "0")))
        for ep in range(n_episodes):
            # leave the game-over screen: one space press doesn't always
            # take, and a stale game-over frame would log a phantom 0s game
            deadline = time.time() + 10
            while is_done(grab(sct, mon)) and time.time() < deadline:
                key("space")
                time.sleep(0.8)
            player.reset()
            log(f"=== episode {ep} ===")
            while player.step() != "done":
                pass
            score = read_score(grab(sct, mon))
            dur = time.time() - player.t0
            if dur < 5:
                log("stale game-over screen, not a real game -- skipping")
                continue
            log(f"GAME OVER score={score} dur={dur:.1f}s planes={player.plane_stats} troopers={player.troopers.stats} "
                f"helis={player.heli_stats} bombs={player.done_bombs}")
            time.sleep(0.3)
            best = max([e["score"] for e in summary], default=-1)
            landed = any(b["outcome"] == "LANDED" for b in player.done_bombs)
            path = None
            if keep == "all" or score > best or landed:
                frames = rec.snapshot()
                path = f"{out_dir}/ep_{int(player.t0)}.npz"
                np.savez_compressed(path, t=np.array([f[0] for f in frames]) - player.t0,
                                    frames=np.stack([f[1] for f in frames]))
            shots = [(t - player.t0, c) for t, c in turret.SHOTS if t >= player.t0]
            turret.SHOTS.clear()
            with open(f"{out_dir}/shots_{int(player.t0)}.json", "w") as fh:
                json.dump(shots, fh, default=int)
            summary.append(dict(ep=ep, score=score, dur=dur, bombs=player.done_bombs,
                                helis=player.heli_stats, troopers=player.troopers.stats,
                                planes=player.plane_stats, npz=path))
            scores = [e["score"] for e in summary]
            nb = sum(len(e["bombs"]) for e in summary)
            nk = sum(b["outcome"].startswith("GONE") for e in summary for b in e["bombs"])
            hk = sum(e["helis"]["kills"] for e in summary)
            he = sum(e["helis"]["kills"] + e["helis"]["misses"] for e in summary)
            log(f"STATS episodes={len(scores)} best={max(scores)} mean={np.mean(scores):.1f} "
                f"median={np.median(scores):.0f} bombs_killed={nk}/{nb} helis_killed={hk}/{he}")
            csv = f"{out_dir}/scores.csv"
            new_file = not os.path.exists(csv)
            with open(csv, "a") as fh:
                if new_file:
                    fh.write("finished_at,episode,score,duration_s,bombs_killed,bombs_seen,"
                             "trooper_kills,trooper_misses,heli_kills,heli_misses,plane_kills,plane_misses\n")
                ts = player.troopers.stats
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{ep},{score},{dur:.1f},"
                         f"{sum(b['outcome'].startswith('GONE') for b in player.done_bombs)},"
                         f"{len(player.done_bombs)},{ts['body_kills'] + ts['chute_kills']},"
                         f"{ts['misses']},{player.heli_stats['kills']},{player.heli_stats['misses']},"
                         f"{player.plane_stats['kills']},{player.plane_stats['misses']}\n")
            with open(f"{out_dir}/summary.json", "w") as fh:
                json.dump(summary, fh, indent=1, default=int)
    rec.stop_flag = True


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = sys.argv[2] if len(sys.argv) > 2 else "/captures/bomb_defense"
    keep = sys.argv[3] if len(sys.argv) > 3 else "all"
    main(n, out, keep)
