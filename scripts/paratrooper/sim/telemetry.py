"""Telemetry -> clean per-game event lists for fitting the game model.

Reads the game_*.jsonl files paratrooper.run writes and turns them into
plain records the fitters (sim/fit.py) and the analysis notebook use:

  Heli    : one helicopter -- lane, direction, ticks it was tracked, its
            skid x on any tick (x_at), fate, whether we shot at it
  Plane   : one bomber plane
  Trooper : one trooper -- drop column x, first tick / y, fate
  Drop    : a trooper matched to the helicopter that dropped it

All ticks are the telemetry's own game ticks (GameClock, 18.2065 Hz) and
all x / y are game coordinates (see geometry.py).

Telemetry quirks this module absorbs (each found in the 2026-10 data pass):
  * some files have a broken tick counter (ticks jump into the 10^4..10^5
    range mid-game) -- quality() rejects them;
  * aircraft records store only the LAST tracked x, so the skid x on any
    earlier tick is reconstructed from the 8 px/tick constant speed;
  * a trooper's x is a fixed offset from the dropping helicopter's skid x,
    but which of two offsets (8 px apart, i.e. one tick of heli motion)
    depends on the tick phase of the frames -- DROP_OFFSETS holds both.
"""
import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np

from ..physics import chute
from ..physics import model as M

# trooper x - helicopter skid x0 on the trooper's first tick, per direction:
# two values one tick (8 px) apart, see module docstring
DROP_OFFSETS = {-1: (16, 24), 1: (8, 16)}
# a freshly dropped trooper's body top is first seen 16 px below the skids
DROP_Y_BELOW_LANE = 16


@dataclass
class Heli:
    id: int
    lane: int
    dir: int
    first_tick: int
    last_tick: int
    last_x: int
    fate: str
    shot_at: bool

    def x_at(self, k):
        """Skid x0 on tick k (constant 8 px/tick)."""
        return self.last_x - M.HELI_VX * self.dir * (self.last_tick - k)

    @property
    def entry_x(self):
        return self.x_at(self.first_tick)

    @property
    def crossed(self):
        """Flew the whole screen (left it without being shot)."""
        return self.fate == "left_screen"


@dataclass
class Plane:
    id: int
    dir: int
    first_tick: int
    last_tick: int
    last_x: int
    fate: str


@dataclass
class Trooper:
    id: int
    x: int
    first_tick: int
    first_y: int
    last_tick: int
    fate: str
    side: str = None
    last_y: int = None
    open_tick: int = None      # tick the fall slowed 8 -> 4 px (chute.true_open_tick), if seen
    cf_land_tick: float = None  # when it WOULD have landed had nobody shot it (= last_tick if it landed)

    @property
    def seen_from_drop(self):
        """First seen right under the skids (not picked up mid-fall)."""
        return self.first_y - DROP_Y_BELOW_LANE in LANES_SEEN


LANES_SEEN = (17, 29, 41, 53, 65, 89)


@dataclass
class Drop:
    tick: int
    x: int
    heli: Heli
    trooper: Trooper


@dataclass
class Game:
    path: str
    run: str
    helis: list = field(default_factory=list)
    planes: list = field(default_factory=list)
    troopers: list = field(default_factory=list)
    phases: list = field(default_factory=list)   # (tick, old, new)
    scores: list = field(default_factory=list)   # (tick, delta)
    bombs: list = field(default_factory=list)    # raw bomb records
    last_tick: int = 0
    drops: list = field(default_factory=list)
    unmatched: list = field(default_factory=list)  # troopers with no / ambiguous heli
    problems: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.problems


def _load_one(path):
    g = Game(path=path, run=os.path.basename(os.path.dirname(path)))
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a file cut off mid-line by a killed run
        t = r.get("type")
        for key in ("tick", "last_tick"):
            if isinstance(r.get(key), int):
                g.last_tick = max(g.last_tick, r[key])
        if t == "aircraft":
            if r.get("dir") not in (-1, 1) or r.get("last_x") is None:
                continue  # direction never measured: a one-frame sighting
            if r["kind"] == "heli":
                g.helis.append(Heli(r["id"], r["lane"], r["dir"], r["first_tick"], r["last_tick"],
                                    r["last_x"], r["fate"], bool(r.get("shots"))))
            else:
                g.planes.append(Plane(r["id"], r["dir"], r["first_tick"], r["last_tick"], r["last_x"], r["fate"]))
        elif t == "trooper":
            yb = r.get("y_by_tick") or [[r["first_tick"], r["first_y"]]]
            tr = Trooper(r["id"], r["x"], r["first_tick"], r["first_y"], yb[-1][0],
                         r.get("fate"), r.get("side"), yb[-1][1], chute.true_open_tick(yb))
            tr.cf_land_tick = counterfactual_landing(tr)
            g.troopers.append(tr)
        elif t == "phase":
            g.phases.append((r["tick"], r["old"], r["new"]))
        elif t == "score":
            g.scores.append((r["tick"], r["delta"]))
        elif t == "bomb":
            g.bombs.append(r)
    g.helis.sort(key=lambda h: h.first_tick)
    g.planes.sort(key=lambda p: p.first_tick)
    g.troopers.sort(key=lambda t: t.first_tick)
    return g


def match_drops(g):
    """Pair each trooper with the helicopter in its lane that was over its
    column on its first tick. Fills g.drops / g.unmatched."""
    for tr in g.troopers:
        lane = tr.first_y - DROP_Y_BELOW_LANE
        cands = [h for h in g.helis
                 if h.lane == lane and h.first_tick - 1 <= tr.first_tick <= h.last_tick + 1
                 and tr.x - h.x_at(tr.first_tick) in DROP_OFFSETS[h.dir]]
        if len(cands) == 1:
            g.drops.append(Drop(tr.first_tick, tr.x, cands[0], tr))
        else:
            g.unmatched.append(tr)


def counterfactual_landing(tr):
    """Tick the trooper would have touched down had it not been shot:
    its real landing if it landed; else from its last seen state -- chute
    open: 4 px/tick to the ground; still free-falling: the expected
    landing over the chute-opening distribution (physics.chute)."""
    if tr.fate == "landed" or tr.last_y is None:
        return float(tr.last_tick)
    if tr.open_tick is not None and tr.open_tick <= tr.last_tick:
        return tr.last_tick + max(0, M.GROUND_BODY_Y - tr.last_y) / M.CANOPY_VY
    s_now = max(0, (tr.last_y - tr.first_y) // M.FREE_VY)
    p = chute.open_distribution(s_now, tr.last_y)
    k = np.arange(1, len(p) + 1)
    y_open = tr.last_y + M.FREE_VY * k
    land = k + np.maximum(0, M.GROUND_BODY_Y - y_open) / M.CANOPY_VY
    return tr.last_tick + float((p * land).sum() / p.sum())


# A real game never goes this long without a new aircraft (the longest
# quiet stretch, helicopters gone -> first bomber, is ~200 ticks).
MAX_AIRCRAFT_GAP = 700


def quality(g):
    """Reasons to distrust a game's tick numbering (empty = fine)."""
    p = []
    starts = sorted([h.first_tick for h in g.helis] + [pl.first_tick for pl in g.planes])
    if not starts:
        p.append("no aircraft")
        return p
    if starts[0] > 120:
        p.append(f"first aircraft at tick {starts[0]} (joined mid-game?)")
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    if gaps and max(gaps) > MAX_AIRCRAFT_GAP:
        p.append(f"{max(gaps)}-tick gap between aircraft (broken tick counter)")
    if g.last_tick > 60000:
        p.append(f"last tick {g.last_tick}")
    return p


def load(run_dirs=None, pattern="captures/*/", root=None, keep_bad=False):
    """All games under run_dirs (default: every captures/* dir with
    telemetry). -> list[Game], drops matched, bad games flagged (and
    dropped unless keep_bad)."""
    root = root or os.getcwd()
    if run_dirs is None:
        run_dirs = sorted(glob.glob(os.path.join(root, pattern)))
    games = []
    for d in run_dirs:
        for path in sorted(glob.glob(os.path.join(d, "game_*.jsonl"))):
            g = _load_one(path)
            g.problems = quality(g)
            match_drops(g)
            if g.ok or keep_bad:
                games.append(g)
    return games
