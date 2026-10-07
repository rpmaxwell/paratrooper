"""Fit the helicopter / drop model from telemetry -> data/sim_params.json.

    cd scripts && python3 -m paratrooper.sim.fit            # every captures/* run
    cd scripts && python3 -m paratrooper.sim.fit ../captures/ab_prob ...

What the 2026-10 pass found (233 clean games, ~10k helicopters, 8.5k
matched drops) -- the model this module encodes:

SCHEDULE. The game runs on a fixed clock: helicopters may spawn only in
fixed windows (wave 1 ~0..590, then ~1500 ticks every ~2452 ticks), and
bomber windows sit in between. Wave 1 flies lanes 17 (leftward) / 41
(rightward); later waves change lane pair (17/41, 29/65, 41/89) with
the directions swapped at random per game.

SPAWN. Inside a window each lane spawns on its own, at a rate set by
how many of ITS OWN helicopters hold a slot (lane_rate[n]; the other
lane's count doesn't matter). A helicopter keeps its slot release_shot
ticks after being shot (explosion) and release_exit ticks after leaving
the screen (sprite still sliding off). A lane can't spawn again within
min_lane_gap ticks (the helicopters would overlap). This replaced a
shared table of SLOTS slots (slots / p_spawn, still fitted: same totals,
but it misses the lane alternation). Spawns are NOT policy-independent:
killing helicopters faster brings more of them. Real wave totals are
still more regular than this model (dispersion ~0.74, unexplained).

DROPS. Troopers are dropped only at 22 fixed columns 24 px apart, none
over the base. A helicopter passing a column drops with a per-wave
probability (p_free_by_wave, same for every column, direction and lane)
raised by cluster_beta (log-odds) per drop anywhere in the last
CLUSTER_WINDOW ticks -- drops come in bursts, whether or not we shoot
the troopers -- unless a trooper is already falling in that column
(then almost never).

Estimators are written to be unbiased under OUR policy's censoring: we
shoot helicopters, so every rate is a hazard over the exposure that was
actually observed (ticks in a window with N alive; column passes made
while alive), never a per-helicopter average over survivors -- e.g.
helicopters we never shot at average 1.27 drops per crossing, ~2x the
true rate, because a drop is what makes the bot leave a helicopter alone.
"""
import collections
import datetime
import math
import sys

import numpy as np

from . import params as PR
from . import telemetry as T

DROP_COLUMNS_MIN_COUNT = 50     # a column must have this many drops to count
WINDOW_PCT = 2                  # window edges: robust min / max of spawn ticks
MIN_WAVE_GAMES = 10             # fit a wave's schedule / lanes only with this many games


# ---- waves ------------------------------------------------------------------------
def blocks(g):
    """Consecutive runs of the same aircraft kind -> [(kind, [aircraft])],
    in spawn order. Helicopter run n is wave n."""
    ev = sorted([(h.first_tick, 0, "heli", h) for h in g.helis] +
                [(p.first_tick, 1, "plane", p) for p in g.planes], key=lambda e: e[:2])
    out = []
    for _, _, kind, a in ev:
        if out and out[-1][0] == kind:
            out[-1][1].append(a)
        else:
            out.append((kind, [a]))
    return out


def waves(g):
    """-> [dict(wave, helis, planes, complete)] -- complete = the game went
    on past this wave's planes, so its window edges are fully observed."""
    out, bl = [], blocks(g)
    w = 0
    for i, (kind, b) in enumerate(bl):
        if kind == "heli":
            w += 1
            out.append(dict(wave=w, helis=b, planes=[], heli_complete=i + 1 < len(bl), plane_complete=False))
        elif out:
            out[-1]["planes"] = b
            out[-1]["plane_complete"] = i + 1 < len(bl)
    return out


def fit_schedule(games):
    st = collections.defaultdict(lambda: collections.defaultdict(list))
    for g in games:
        for w in waves(g):
            if w["heli_complete"]:
                st[w["wave"]]["h_open"].append(w["helis"][0].first_tick)
                st[w["wave"]]["h_close"].append(w["helis"][-1].first_tick)
            if w["plane_complete"] and w["planes"]:
                st[w["wave"]]["p_open"].append(w["planes"][0].first_tick)
                st[w["wave"]]["p_close"].append(w["planes"][-1].first_tick)
    def edges(kind):
        out, n = [], []
        for wave in sorted(st):
            s = st[wave]
            if len(s[kind + "_open"]) < MIN_WAVE_GAMES:
                break   # later waves: extrapolated by the period (params.heli_window)
            out.append([int(np.percentile(s[kind + "_open"], WINDOW_PCT)),
                        int(np.percentile(s[kind + "_close"], 100 - WINDOW_PCT))])
            n.append(len(s[kind + "_open"]))
        return out, n
    hw, hn = edges("h")
    pw, pn = edges("p")
    # wave 1 is shorter than the rest; the period is measured from wave 2 on
    period = float(np.mean(np.diff([o for o, _ in hw[1:]]))) if len(hw) >= 3 else None
    return dict(heli_windows=hw, plane_windows=pw, period=round(period, 1),
                heli_games_per_wave=hn, plane_games_per_wave=pn)


def fit_lanes(games, n_waves):
    c = collections.defaultdict(collections.Counter)
    for g in games:
        for w in waves(g):
            if w["wave"] <= n_waves and w["heli_complete"]:
                c[w["wave"]][tuple(sorted({(h.lane, h.dir) for h in w["helis"]}))] += 1
    out = []
    for wave in range(1, n_waves + 1):
        pairs = c[wave]
        top = pairs.most_common(1)[0][0]
        lanes_ = sorted({ln for ln, _ in top})
        swapped = tuple(sorted((ln, -d) for ln, d in top))
        frac = pairs[swapped] / max(1, pairs[top] + pairs[swapped])
        out.append(dict(pair=[list(x) for x in top], lanes=lanes_, random_dirs=bool(frac > 0.2),
                        swapped_fraction=round(frac, 3), games=sum(pairs.values()),
                        other_pairs=sum(v for k, v in pairs.items() if k not in (top, swapped))))
    return out


# ---- spawns -----------------------------------------------------------------------
def spawn_exposure(games, schedule):
    """One row per (game, tick) inside a fully observed helicopter window:
    how many helicopters were alive (tracked) and how many spawned."""
    rows = []
    for gi, g in enumerate(games):
        for w in waves(g):
            if not w["heli_complete"] or w["wave"] > len(schedule["heli_windows"]):
                continue
            o, c = schedule["heli_windows"][w["wave"] - 1]
            hs = w["helis"]
            starts = collections.Counter(h.first_tick for h in hs)
            lo, hi = max(o, hs[0].first_tick - 300), min(c, hs[-1].first_tick)
            for k in range(lo + 1, hi + 1):
                alive = sum(1 for h in hs if h.first_tick < k <= h.last_tick)
                rows.append((gi, w["wave"], k, alive, starts.get(k, 0)))
    return np.array(rows, dtype=np.int64).reshape(-1, 5)   # game, wave, tick, alive, spawned


def fit_spawn(games, schedule, slot_range=range(4, 11)):
    X = spawn_exposure(games, schedule)
    alive, spawned = X[:, 3], X[:, 4]
    S = int(spawned.sum())
    ll = {}
    for K in slot_range:
        free = np.clip(K - alive, 0, None)
        q = S / free.sum()
        rate = np.maximum(q * free, 1e-9)
        ll[K] = float((spawned * np.log(rate) - rate).sum())   # Poisson log-likelihood per tick
    K = max(ll, key=ll.get)
    q = S / np.clip(K - alive, 0, None).sum()
    by_alive = {}
    for n in range(0, K + 1):
        m = alive == n
        if m.sum():
            by_alive[n] = dict(ticks=int(m.sum()), spawns=int(spawned[m].sum()),
                               rate=float(spawned[m].mean()), per_free_slot=float(spawned[m].mean() / (K - n)) if n < K else None)
    gaps = []
    for g in games:
        for ln in {h.lane for h in g.helis}:
            ft = [h.first_tick for h in g.helis if h.lane == ln]
            gaps += [b - a for a, b in zip(ft, ft[1:])]
    gaps = np.array(gaps)
    min_gap = int(np.percentile(gaps[gaps >= 4], 0.5))   # below 4: duplicate tracks, not real spawns
    # canonical spawn x (first tracked skid x) per direction, and the drop
    # offset that goes with it (the two are read one tick apart together)
    pairs = collections.Counter((d.heli.dir, d.heli.entry_x, d.x - d.heli.x_at(d.tick))
                                for g in games for d in g.drops)
    entry, offset = {}, {}
    for dr in (-1, 1):
        (_, e, o), _ = max(((k, v) for k, v in pairs.items() if k[0] == dr), key=lambda kv: kv[1])
        entry[dr], offset[dr] = e, o
    return dict(slots=K, p_spawn=round(q, 6), min_lane_gap=min_gap, vx=8, entry_x=entry, drop_offset=offset,
                loglik_by_slots={k: round(v, 1) for k, v in ll.items()}, by_alive=by_alive,
                window_ticks=int(len(X)), spawns=S)


def lane_spawn_exposure(games, schedule, min_gap, release=(0, 0)):
    """One row per (game, lane, tick) inside a fully observed two-lane
    window, outside the same-lane min gap: (game, wave, tick, lane,
    own-lane alive, other-lane alive, spawned in this lane).

    release = (shot, exit): ticks a helicopter keeps holding its slot after
    its last tracked tick -- shot down (explosion) / flown off the edge
    (the tracker stops at x = 0 / 608, the sprite is still sliding off)."""
    def held(h, k):
        extra = release[0] if h.fate == "shot_down" else release[1]
        return h.first_tick < k <= h.last_tick + extra
    rows = []
    for gi, g in enumerate(games):
        for w in waves(g):
            if not w["heli_complete"] or w["wave"] > len(schedule["heli_windows"]):
                continue
            hs = w["helis"]
            lanes = sorted({h.lane for h in hs})
            if len(lanes) != 2:
                continue
            o, c = schedule["heli_windows"][w["wave"] - 1]
            starts = {ln: {h.first_tick for h in hs if h.lane == ln} for ln in lanes}
            last = {ln: None for ln in lanes}
            for k in range(max(o, hs[0].first_tick - 300) + 1, min(c, hs[-1].first_tick) + 1):
                n = {ln: sum(1 for h in hs if h.lane == ln and held(h, k)) for ln in lanes}
                for i, ln in enumerate(lanes):
                    if last[ln] is None or k - last[ln] >= min_gap:
                        rows.append((gi, w["wave"], k, ln, n[ln], n[lanes[1 - i]], int(k in starts[ln])))
                for ln in lanes:
                    if k in starts[ln]:
                        last[ln] = k
    return np.array(rows, dtype=np.int64).reshape(-1, 7)


def _lane_fit(X, max_alive):
    own = np.minimum(X[:, 4], max_alive)
    rate = [float(X[own == n, 6].mean()) for n in range(max_alive + 1)]
    ll = float(sum((X[own == n, 6] * np.log(r) + (1 - X[own == n, 6]) * np.log(1 - r)).sum()
                   for n, r in enumerate(rate)))
    return rate, ll, [int((own == n).sum()) for n in range(max_alive + 1)]


def fit_lane_spawn(games, schedule, min_gap, max_alive=3,
                   shot_delays=range(0, 7), exit_delays=range(0, 25, 2)):
    """Per-lane spawn hazard by the number of that lane's own helicopters
    holding a slot (n >= max_alive pooled), plus how long a helicopter
    keeps its slot after it was last tracked -- both by likelihood. The
    lanes are independent: the other lane's count barely moves the rate
    (notebook, section 3)."""
    grid = {}
    for rs in shot_delays:          # coordinate search: shot delay, then exit delay
        grid[(rs, 0)] = _lane_fit(lane_spawn_exposure(games, schedule, min_gap, (rs, 0)), max_alive)
    rs = max((k for k in grid if k[1] == 0), key=lambda k: grid[k][1])[0]
    for re_ in exit_delays:
        if (rs, re_) not in grid:
            grid[(rs, re_)] = _lane_fit(lane_spawn_exposure(games, schedule, min_gap, (rs, re_)), max_alive)
    best = max(grid, key=lambda k: grid[k][1])
    rate, ll, ticks = grid[best]
    return dict(lane_rate=[round(r, 6) for r in rate], lane_ticks=ticks, lane_loglik=round(ll, 1),
                release_shot=best[0], release_exit=best[1],
                release_loglik={f"{a},{b}": round(v[1], 1) for (a, b), v in sorted(grid.items())})


# ---- drops -------------------------------------------------------------------------
def fit_drop_columns(games):
    c = collections.Counter(d.x for g in games for d in g.drops)
    return sorted(x for x, n in c.items() if n >= DROP_COLUMNS_MIN_COUNT)


def drop_exposure(games, columns, entry, offset):
    """One row per column a helicopter passed while alive: (game, heli id,
    dir, lane, column, tick, a trooper already falling in that column,
    troopers falling anywhere, dropped, troopers that WOULD be falling
    anywhere had none been shot -- telemetry.counterfactual_landing)."""
    rows = []
    for gi, g in enumerate(games):
        dropped = {(d.heli.id, d.x) for d in g.drops}
        falling = [(t.x, t.first_tick, t.last_tick) for t in g.troopers]
        falling_cf = [(t.first_tick, t.cf_land_tick) for t in g.troopers]
        for h in g.helis:
            off = offset[h.dir]
            last = h.last_tick + 1 if h.crossed else h.last_tick - 1   # a dying heli can't drop on its last tick
            for x in columns:
                # tick the skids are over column x (exact: everything is on the 8 px grid)
                k = h.last_tick - (h.last_x + off - x) // (T.M.HELI_VX * h.dir)
                if not h.first_tick - 1 <= k <= last:
                    continue
                # "already falling": dropped >2 ticks earlier (the +-1 tick
                # read ambiguity must not let a trooper block its own drop)
                col = any(tx == x and t0 < k - 2 and k <= t1 for tx, t0, t1 in falling)
                n_air = sum(1 for _, t0, t1 in falling if t0 < k - 2 and k <= t1)
                n_cf = sum(1 for t0, t1 in falling_cf if t0 < k - 2 and k <= t1)
                rows.append((gi, h.id, h.dir, h.lane, x, k, col, n_air, (h.id, x) in dropped, n_cf))
    return np.array(rows, dtype=np.int64).reshape(-1, 10)


CLUSTER_WINDOW = (3, 50)   # recent drops: dropped this many ticks ago, [lo, hi)
CLUSTER_CAP = 8


def recent_drops(games, X, window=CLUSTER_WINDOW):
    """Per drop_exposure row: drops anywhere on screen (shot or not) in the
    window before the pass -- what the drop probability responds to."""
    lo, hi = window
    ticks = {gi: np.array(sorted(t.first_tick for t in g.troopers)) for gi, g in enumerate(games)}
    out = np.zeros(len(X), np.int64)
    for i, (gi, k) in enumerate(zip(X[:, 0], X[:, 5])):
        d = ticks[gi]
        out[i] = np.searchsorted(d, k - lo) - np.searchsorted(d, k - hi)
    return out


def _logistic(Z, y, iters=30):
    """Newton-Raphson logistic regression -> (coef, se, loglik)."""
    b = np.zeros(Z.shape[1])
    b[: Z.shape[1]] = 0
    b[0] = -3.4
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ b)))
        H = Z.T @ (Z * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, Z.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-9:
            break
    p = 1 / (1 + np.exp(-(Z @ b)))
    ll = float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
    return b, np.sqrt(np.diag(np.linalg.inv(H))), ll


def fit_drops(games, columns, entry, offset, wave_ends):
    """P(drop) per column pass. Unblocked passes: logit p = per-wave level
    + cluster_beta * (drops anywhere in CLUSTER_WINDOW ticks before, capped)
    -- drops come in bursts, and what drives it is how many drops happened
    recently, NOT how many troopers are still on screen (our shooting
    removes those and the rate doesn't care; notebook section 4b)."""
    X = drop_exposure(games, columns, entry, offset)
    blocked, d = X[:, 6].astype(bool), X[:, 8]
    free = X[~blocked]
    y = free[:, 8].astype(float)
    n_waves = len(wave_ends)
    wave = np.minimum(np.searchsorted(wave_ends, free[:, 5]) + 1, n_waves)
    W = np.stack([(wave == w).astype(float) for w in range(1, n_waves + 1)], 1)
    r = np.minimum(recent_drops(games, free), CLUSTER_CAP).astype(float)
    b, se, ll = _logistic(np.c_[W, r], y)
    _, _, ll0 = _logistic(W, y)
    p0 = [round(float(1 / (1 + np.exp(-v))), 5) for v in b[:n_waves]]
    return dict(columns=columns, p_free=round(float(d[~blocked].mean()), 5),
                p_blocked=round(float(d[blocked].mean()), 5),
                p_free_by_wave=p0, cluster_window=list(CLUSTER_WINDOW), cluster_cap=CLUSTER_CAP,
                cluster_beta=round(float(b[-1]), 4), cluster_beta_se=round(float(se[-1]), 4),
                cluster_loglik_gain=round(ll - ll0, 1),
                passes_free=int((~blocked).sum()), passes_blocked=int(blocked.sum()),
                drops=int(d.sum()))


# ---- all ---------------------------------------------------------------------------
def fit_all(games):
    sched = fit_schedule(games)
    lanes_ = fit_lanes(games, len(sched["heli_windows"]))
    heli = fit_spawn(games, sched)
    heli.update(fit_lane_spawn(games, sched, heli["min_lane_gap"]), spawn_model="per_lane")
    cols = fit_drop_columns(games)
    drop = fit_drops(games, cols, heli["entry_x"], heli["drop_offset"], [c for _, c in sched["heli_windows"]])
    meta = dict(fitted=datetime.date.today().isoformat(), games=len(games),
                runs=sorted({g.run for g in games}), drops_matched=sum(len(g.drops) for g in games),
                troopers_unmatched=sum(len(g.unmatched) for g in games),
                helis=sum(len(g.helis) for g in games))
    return dict(meta=meta, schedule=sched, lanes=lanes_, heli=heli, drop=drop)


def main(argv):
    games = T.load(argv or None, root="..")
    p = fit_all(games)
    path = PR.save(p)
    h, d, s = p["heli"], p["drop"], p["schedule"]
    print(f"{p['meta']['games']} games, {p['meta']['helis']} helicopters, {d['drops']} drops -> {path}")
    print(f"schedule: heli windows {s['heli_windows']}, period {s['period']}")
    print(f"spawn: {h['slots']} slots x {h['p_spawn']:.4f}/tick, same-lane gap >= {h['min_lane_gap']} ticks")
    print(f"drops: {len(d['columns'])} columns, p_free {d['p_free']:.4f}, p_blocked {d['p_blocked']:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
