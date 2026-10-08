"""Expected future points: a common currency for decisions.

Every decision the policy makes (which trooper first, crush or kill, a
helicopter now or later, a bomb) changes two things: the points it scores
now, and the game state -- above all how many troopers have landed on each
side, because the fourth on one side ends the game (bombs end it too). This
module values states in points, so any decision can be priced as

    value = points now + E[V(state after)] - V(state now)

V(round, tick in round, L, R) = expected points from here to game over, for
L / R troopers landed left / right of the turret, under the CURRENT policy
(a policy-evaluation step: price one decision assuming the bot plays on as
today, then re-fit after the policy changes).

The model (fit() from telemetry, solve() by backward induction):
  * a game is a sequence of rounds: helicopter wave, then the bomber phase;
    round r runs from heli window r open to heli window r+1 open
    (data/sim_params.json). Rounds past the last one with enough data repeat
    it (the game's own schedule: rounds 6+ are like round 5).
  * per (round, phase) and tick alive: points scored (rho), landings per
    side (lam), crush removals per landed trooper (mu), death by bomb (beta).
  * a fourth trooper on one side ends the game after GRACE ticks of
    play (the assault takes a while; points still come in meanwhile).

Scoring (measured from score deltas): helicopter 10, plane 10, trooper 5
(a crush scores 5 + 5), bomb 30, every bullet -1.
"""
import collections
import json

import numpy as np
import pandas as pd

from . import params as SP

POINTS = dict(heli=10, plane=10, trooper=5, bomb=30, bullet=-1)
# what becomes of a dropped trooper under the current bot (A/B runs, 2026-10;
# landings from recorded frames): P(it lands), P(we kill it), per round
P_LAND_PER_DROP = {1: 0.125, 2: 0.165, 3: 0.124, 4: 0.185}
P_KILL_PER_DROP = {1: 0.854, 2: 0.807, 3: 0.847, 4: 0.803}
LOSE_AT = 4            # troopers on one side that storm the gun
STEP = 25              # ticks per DP step
GRACE = 367            # ticks of play between the fatal landing and game over (median, 108 recorded games)
MIN_EXPOSURE = 5000    # ticks alive in a round-phase needed to fit it on its own


# ---- fitting -------------------------------------------------------------------
def _phase(p, rnd, k):
    """'heli' or 'plane' for game tick k in round rnd (k may be past the windows)."""
    return "plane" if k >= SP.plane_window(p, rnd)[0] else "heli"


def round_of(p, k):
    r = 1
    while SP.heli_window(p, r + 1)[0] <= k:
        r += 1
    return r


def _side_timeline(g, recs, end):
    """[(tick, side, +1 landing / -1 crush)] sorted, and whether the game
    logged crushes at all (older telemetry did not)."""
    ev = [(t.last_tick, "L" if t.x < 320 else "R", +1) for t in g.troopers if t.fate == "landed"]
    crush_era = any('"over_landed"' in l for l in open(g.path) if '"bullet"' in l)
    for t in recs:
        if t.get("type") == "trooper" and t.get("crushed_landed_at") is not None and t.get("y_by_tick"):
            ev.append((t["y_by_tick"][-1][0], "L" if t["x"] < 320 else "R", -1))
    return sorted(ev), crush_era


def fit(games, p=None, max_round=5, max_n=LOSE_AT - 1):
    """Rates per (round, phase) from telemetry Games (sim.telemetry.load).
    Landings are per side and by how many troopers that side already has
    (the policy defends a crowded side harder); crushes per landed trooper
    and tick, from games that log them."""
    p = p or SP.load()
    expo = collections.Counter()          # (r, ph) ticks alive
    side_expo = collections.Counter()     # (r, ph, n) side-ticks with n landed on that side
    land = collections.Counter()          # (r, ph, n)
    crush, crush_expo = collections.Counter(), collections.Counter()   # (r, ph); per landed-trooper tick
    pts = collections.Counter()
    bomb_deaths = collections.Counter()
    games_in = collections.Counter()
    for g in games:
        recs = [json.loads(l) for l in open(g.path) if '"bomb"' in l or '"trooper"' in l]
        end = g.last_tick
        for r in range(1, round_of(p, end) + 1):
            o, po, nxt = SP.heli_window(p, r)[0], SP.plane_window(p, r)[0], SP.heli_window(p, r + 1)[0]
            rr = min(r, max_round)
            expo[(rr, "heli")] += max(0, min(end, po) - o)
            expo[(rr, "plane")] += max(0, min(end, nxt) - po)
            games_in[rr] += 1
        for k, d in g.scores:
            r = round_of(p, k)
            pts[(min(r, max_round), _phase(p, r, k))] += d
        ev, crush_era = _side_timeline(g, recs, end)
        n = {"L": 0, "R": 0}
        k_prev = 0
        for k_ev in [e[0] for e in ev] + [end]:
            # exposure from k_prev to k_ev at the current counts, in STEP slices
            for k in range(k_prev, k_ev, STEP):
                r = round_of(p, k)
                key = (min(r, max_round), _phase(p, r, k))
                dt = min(STEP, k_ev - k)
                for s in "LR":
                    side_expo[key + (min(n[s], max_n),)] += dt
                    if crush_era:
                        crush_expo[key] += dt * n[s]
            k_prev = max(k_prev, k_ev)
            # apply the events at this tick
            while ev and ev[0][0] == k_ev:
                k, s, d = ev.pop(0)
                r = round_of(p, k)
                key = (min(r, max_round), _phase(p, r, k))
                if d > 0:
                    land[key + (min(n[s], max_n),)] += 1
                    n[s] += 1
                elif n[s] > 0:
                    if crush_era:
                        crush[key] += 1
                    n[s] -= 1
        if any(r.get("type") == "bomb" and r.get("fate") == "landed" for r in recs):
            r = round_of(p, end)
            bomb_deaths[(min(r, max_round), _phase(p, r, end))] += 1
    rows = []
    for r in range(1, max_round + 1):
        for ph in ("heli", "plane"):
            e = expo[(r, ph)]
            row = dict(round=r, phase=ph, exposure=e, games=games_in[r],
                       rho=pts[(r, ph)] / e if e else np.nan,
                       mu=crush[(r, ph)] / crush_expo[(r, ph)] if crush_expo[(r, ph)] else 0.0,
                       beta=bomb_deaths[(r, ph)] / e if e else np.nan,
                       landings=sum(land[(r, ph, i)] for i in range(max_n + 1)), crushes=crush[(r, ph)],
                       bomb_deaths=bomb_deaths[(r, ph)])
            for i in range(max_n + 1):
                se = side_expo[(r, ph, i)]
                row[f"lam{i}"] = land[(r, ph, i)] / se if se > 2000 else np.nan
                row[f"n_land{i}"] = land[(r, ph, i)]
            rows.append(row)
    rates = pd.DataFrame(rows).set_index(["round", "phase"])
    lam = [f"lam{i}" for i in range(max_n + 1)]
    # a count with too little exposure: the next lower count's rate
    for i in range(1, max_n + 1):
        rates[lam[i]] = rates[lam[i]].fillna(rates[lam[i - 1]])
    # thin rounds borrow the previous round's rates (bomb deaths are kept: they are the round's own)
    rates["borrowed"] = np.nan
    for r in range(2, max_round + 1):
        for ph in ("heli", "plane"):
            if rates.loc[(r, ph), "exposure"] < MIN_EXPOSURE:
                for c in ["rho", "mu"] + lam:
                    rates.loc[(r, ph), c] = rates.loc[(r - 1, ph), c]
                if rates.loc[(r, ph), "bomb_deaths"] == 0:
                    rates.loc[(r, ph), "beta"] = rates.loc[(r - 1, ph), "beta"]
                rates.loc[(r, ph), "borrowed"] = r - 1
    return rates


def landed_timeline(rows, smooth=3):
    """Recorded frames -> (tick, L, R): landed heads per side, from
    perception (sprites.detect(...).landed) sampled every few ticks; a
    rolling median drops one-frame flicker. rows: [(tick, ((x, y), ...)), ...]
    as captures/derived/landed_frames.pkl stores them."""
    t = np.array([r[0] for r in rows])
    L = np.array([sum(1 for x, y in r[1] if x < 320) for r in rows])
    R = np.array([sum(1 for x, y in r[1] if x >= 320) for r in rows])

    def med(a):
        return np.array([int(np.median(a[max(0, i - smooth):i + smooth + 1])) for i in range(len(a))])
    return t, med(L), med(R)


def fit_landings_from_frames(timelines, rates, p=None, max_round=5, max_n=LOSE_AT - 1, by_count=False,
                             min_side_ticks=10000):
    """Replace the telemetry landing / removal rates with ones measured on
    recorded frames, which see the true number of troopers standing on each
    side (telemetry over-counts landings ~30% and misses most removals --
    a dead trooper falling on a stack clears the whole column). Up to the
    first time a side holds LOSE_AT (after that they march on the gun).
    by_count=False pools the landing rate over the side's current count: on
    120 recorded games it does not depend on it (0.00103, 0.00112, 0.00095,
    0.00098 per side-tick at 0..3 landed), so per-count rates only add noise.
    Round-phases seen for fewer than min_side_ticks borrow the previous round's."""
    p = p or SP.load()
    side_expo, ups, heads, removed = (collections.Counter() for _ in range(4))
    for rows in timelines.values():
        if len(rows) < 50:
            continue
        t, L, R = landed_timeline(rows)
        for i in range(1, len(t)):
            if max(L[i - 1], R[i - 1]) >= LOSE_AT:
                break
            k = t[i - 1]
            r = round_of(p, k)
            key = (min(r, max_round), _phase(p, r, k))
            dt = t[i] - t[i - 1]
            for a in (L, R):
                side_expo[key + (min(a[i - 1], max_n),)] += dt
                heads[key] += a[i - 1] * dt
                if a[i] > a[i - 1]:
                    ups[key + (min(a[i - 1], max_n),)] += a[i] - a[i - 1]
                elif a[i] < a[i - 1]:
                    removed[key] += a[i - 1] - a[i]
    rates = rates.copy()
    for (r, ph) in rates.index:
        tot_e = sum(side_expo[(r, ph, n)] for n in range(max_n + 1))
        tot_u = sum(ups[(r, ph, n)] for n in range(max_n + 1))
        pooled = tot_u / tot_e if tot_e >= min_side_ticks else np.nan
        for n in range(max_n + 1):
            e = side_expo[(r, ph, n)]
            rates.loc[(r, ph), f"lam{n}"] = (ups[(r, ph, n)] / e if by_count and e > 3000 else pooled)
        rates.loc[(r, ph), "mu"] = removed[(r, ph)] / heads[(r, ph)] if tot_e >= min_side_ticks and heads[(r, ph)] else np.nan
        rates.loc[(r, ph), "frame_side_ticks"] = tot_e
    # rounds / phases the frames barely saw: the previous round's
    for r in range(2, max_round + 1):
        for ph in ("heli", "plane"):
            for c in [f"lam{n}" for n in range(max_n + 1)] + ["mu"]:
                if np.isnan(rates.loc[(r, ph), c]):
                    rates.loc[(r, ph), c] = rates.loc[(r - 1, ph), c]
    for c in [f"lam{n}" for n in range(max_n + 1)] + ["mu"]:
        rates[c] = rates[c].fillna(0.0)
    return rates


# ---- solving ---------------------------------------------------------------------
class Values:
    """V[r][i, L, R]: expected future points at step i of round r."""

    def __init__(self, rates, p=None, extra_rounds=12, grace=GRACE, lose_at=LOSE_AT):
        self.rates, self.p, self.grace, self.lose_at = rates, p or SP.load(), grace, lose_at
        self.max_round = int(rates.index.get_level_values(0).max())
        self.V = {}
        n = lose_at
        last = self.max_round + extra_rounds
        V_next = np.zeros((n, n))          # beyond the horizon: nothing (negligible: survival^12)
        for r in range(last, 0, -1):
            V_next = self._solve_round(r, V_next)
        self.V0 = V_next

    def _rate(self, r, ph):
        return self.rates.loc[(min(r, self.max_round), ph)]

    def steps(self, r):
        o = SP.heli_window(self.p, r)[0]
        po = SP.plane_window(self.p, r)[0]
        nxt = SP.heli_window(self.p, r + 1)[0]
        ks = np.arange(o, nxt, STEP)
        return ks, np.where(ks >= po, "plane", "heli")

    def _solve_round(self, r, V_next):
        ks, phs = self.steps(r)
        n = self.lose_at
        V = np.zeros((len(ks) + 1, n, n))
        V[-1] = V_next
        for i in range(len(ks) - 1, -1, -1):
            a = self._rate(r, phs[i])
            rho, mu, beta = (float(a[c]) for c in ("rho", "mu", "beta"))
            lam = [float(a[f"lam{j}"]) for j in range(n)]
            nxt = V[i + 1]
            grace_pts = rho * self.grace
            for L in range(n):
                for R in range(n):
                    pL, pR = lam[L] * STEP, lam[R] * STEP
                    cL, cR = mu * L * STEP, mu * R * STEP
                    pb = beta * STEP
                    stay = 1 - pL - pR - cL - cR
                    v = stay * nxt[L, R]
                    v += pL * (nxt[L + 1, R] if L + 1 < n else grace_pts)
                    v += pR * (nxt[L, R + 1] if R + 1 < n else grace_pts)
                    v += cL * nxt[L - 1, R] if L else 0
                    v += cR * nxt[L, R - 1] if R else 0
                    V[i, L, R] = rho * STEP + (1 - pb) * v
        self.V[r] = V
        return V[0]

    # -- lookup ---------------------------------------------------------------------
    def at(self, k, L, R):
        """V at game tick k with L / R landed (L or R >= lose_at: grace only)."""
        r = round_of(self.p, k)
        if r not in self.V:
            r = max(self.V)
        ks, phs = self.steps(r)
        i = int(np.clip((k - ks[0]) // STEP, 0, len(ks) - 1))
        if L >= self.lose_at or R >= self.lose_at:
            return float(self._rate(r, phs[i])["rho"]) * self.grace
        return float(self.V[r][i, L, R])

    # -- prices (points) --------------------------------------------------------------
    def landing_cost(self, k, L, R, side):
        """Points lost when one more trooper lands on `side` ("L"/"R") now."""
        after = (L + 1, R) if side == "L" else (L, R + 1)
        return self.at(k, L, R) - self.at(k, *after)

    def crush_value(self, k, L, R, side):
        """Canopy kill over a landed trooper: both score, one fewer landed."""
        after = (max(L - 1, 0), R) if side == "L" else (L, max(R - 1, 0))
        return 2 * POINTS["trooper"] + self.at(k, *after) - self.at(k, L, R)

    def trooper_kill_value(self, k, L, R, side, p_land):
        """Killing a falling trooper that would otherwise land with p_land
        (its chance of landing if we leave it to the rest of the policy)."""
        return POINTS["trooper"] + p_land * self.landing_cost(k, L, R, side)

    def heli_kill_value(self, k, L, R, drops_left, p_land_per_drop=None, p_kill_per_drop=None):
        """Killing a helicopter: 10 now, and the troopers it would still drop
        -- drops_left = (expected left, expected right), see drops_left() --
        neither land nor score (a dropped trooper is worth +5 when we kill it)."""
        r = min(round_of(self.p, k), max(P_LAND_PER_DROP))
        pl = P_LAND_PER_DROP[r] if p_land_per_drop is None else p_land_per_drop
        pk = P_KILL_PER_DROP[r] if p_kill_per_drop is None else p_kill_per_drop
        v = POINTS["heli"]
        for side, n in zip("LR", drops_left):
            v += n * (pl * self.landing_cost(k, L, R, side) - pk * POINTS["trooper"])
        return v

    def bomb_kill_value(self, k, L, R, p_hits_turret):
        """Shooting a bomb that would otherwise end the game with p_hits_turret."""
        return POINTS["bomb"] + p_hits_turret * self.at(k, L, R)


def drops_left(x0, d, rnd, p=None):
    """Expected troopers a helicopter (skid x0 now, direction d) will still
    drop on (left, right) if left alone: P(drop) per column it has yet to
    pass (sim params; no bursts, columns assumed free)."""
    p = p or SP.load()
    D, H = p["drop"], p["heli"]
    pf = D["p_free_by_wave"][min(rnd, len(D["p_free_by_wave"])) - 1]
    x = x0 + H["drop_offset"][d]
    ahead = [c for c in D["columns"] if (c > x if d > 0 else c < x)]
    return (pf * sum(1 for c in ahead if c < 320), pf * sum(1 for c in ahead if c >= 320))


def forward(values, rounds=6):
    """Distribution over (L, R) and P(alive) at each round start, under the
    model -- to check it against how far real games get."""
    n = values.lose_at
    dist = np.zeros((n, n))
    dist[0, 0] = 1.0
    out, pts = [], 0.0
    for r in range(1, rounds + 1):
        out.append(dict(round=r, alive=dist.sum(), mean_L=(dist.sum(1) * np.arange(n)).sum() / max(dist.sum(), 1e-12),
                        points_so_far=pts))
        ks, phs = values.steps(r)
        for i in range(len(ks)):
            a = values._rate(r, phs[i])
            rho, mu, beta = (float(a[c]) for c in ("rho", "mu", "beta"))
            lam = [float(a[f"lam{j}"]) for j in range(n)]
            pts += rho * STEP * dist.sum()
            new = np.zeros_like(dist)
            dead = 0.0
            for L in range(n):
                for R in range(n):
                    m = dist[L, R] * (1 - beta * STEP)
                    pL, pR, cL, cR = lam[L] * STEP, lam[R] * STEP, mu * L * STEP, mu * R * STEP
                    new[L, R] += m * (1 - pL - pR - cL - cR)
                    if L + 1 < n:
                        new[L + 1, R] += m * pL
                    else:
                        dead += m * pL
                    if R + 1 < n:
                        new[L, R + 1] += m * pR
                    else:
                        dead += m * pR
                    if L:
                        new[L - 1, R] += m * cL
                    if R:
                        new[L, R - 1] += m * cR
            pts += dead * rho * values.grace
            dist = new
    return pd.DataFrame(out)


def landing_cost_table(values, ticks_into_round=(0, 500, 1000, 1500), rounds=(1, 2, 3, 4, 5)):
    """Points lost by one more landing on the left, per round / time / state."""
    rows = []
    for r in rounds:
        o = SP.heli_window(values.p, r)[0]
        for dt in ticks_into_round:
            for L in range(values.lose_at):
                for R in range(values.lose_at):
                    rows.append(dict(round=r, tick_in_round=dt, L=L, R=R,
                                     V=values.at(o + dt, L, R), cost_L=values.landing_cost(o + dt, L, R, "L")))
    return pd.DataFrame(rows)


def main(argv):
    """python3 -m paratrooper.sim.value [run_dir ...]: fit on the given runs
    (default: the current bot's A/B runs) + their recorded frames, print the
    tables, save data/value_rates.csv."""
    import glob
    import os
    import pathlib
    import pickle
    from . import telemetry
    root = pathlib.Path(__file__).resolve().parents[3]
    runs = argv or [str(root / "captures/ab_helibox_model"), str(root / "captures/instance-1/ab_helibox_fitted")]
    games = telemetry.load(runs)
    rates = fit(games)
    cache = root / "captures/derived/landed_frames.pkl"
    if cache.exists():
        tl = pickle.load(open(cache, "rb"))
        want = {os.path.abspath(p) for d in runs for p in glob.glob(os.path.join(d, "game_*.npz"))}
        rates = fit_landings_from_frames({f: v for f, v in tl.items() if os.path.abspath(f) in want}, rates)
    else:
        print(f"(no {cache}: landing rates from telemetry, which over-counts them)")
    out = pathlib.Path(__file__).resolve().parents[1] / "data" / "value_rates.csv"
    rates.to_csv(out)
    vals = Values(rates)
    pd.set_option("display.width", 200)
    print(rates[["exposure", "rho", "mu", "beta", "lam0", "bomb_deaths", "borrowed"]].round(5))
    print(f"\nV(start) = {vals.at(0, 0, 0):.0f}; observed mean score "
          f"{np.mean([sum(d for _, d in g.scores) for g in games]):.0f} over {len(games)} games")
    print(forward(vals).round(3))
    T = landing_cost_table(vals, ticks_into_round=(500,), rounds=(1, 2, 3, 4))
    print("\ncost of one more landing on the left, 500 ticks into the round (rows L, cols R):")
    for r, g in T.groupby("round"):
        print(f"round {r}:\n", g.pivot(index="L", columns="R", values="cost_L").round(0))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
