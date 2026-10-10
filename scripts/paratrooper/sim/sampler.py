"""Generate helicopter / drop event streams from the fitted model.

sample_game() plays the spawn + drop processes of params.json forward
tick by tick and returns a telemetry.Game -- the same objects
telemetry.load() returns for real games, so every fitter and every
notebook plot runs unchanged on simulated games. Two uses:

  * parameter recovery: fit.fit_all(simulated games) must give back the
    parameters they were generated with (catches estimator bias);
  * fidelity: simulated vs real distributions (spawns per wave, drops per
    column, ...) under a kill model that mimics the bot.

This is only the spawn / drop layer of the simulator: no gun, no
bullets, no bombers (planes appear as placeholders at their window
edges so the wave structure is recoverable). Troopers fall by the
measured two-phase model (physics.chute) and land; they matter here only
because a falling trooper blocks its column for further drops.

Kill hooks stand in for a policy until the gun exists:
  heli_kill(rng, heli_dict)    -> tick the helicopter is shot down, or None
  trooper_kill(rng, trooper)   -> tick the trooper is shot, or None

A free slot that picks a lane still inside min_lane_gap spawns in the
other lane instead. Inferred, not observed directly: losing the spawn
instead makes the per-free-slot rate fall off at 4+ alive, which the
real data doesn't show (notebook, section 3).
"""
import numpy as np

from ..physics import chute
from ..physics import model as M
from . import params as PR
from .telemetry import DROP_Y_BELOW_LANE, Game, Heli, Plane, Trooper

X_MIN, X_MAX = 0, 608     # skid x range over which a helicopter is tracked


def _fall_ticks(rng, first_y):
    """Ticks from the drop until the trooper lands (two-phase fall, chute
    opening sampled from physics.chute)."""
    p = chute.open_distribution(0, first_y)
    k_open = 1 + int(rng.choice(len(p), p=p / p.sum()))
    y_open = first_y + M.FREE_VY * k_open
    return k_open + int(np.ceil(max(0, M.GROUND_BODY_Y - y_open) / M.CANOPY_VY))


def sample_game(p, rng, horizon=None, heli_kill=None, trooper_kill=None, n_waves=None):
    """One game of helicopter spawns and drops -> telemetry.Game."""
    H, D = p["heli"], p["drop"]
    n_waves = n_waves or len(p["schedule"]["heli_windows"])
    horizon = horizon or PR.plane_window(p, n_waves)[1] + 100
    windows = []
    for w in range(1, n_waves + 1):
        pairs, random_dirs = PR.lanes(p, w)
        if random_dirs and rng.random() < 0.5:
            pairs = [(ln, -d) for ln, d in pairs]
        windows.append((PR.heli_window(p, w), pairs))
    columns = set(D["columns"])
    g = Game(path="<sim>", run="sim")
    alive, falling = [], []          # heli dicts; (x, first_tick, last_tick)
    held = []                        # (lane, tick its slot frees): gone, slot not yet released
    drop_ticks = []
    wave_ends = [PR.heli_window(p, w)[1] for w in range(1, n_waves + 1)]

    def wave_of(k):
        return next((w for w, c in enumerate(wave_ends, 1) if k <= c + 100), n_waves)
    lane_last = {}
    next_id = 1
    for k in range(horizon):
        # leave / die
        for h in alive:
            x = h["entry"] + H["vx"] * h["dir"] * (k - h["spawn"])
            if h["kill"] is not None and k >= h["kill"]:
                h["fate"], h["last"] = "shot_down", h["kill"]
            elif not X_MIN <= x <= X_MAX:
                h["fate"], h["last"] = "left_screen", k - 1
        for h in [h for h in alive if h.get("fate")]:
            alive.remove(h)
            held.append((h["lane"], h["last"] + 1 + H.get("release_shot" if h["fate"] == "shot_down" else "release_exit", 0)))
            g.helis.append(Heli(h["id"], h["lane"], h["dir"], h["spawn"], h["last"],
                                h["entry"] + H["vx"] * h["dir"] * (h["last"] - h["spawn"]), h["fate"],
                                h["kill"] is not None))
        falling = [f for f in falling if f[2] >= k]
        held = [(ln, t) for ln, t in held if t > k]
        # spawn: each free slot, inside a window
        win = next((pairs for (o, c), pairs in windows if o <= k <= c), None)
        if win is not None and H.get("spawn_model") == "per_lane":
            # each lane spawns on its own, at a rate set by how many of its
            # own helicopters are alive (fit.fit_lane_spawn)
            for lane, d in win:
                n_own = sum(1 for h in alive if h["lane"] == lane) + sum(1 for ln, _ in held if ln == lane)
                rate = H["lane_rate"][min(n_own, len(H["lane_rate"]) - 1)]
                if k - lane_last.get(lane, -10 ** 9) < H["min_lane_gap"] or rng.random() >= rate:
                    continue
                lane_last[lane] = k
                h = dict(id=next_id, lane=lane, dir=d, spawn=k, entry=H["entry_x"][d], kill=None)
                next_id += 1
                if heli_kill is not None:
                    h["kill"] = heli_kill(rng, h)
                alive.append(h)
        elif win is not None:
            for _ in range(H["slots"] - len(alive)):
                if rng.random() >= H["p_spawn"]:
                    continue
                # random lane; one still inside min_lane_gap hands the spawn
                # to the other lane (dropping it instead under-spawns when
                # many are alive -- the real per-slot rate stays flat)
                order = rng.permutation(len(win))
                free = [win[i] for i in order if k - lane_last.get(win[i][0], -10 ** 9) >= H["min_lane_gap"]]
                if not free:
                    continue
                lane, d = free[0]
                lane_last[lane] = k
                h = dict(id=next_id, lane=lane, dir=d, spawn=k, entry=H["entry_x"][d], kill=None)
                next_id += 1
                if heli_kill is not None:
                    h["kill"] = heli_kill(rng, h)
                alive.append(h)
        # drops: helicopters over a column
        for h in alive:
            col = h["entry"] + H["vx"] * h["dir"] * (k - h["spawn"]) + H["drop_offset"][h["dir"]]
            if col not in columns:
                continue
            blocked = any(fx == col for fx, _, _ in falling)
            if blocked:
                p_drop = D["p_blocked"]
            elif "cluster_beta" in D:
                # bursts: each drop (shot or not) in the window raises the odds
                lo, hi = D["cluster_window"]
                n_recent = min(sum(1 for t in drop_ticks if lo <= k - t < hi), D["cluster_cap"])
                base = D["p_free_by_wave"][min(wave_of(k), len(D["p_free_by_wave"])) - 1]
                p_drop = 1 / (1 + np.exp(-(np.log(base / (1 - base)) + D["cluster_beta"] * n_recent)))
            else:
                p_drop = D["p_free"]
            if rng.random() < p_drop:
                drop_ticks.append(k)
                y0 = h["lane"] + DROP_Y_BELOW_LANE
                land = k + _fall_ticks(rng, y0)
                tr = Trooper(next_id, col, k, y0, land, "landed", cf_land_tick=float(land))
                next_id += 1
                if trooper_kill is not None:
                    kt = trooper_kill(rng, tr)
                    if kt is not None and kt < land:
                        tr.last_tick, tr.fate = kt, "killed"
                g.troopers.append(tr)
                falling.append((col, k, tr.last_tick))
    for h in alive:   # still flying at the horizon
        g.helis.append(Heli(h["id"], h["lane"], h["dir"], h["spawn"], horizon - 1,
                            h["entry"] + H["vx"] * h["dir"] * (horizon - 1 - h["spawn"]), "horizon",
                            h["kill"] is not None))
    # placeholder bombers at each window's edges (keeps waves() splitting right)
    for w in range(1, n_waves + 1):
        o, c = PR.plane_window(p, w)
        for t in (o, c):
            if t < horizon:
                g.planes.append(Plane(next_id, 1, t, t + 70, 592, "left_screen"))
                next_id += 1
    g.helis.sort(key=lambda h: h.first_tick)
    g.planes.sort(key=lambda pl: pl.first_tick)
    g.troopers.sort(key=lambda t: t.first_tick)
    g.last_tick = horizon - 1
    from .telemetry import match_drops
    match_drops(g)
    return g


# ---- kill models that mimic a policy -------------------------------------------------
def empirical_heli_kill(games):
    """Kill hook resampling the bot's real behaviour: per wave, the
    fraction of helicopters shot down and their tracked lifetimes."""
    from .fit import waves
    by_wave = {}
    for g in games:
        for w in waves(g):
            by_wave.setdefault(w["wave"], []).extend(
                (h.last_tick - h.first_tick) if h.fate == "shot_down" else None for h in w["helis"])
    p = PR.load()

    def wave_of(k):
        for w in range(1, 20):
            o, c = PR.heli_window(p, w)
            if k <= c + 100:
                return w
        return 1

    def kill(rng, h):
        pool = by_wave.get(wave_of(h["spawn"])) or by_wave[max(by_wave)]
        life = pool[rng.integers(len(pool))]
        return None if life is None else h["spawn"] + life
    return kill
