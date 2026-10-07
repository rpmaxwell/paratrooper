"""The fitted game model's numbers: one JSON file, data/sim_params.json.

Every number the simulator uses that is NOT a measured constant in
physics/model.py lives here, written by `python3 -m paratrooper.sim.fit`
together with what it was fitted on (meta). Refit after new live runs;
diff the file to see what moved.

Layout (see fit.py for how each is estimated):
  schedule.heli_windows   [[open, close], ...] tick windows in which
                          helicopters may spawn, per wave (1-based list)
  schedule.plane_windows  same for bomber planes
  schedule.period         ticks between consecutive helicopter waves
                          (extrapolates windows past the fitted ones)
  lanes                   per wave: [[lane_y, dir], [lane_y, dir]] and
                          whether the direction pairing is random
  heli.spawn_model        "per_lane" (sampler default) or "shared"
  heli.lane_rate          per lane, per tick: P(spawn) by how many of the
                          lane's own helicopters hold a slot (0, 1, 2, 3+)
  heli.release_shot/_exit ticks a helicopter keeps its slot after it was
                          last tracked: shot down / flown off the screen
  heli.slots, p_spawn     the "shared" alternative: one table of slots, each
                          free slot spawning with p_spawn per tick
  heli.min_lane_gap       ticks between spawns in the same lane
  heli.entry_x / drop_offset
                          canonical skid x on the spawn tick, and trooper
                          x - skid x on the drop tick, per direction
  drop.columns            the only x a trooper is ever dropped at
  drop.p_free_by_wave     P(drop) when a helicopter passes a column with
                          no trooper falling in it and no recent drops
  drop.cluster_beta       log-odds added per drop anywhere in the last
                          cluster_window ticks (capped at cluster_cap)
  drop.p_free             the plain average (no bursts)
  drop.p_blocked          ... with a trooper already falling in it
"""
import json
import pathlib

PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "sim_params.json"


def load(path=PATH):
    p = json.loads(pathlib.Path(path).read_text())
    for k in ("entry_x", "drop_offset"):   # JSON keys are strings
        p["heli"][k] = {int(d): v for d, v in p["heli"][k].items()}
    for k in ("loglik_by_slots", "by_alive"):
        p["heli"][k] = {int(n): v for n, v in p["heli"][k].items()}
    return p


def save(p, path=PATH):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(p, indent=1) + "\n")
    return path


def heli_window(p, wave):
    """[open, close] of helicopter wave `wave` (1-based), extrapolated by
    the period past the fitted ones."""
    w = p["schedule"]["heli_windows"]
    if wave <= len(w):
        return w[wave - 1]
    o, c = w[-1]
    shift = round(p["schedule"]["period"] * (wave - len(w)))
    return [o + shift, c + shift]


def plane_window(p, wave):
    """Bomber window that follows helicopter wave `wave`."""
    w = p["schedule"]["plane_windows"]
    if wave <= len(w):
        return w[wave - 1]
    o, c = w[-1]
    shift = round(p["schedule"]["period"] * (wave - len(w)))
    return [o + shift, c + shift]


def lanes(p, wave):
    """-> (pairs, random_dirs): the wave's two (lane, dir) and whether
    the directions are swapped at random per game."""
    L = p["lanes"]
    e = L[min(wave, len(L)) - 1]
    return [tuple(x) for x in e["pair"]], e["random_dirs"]
