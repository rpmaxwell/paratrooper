"""Where does a bullet actually hit a helicopter? Frame-level ground truth.

Telemetry can't answer this: its helicopter and bullet positions are
reconstructed on tick labels that are off by +-1 tick often enough
(8 px of helicopter, 16 px of bullet) to blur any hitbox. Recorded
frames (game_*.npz, `paratrooper.run --record`) show the bullet and the
helicopter at the same instant, so:

  extract()  every bullet dot near helicopters in a frame, with every
             fully visible helicopter (skid x/y, direction from the next
             tick's frame) and whether the dot shows up on its lane on
             each of the next 5 ticks (tick of a later frame = how far the
             helicopters have moved, 4 native px per tick)
  observed() -> 'none' (flew on) | m (absorbed on tick m) | None (can't tell)
  predict()  first tick a rule says the bullet collides
  fit_box()  coordinate descent on the box edges

Everything here is in NATIVE pixels (320x200) relative to the
helicopter's skid top-left (geometry.py: game = 2 * native).

Finding (2026-10-06, 80 recorded games, ~28k bullet observations): the
game tests the bullet's drawn position against a SOLID box (the sprite's
hollow interior counts), on the helicopter's position after its move --
and the box sits 4 native px (8 game px) further right than
physics.model.heli_box, for both directions. 96.9% of outcomes exact;
the remaining misses are almost all bullets absorbed far from any
helicopter (troopers, bombs, planes).
"""
import glob
import pickle

import numpy as np

from ..physics import model as M

# native px, relative to the skid top-left: (c0, c1, r0, r1) inclusive
FITTED_BOX = {-1: (4, 27, -8, 1), 1: (-4, 19, -8, 1)}
# physics.model.heli_box in the same frame (game x0..x0+47 / x0-16..x0+31, y0-16..y0+3)
MODEL_BOX = {-1: (0, 23, -8, 1), 1: (-8, 15, -8, 1)}

NATIVE_LANES = [((sx // 2, (sy - 1) // 2), (vx // 2, vy // 2)) for (sx, sy), (vx, vy) in M.LANES]


def _lanes_of(p):
    out = []
    for li, ((sx, sy), (vx, vy)) in enumerate(NATIVE_LANES):
        j = (p[1] - sy) / vy
        if j == int(j) and j >= 0 and p[0] == sx + vx * int(j):
            out.append(li)
    return out


def extract(files, progress=False):
    """Recorded games -> list of observation dicts (see module doc)."""
    from ..perception import sprites as S   # scipy: only needed here
    rows = []
    for fi, f in enumerate(files):
        if progress:
            print(f"{fi + 1}/{len(files)} {f}", flush=True)
        F = np.load(f)["frames"]
        det = [S.detect(fr) for fr in F]
        H = [[(gx // 2, (gy - 1) // 2) for gx, gy in d.helis] for d in det]
        D = [set((gx // 2, (gy - 1) // 2) for gx, gy in d.dots) for d in det]
        for i in range(len(F) - 8):
            if not H[i]:
                continue
            j = next((jj for jj in (i + 1, i + 2) if H[jj] and H[jj] != H[i]), None)
            if j is None:
                continue
            helis = []
            for hx, hy in H[i]:
                m = [x for x, y in H[j] if y == hy and abs(x - hx) == 4]
                if len(m) != 1:
                    helis = None
                    break
                helis.append((hx, hy, 1 if m[0] > hx else -1))
            if not helis:
                continue
            h0x, h0y, h0d = helis[0]
            off = {}   # tick offset -> frames showing that tick
            for ff in range(i + 1, min(i + 9, len(F))):
                mm = [x for x, y in H[ff] if y == h0y and (x - h0x) * h0d > 0 and (x - h0x) % 4 == 0]
                if len(mm) == 1:
                    off.setdefault((mm[0] - h0x) // (4 * h0d), []).append(ff)
            lo_y, hi_y = min(hy for _, hy, _ in helis) - 12, max(hy for _, hy, _ in helis) + 14
            for b in D[i]:
                if b in D[j] or not lo_y <= b[1] <= hi_y:
                    continue
                ls = _lanes_of(b)
                if len(ls) != 1:
                    continue
                vx, vy = NATIVE_LANES[ls[0]][1]
                vis = []
                for m in range(1, 6):
                    p = (b[0] + m * vx, b[1] + m * vy)
                    if not (0 <= p[0] < 320 and 0 <= p[1] < 200):
                        vis.append(None)
                    elif m not in off:
                        vis.append("nf")
                    else:
                        vis.append(any(p in D[ff] for ff in off[m]))
                rows.append(dict(f=f, i=i, b=b, v=(vx, vy), lane=ls[0], helis=helis, vis=vis))
    return rows


def load_or_extract(cache, files, progress=False):
    try:
        return pickle.load(open(cache, "rb"))
    except (OSError, EOFError):
        rows = extract(files, progress)
        pickle.dump(rows, open(cache, "wb"))
        return rows


def observed(x):
    """'none' = flew on, m = absorbed on tick m (missing on two on-screen
    ticks running, never seen again), None = can't tell."""
    vis = x["vis"]
    last = max([m for m in range(1, 6) if vis[m - 1] is True], default=0)
    if last == 5 or vis[last] is None:
        return "none"
    if vis[last] is False and (last + 1 == 5 or vis[last + 1] is False):
        return last + 1
    return None


def predict(x, box=FITTED_BOX, heli="new", sweep=False):
    """First tick the bullet's drawn position (or, sweep=True, any point of
    its path that tick) is inside a helicopter's box; heli='new' = the
    helicopter after its move on that tick, 'old' = before."""
    vx, vy = x["v"]
    bx, by = x["b"]
    for m in range(1, 6):
        n = max(abs(vx), abs(vy)) if sweep else 1
        for f in range(1, n + 1):
            px = bx + (m - 1) * vx + round(vx * f / n)
            py = by + (m - 1) * vy + round(vy * f / n)
            for hx, hy, d in x["helis"]:
                c0, c1, r0, r1 = box[d]
                c = px - (hx + 4 * d * (m if heli == "new" else m - 1))
                if c0 <= c <= c1 and r0 <= py - hy <= r1:
                    return m
    return "none"


def score(rows, **kw):
    """-> Counter of exact / missed absorption / false hit / wrong tick."""
    import collections
    C = collections.Counter()
    for x, o in rows:
        p = predict(x, **kw)
        C["exact" if p == o else ("missed absorption" if p == "none" else
                                  ("false hit" if o == "none" else "wrong tick"))] += 1
    return C


def fit_box(rows, start=FITTED_BOX, rounds=3, deltas=(-3, -2, -1, 0, 1, 2, 3)):
    best = {d: list(v) for d, v in start.items()}

    def acc(B):
        C = score(rows, box={d: tuple(v) for d, v in B.items()})
        return C["exact"] / sum(C.values())
    for _ in range(rounds):
        for d in (-1, 1):
            for k in range(4):
                cands = []
                for dl in deltas:
                    B = {dd: list(v) for dd, v in best.items()}
                    B[d][k] += dl
                    cands.append((acc(B), dl))
                best[d][k] += max(cands)[1]
    return {d: tuple(v) for d, v in best.items()}, acc(best)


def sprite_masks(files, n_files=12):
    """Mean colour occupancy of the helicopter sprite per direction:
    [dir(-1, 1), rows -12..2, cols -14..29, colour 0..3] (native, skid top-left)."""
    from ..perception import sprites as S
    R0, R1, C0, C1 = -12, 3, -14, 30
    acc = {d: np.zeros((R1 - R0, C1 - C0, 4)) for d in (-1, 1)}
    n = {-1: 0, 1: 0}
    for f in files[:n_files]:
        F = np.load(f)["frames"]
        prev = None
        for i in range(len(F)):
            cur = S.detect(F[i]).helis
            if prev is not None and len(cur) == 1 and len(prev) == 1 and cur[0][1] == prev[0][1] \
                    and cur[0][0] - prev[0][0] in (-8, 8):
                d = 1 if cur[0][0] > prev[0][0] else -1
                nx, ny = cur[0][0] // 2, (cur[0][1] - 1) // 2
                if ny + R0 >= 0 and nx + C0 >= 0 and nx + C1 <= 320:
                    crop = F[i, ny + R0:ny + R1, nx + C0:nx + C1]
                    for c in range(4):
                        acc[d][:, :, c] += crop == c
                    n[d] += 1
            prev = cur
    return np.stack([acc[-1] / max(n[-1], 1), acc[1] / max(n[1], 1)]), (R0, C0)


def recorded_games(root):
    return sorted(glob.glob(f"{root}/captures/*/game_*.npz"))
