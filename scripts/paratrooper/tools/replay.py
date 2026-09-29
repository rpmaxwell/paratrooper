"""Replay recorded frames through the world model (no live game needed).
    python3 -m paratrooper.tools.replay <npz> [<npz> ...]
Prints record counts, trooper fates, and per-frame update timing.
"""
import sys
import time
from collections import Counter

import numpy as np

from ..world.world import World


def replay(path):
    d = np.load(path)
    t, F = d["t"], d["frames"]
    recs = []
    w = World(float(t[0]), recs.append)
    prev, times = None, []
    for i in range(len(t)):
        nat = F[i][1::2, ::2]
        if prev is not None and np.array_equal(nat, prev):
            continue
        prev = nat
        a = time.perf_counter()
        w.update(float(t[i]), nat)
        times.append(time.perf_counter() - a)
    w.close_all(float(t[-1]))
    return w, recs, np.array(times) * 1000


def main(paths):
    allrecs, alltimes = [], []
    for p in paths:
        w, recs, times = replay(p)
        allrecs += recs
        alltimes.append(times)
    times = np.concatenate(alltimes)
    print(f"{len(paths)} recordings, {len(times)} distinct frames: update median {np.median(times):.2f} ms, "
          f"p99 {np.percentile(times, 99):.2f} ms, max {times.max():.2f} ms")
    print("records:", dict(Counter(r["type"] for r in allrecs)))
    tr = [r for r in allrecs if r["type"] == "trooper"]
    print("trooper fates:", dict(Counter(r["fate"] for r in tr)))
    print("canopy opened:", sum(1 for r in tr if r.get("canopy_open")), "of", len(tr))
    print("aircraft fates:", dict(Counter((r["kind"], r["fate"]) for r in allrecs if r["type"] == "aircraft")))
    print("bomb fates:", dict(Counter(r["fate"] for r in allrecs if r["type"] == "bomb")))
    for r in tr[:4]:
        print("  e.g.", {k: r[k] for k in ("id", "x", "first_tick", "first_y", "canopy_open", "canopy_lost", "fate", "last_y")})


if __name__ == "__main__":
    main(sys.argv[1:])
