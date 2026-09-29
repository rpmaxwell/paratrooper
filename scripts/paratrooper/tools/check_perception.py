"""Phase 2 gate (perception): replay recorded frames through the old,
validated detectors (game-coordinate frames) and the new native
detectors; count disagreements per sprite type.
    python3 -m paratrooper.tools.check_perception [n_files] [stride]
"""
import glob
import sys
import types
from collections import Counter

import numpy as np

sys.modules.setdefault("mss", types.ModuleType("mss"))  # old modules import it
import bomb_model as old_bomb  # noqa: E402
import heli_model as old_heli  # noqa: E402
import trooper_model as old_trooper  # noqa: E402
from read_score import is_done as old_is_done, read_score as old_read_score  # noqa: E402

from ..perception import barrel, hud, sprites  # noqa: E402

PAL = np.array([[0, 0, 0], [85, 255, 255], [255, 85, 255], [255, 255, 255]], np.uint8)


def old_dots(crop):
    return sorted((int(xs.min()), int(ys.min())) for xs, ys in old_bomb.components(crop[:372] == 3)
                  if xs.size == 4 and xs.max() - xs.min() == 1 and ys.max() - ys.min() == 1)


def main(n_files=40, stride=11):
    files = sorted(glob.glob("../captures/*/ep_*.npz"))[:n_files]
    diff, total = Counter(), 0
    examples = {}
    for f in files:
        F = np.load(f)["frames"]
        for crop in F[::stride]:
            nat = crop[1::2, ::2]
            new = sprites.detect(nat)
            old_s = old_bomb.sky_sprites(crop)
            bodies, cans, landed = old_trooper.detect(crop)
            full = np.zeros((768, 1024, 3), np.uint8)
            full[200:600, 192:832] = PAL[crop]
            checks = {
                "planes": (sorted(old_s["planes"]), sorted(new.planes)),
                "heli_any": (bool(old_s["helis"]), new.heli_any),
                "helis_full": (sorted(old_heli.find_helis(crop)), sorted(new.helis)),
                "bombs": (sorted(old_s["bombs"]), sorted(new.bombs)),
                # old tracker (not old detect) excluded the turret zone; bodies
                # below y 359 are now reported separately as grounded
                "bodies": (sorted(b for b in bodies if not (280 <= b[0] <= 360 and b[1] > 251) and b[1] <= 359),
                           sorted(new.bodies)),
                "canopies": (sorted(cans), sorted(new.canopies)),
                "landed": (sorted(landed), sorted(x for x, _ in new.landed)),
                "dots": ([d for d in old_dots(crop) if d[1] < 372], sorted(new.dots)),
                "barrel": (old_bomb.barrel_pos(crop), barrel.barrel_pos(nat)),
                "score": (old_read_score(full), hud.read_score(nat)),
                "done": (old_is_done(full), hud.is_done(nat)),
            }
            total += 1
            for k, (a, b) in checks.items():
                if a != b:
                    diff[k] += 1
                    examples.setdefault(k, (f, a, b))
    print(f"{total} frames from {len(files)} recordings")
    for k in ("planes", "heli_any", "helis_full", "bombs", "bodies", "canopies", "landed",
              "dots", "barrel", "score", "done"):
        print(f"  {k:10s} disagreements: {diff[k]:5d} ({diff[k] / total * 100:.2f}%)")
    for k, ex in examples.items():
        print("  e.g.", k, ex[0][-30:], "old", str(ex[1])[:120], "new", str(ex[2])[:120])


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
