"""Render one trooper's fall from a recorded game (needs --record frames),
with every bullet we fired at it, as a contact sheet PNG.

    python3 -m paratrooper.tools.show_trooper <game_T.jsonl> <trooper_id> [out.png]

Use it to check a verdict from audit_misses by eye.
"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

PAL = np.array([[0, 0, 0], [85, 255, 255], [255, 85, 255], [255, 255, 255]], np.uint8)
TICK_S = 1 / 18.2065


def main(jsonl, tid, out=None):
    R = [json.loads(l) for l in open(jsonl)]
    tr = next(r for r in R if r["type"] == "trooper" and r["id"] == int(tid))
    d = np.load(jsonl.replace(".jsonl", ".npz"))
    t, F = d["t"], d["frames"]
    t0, t1 = tr["y_by_tick"][0][0] * TICK_S - 0.2, tr["y_by_tick"][-1][0] * TICK_S + 0.4
    idx = [i for i in range(len(t)) if t0 <= t[i] <= t1]
    idx = idx[::max(1, len(idx) // 24)][:24]
    nx = tr["x"] // 2
    x0, x1 = max(0, nx - 40), min(320, nx + 44)
    shots = [r for r in R if r["type"] == "press" and (r["ctx"] or {}).get("target") == tr["id"]]
    tiles = []
    for i in idx:
        im = Image.fromarray(PAL[F[i][:, x0:x1]]).resize(((x1 - x0) * 3, 600), Image.NEAREST)
        tiles.append((im, t[i]))
    W, H = tiles[0][0].size
    sheet = Image.new("RGB", (8 * (W + 4), 3 * (H + 14)), (50, 50, 50))
    dr = ImageDraw.Draw(sheet)
    for k, (im, tt) in enumerate(tiles):
        x, y = (k % 8) * (W + 4), (k // 8) * (H + 14)
        sheet.paste(im, (x, y + 14))
        dr.text((x + 2, y), f"t={tt:.2f}s tick={int(tt / TICK_S)}", fill=(255, 255, 0))
    out = out or f"trooper_{tid}.png"
    sheet.save(out)
    print(f"trooper {tid}: x={tr['x']} fate={tr['fate']} canopy_open={tr.get('canopy_open')} "
          f"missed_reason={tr.get('missed_reason')} shots aimed at it: {[s['shot'] for s in shots]} -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
