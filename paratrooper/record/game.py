"""Records one game into <arm dir>/<game_id>/ (layout: store.py)."""
import json
import pathlib

import numpy as np

from .writer import Writer

# events whose surroundings are kept under record.frames = "windows": the
# misses and the things telemetry couldn't explain
_UNEXPLAINED = ("vanished_unexplained", "lost")


def interesting_ticks(events):
    """Ticks worth keeping frames around: landed troopers and bombs, and
    every fate the telemetry could not explain."""
    out = []
    for e in events:
        typ, fate = e.get("type"), e.get("fate")
        if typ == "trooper" and (fate == "landed" or fate in _UNEXPLAINED or e.get("fate_evidence") == "unexplained"):
            ys = e.get("y_by_tick") or []
            out.append(ys[-1][0] if ys else e.get("first_tick"))
        elif typ in ("aircraft", "bomb") and fate in _UNEXPLAINED + ("landed",):
            out.append(e.get("last_tick", e.get("first_tick")))
        elif typ == "bullet" and (e.get("end") or {}).get("kind") == "vanished":
            out.append(e["end"]["tick"])
    return sorted(t for t in out if t is not None)


def window_mask(ticks, keep, half):
    keep = np.asarray(keep)
    if not len(keep):
        return np.zeros(len(ticks), bool)
    ticks = np.asarray(ticks)
    i = np.searchsorted(keep, ticks)
    lo = np.abs(ticks - keep[np.clip(i - 1, 0, len(keep) - 1)])
    hi = np.abs(ticks - keep[np.clip(i, 0, len(keep) - 1)])
    return np.minimum(lo, hi) <= half


class GameRecorder:
    def __init__(self, arm_dir, game_id, record_cfg, writer=None):
        self.dir = pathlib.Path(arm_dir) / game_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = record_cfg
        self.writer = writer or Writer()
        self.writer.open(str(self.dir / "events.jsonl"))
        self.events = []          # kept for the frame windows and the summary
        self.frames = []          # (t, tick, frame)
        self.mem = []             # (t, tick, snap)

    def emit(self, record):
        self.events.append(record)
        self.writer.emit(record)

    def frame(self, t, tick, frame):
        if self.cfg.frames != "none":
            self.frames.append((t, tick, frame))

    def memory(self, t, tick, snap):
        if self.cfg.memory:
            self.mem.append((t, tick, snap))

    def close(self, summary, mem_lo=None, mem_base=None):
        self.writer.close()
        self.writer.flush()
        if self.frames:
            t, tk, f = (np.array(x) for x in zip(*self.frames))
            if self.cfg.frames == "windows":
                m = window_mask(tk, interesting_ticks(self.events), self.cfg.window_ticks)
                t, tk, f = t[m], tk[m], f[m]
            if len(t):
                np.savez_compressed(self.dir / "frames.npz", t=t, tick=tk, frames=f)
            summary["frames_saved"] = int(len(t))
        if self.mem:
            t, tk, m = (np.array(x) for x in zip(*self.mem))
            np.savez_compressed(self.dir / "memory.npz", t=t, tick=tk, mem=m, lo=mem_lo, base=mem_base or 0)
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_default))


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)
