"""Read the running game's memory (DOSBox-X process, same container).

ParaTrooper is a 16 KB .COM program: code and every variable live in one
64 KB real-mode segment (PSP at offset 0, program image from 0x100). The
emulator keeps guest RAM in its own process memory, so the segment can be
found by searching for the program's code bytes and read through
/proc/<pid>/mem (the bot runs as root in the emulator's container).

Two copies of the image show up: the file DOSBox-X loaded (pristine) and the
live one in guest RAM (its variables differ from the file). find() keeps
the one whose bytes actually change.
"""
import os
import time

COM_PATH = "/assets/ParaTrooper.1982.com"

# Segment layout found so far (captures/mem1, round 1; correlation with the
# vision stack's detections -- PROVISIONAL until checked against the code).
# "col" = CGA byte column = game x // 8; "native" = 320x200 pixels.
LAYOUT = {
    "tick": (0x1BC0, "u16: game tick counter, +1 every tick"),
    "score": (0x2C10, "5 bytes: score digits, least significant first"),
    "hiscore": (0x1BAB, "4 bytes: hi-score digits, least significant first (HUD text at 0x1B9F)"),
    "bullet_x": (0x2069, "6 x u16: bullet x, native px (0 = free slot)"),
    "bullet_y": (0x20A5, "6 x u8, stride 2: bullet y, native px"),
    "bullet_dir": (0x20E1, "6 x u8, stride 2: probably the bullet's lane / direction"),
    "bomb_x": (0x203E, "u8 per slot (4 parallel 20-byte arrays from 0x2016): bomb x, col"),
    "bomb_y": (0x2052, "u8 per slot: bomb y, native px"),
    "trooper_y": (0x1D63, "22 drop columns, stride 6: trooper y = native body y - 16"),
    "trooper_a": (0x1CC3, "22 drop columns, stride 6: u16, unknown (timer?)"),
    "trooper_state": (0x1E03, "22 drop columns, stride 6: 2-valued state"),
    "heli_lane17": (0x1BF8, "per column (index = game x // 8): nonzero byte where a helicopter is"),
    "heli_lane41": (0x1C5A, "same for the y=41 lane; lanes 0x62 bytes apart"),
    "debris": (0x24D4, "104-entry tables at 0x24D4 / 0x272C (CGA addresses) and 0x2984 (velocity)"),
}
SEG = 0x10000
_SIG_OFF, _SIG_LEN = 0x100, 64     # a stretch of code past the entry point (not patched at startup)


def _dosbox_pid():
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                cmd = open(f"/proc/{p}/cmdline", "rb").read()
            except OSError:
                continue
            if b"dosbox-x" in cmd.split(b"\0")[0]:
                return int(p)
    raise RuntimeError("no dosbox-x process")


class GameMemory:
    def __init__(self, com_path=COM_PATH):
        self.com = open(com_path, "rb").read()
        self.pid = _dosbox_pid()
        self.fh = open(f"/proc/{self.pid}/mem", "rb", buffering=0)
        self.base = self.find()

    def _candidates(self):
        sig = self.com[_SIG_OFF:_SIG_OFF + _SIG_LEN]
        out = []
        for line in open(f"/proc/{self.pid}/maps"):
            f = line.split()
            if "rw" not in f[1]:
                continue
            a, b = (int(x, 16) for x in f[0].split("-"))
            if b - a > 256 << 20:
                continue
            try:
                self.fh.seek(a)
                buf = self.fh.read(b - a)
            except OSError:
                continue
            i = buf.find(sig)
            while i >= 0:
                out.append(a + i - _SIG_OFF - 0x100)   # segment start = image - 0x100 (PSP)
                i = buf.find(sig, i + 1)
        return out

    def find(self, wait_s=0.5):
        """Segment base of the live copy: the candidate whose bytes change."""
        cands = self._candidates()
        if not cands:
            raise RuntimeError("game image not found in emulator memory")
        first = {c: self._read_at(c) for c in cands}
        time.sleep(wait_s)
        changed = {c: sum(x != y for x, y in zip(first[c], self._read_at(c))) for c in cands}
        live = max(cands, key=lambda c: (changed[c], self._diff_from_file(first[c])))
        return live

    def _diff_from_file(self, seg):
        img = seg[0x100:0x100 + len(self.com)]
        return sum(x != y for x, y in zip(img, self.com))

    def _read_at(self, addr, n=SEG):
        self.fh.seek(addr)
        return self.fh.read(n)

    def read(self, lo=0, hi=SEG):
        """Bytes lo..hi of the game's segment (default: all 64 KB)."""
        return self._read_at(self.base + lo, hi - lo)


# the part of the segment holding all decoded game state (tick, seed, phase,
# helicopter maps, troopers, bombs, bullets, barrel, score): what run.py
# --mem records every frame
STATE_LO, STATE_HI = 0x1B00, 0x2D00
