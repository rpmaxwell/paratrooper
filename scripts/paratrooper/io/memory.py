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

    def read(self):
        """The game's 64 KB segment, as bytes."""
        return self._read_at(self.base)
