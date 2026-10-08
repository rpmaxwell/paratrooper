"""Disassemble ParaTrooper.1982.com (16-bit real mode, loads at CS:0100).

    python3 -m paratrooper.tools.disasm <com_file> [out.asm]

The entry point (CS:0100) is a trampoline: push CS+0x2C4, push 0, retf --
the game's code runs in its own segment starting 0x2C40 into the program
segment (file offset 0x2B40), while DS stays the program segment, so data
operands are offsets in the same numbering as io/memory.py's LAYOUT.
Recursive descent from the code entry (direct jumps / calls, both sides
of conditional jumps), so data is not decoded as code. Listing addresses
are CODE-segment offsets. Memory operands that hit a LAYOUT entry
are labeled, and the listing ends with a cross-reference index: which
instructions touch each known variable. Needs `capstone` (pip).
"""
import collections
import re
import sys

import capstone

from ..io.memory import LAYOUT

ORG = 0x100
CODE_SEG = 0x2C40      # program-segment offset of the code segment (CS + 0x2C4 paragraphs)
_FLOW_END = {"ret", "retf", "iret", "jmp", "ljmp"}
# The code starts with DS = PSP + 0x11 paragraphs: data operands are LAYOUT
# offsets (measured from the PSP) minus 0x110.
DS_SHIFT = 0x110
_KNOWN = sorted((addr - DS_SHIFT, name) for name, (addr, _) in LAYOUT.items())
LABEL_SPAN = 0x30


def _label(addr):
    """name+k of the LAYOUT entry at or just below addr (within 0x100)."""
    best = None
    for a, name in _KNOWN:
        if a <= addr < a + LABEL_SPAN and (best is None or a > best[0]):
            best = (a, name)
    if best is None:
        return None
    return best[1] if addr == best[0] else f"{best[1]}+{addr - best[0]:#x}"


def disassemble(image, entry=0):
    """image: the .COM file. Disassembles the code segment from `entry`."""
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_16)
    md.detail = True
    code = image[CODE_SEG - ORG:]
    ORG_ = 0
    end = len(code)
    insns, todo, seen = {}, [entry], set()
    calls = set()
    while todo:
        pc = todo.pop()
        while ORG_ <= pc < end and pc not in seen:
            ins = next(md.disasm(code[pc:pc + 15], pc), None)
            if ins is None:
                break
            seen.update(range(pc, pc + ins.size))
            insns[pc] = ins
            m = ins.mnemonic
            target = None
            if ins.operands and ins.operands[0].type == capstone.x86.X86_OP_IMM and (
                    m.startswith("j") or m == "call" or m.startswith("loop")):
                target = ins.operands[0].imm & 0xFFFF
                todo.append(target)
                if m == "call":
                    calls.add(target)
            if m in _FLOW_END or (m == "int" and ins.op_str in ("0x20",)):
                break
            pc += ins.size
    return insns, calls


def _jump_tables(insns, image):
    """Targets of `jmp word ptr [reg + table]` (table in the data segment):
    consecutive words that are plausible code offsets."""
    code_len = len(image) - (CODE_SEG - ORG)
    out = set()
    for ins in insns.values():
        if ins.mnemonic in ("jmp", "call") and ins.operands and ins.operands[0].type == capstone.x86.X86_OP_MEM:
            table = ins.operands[0].mem.disp & 0xFFFF
            f = table + DS_SHIFT - ORG                  # file offset of the table
            for k in range(32):
                if f + 2 * k + 2 > len(image):
                    break
                t = image[f + 2 * k] | image[f + 2 * k + 1] << 8
                if not 0 < t < code_len:
                    break
                out.add(t)
    return out


def disassemble_all(image):
    """Recursive descent from the entry, then from jump-table targets, then
    from gaps that decode as plausible code (handlers installed through
    interrupt vectors are reached no other way)."""
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_16)
    code = image[CODE_SEG - ORG:]
    insns, calls = disassemble(image, 0)
    entries = {0}
    for _ in range(6):
        new = (_jump_tables(insns, image) - set(insns)) - entries
        covered = set()
        for pc, ins in insns.items():
            covered.update(range(pc, pc + ins.size))
        # gap starts after a flow end: try them
        for pc in sorted(insns):
            nxt = pc + insns[pc].size
            if insns[pc].mnemonic in _FLOW_END and nxt not in covered and nxt < len(code):
                body = list(md.disasm(code[nxt:nxt + 40], nxt))
                if len(body) >= 4 and not any(i.mnemonic in ("(bad)",) for i in body):
                    new.add(nxt)
        if not new:
            break
        for e in new:
            more, c2 = disassemble(image, e)
            for k, v in more.items():
                insns.setdefault(k, v)
            calls |= c2
            entries.add(e)
    return insns, calls


def listing(insns, calls):
    lines, xref = [], collections.defaultdict(list)
    prev_end = None
    for pc in sorted(insns):
        ins = insns[pc]
        if prev_end is not None and pc != prev_end:
            lines.append("")
        if pc in calls:
            lines.append(f"sub_{pc:04x}:")
        notes = []
        for op in ins.operands:
            if op.type == capstone.x86.X86_OP_MEM and op.mem.base == 0 and op.mem.index == 0 or \
                    op.type == capstone.x86.X86_OP_MEM:
                disp = op.mem.disp & 0xFFFF
                name = _label(disp)
                if name:
                    notes.append(name)
                    write = ins.operands[0].type == capstone.x86.X86_OP_MEM and ins.operands[0].mem.disp == op.mem.disp \
                        and ins.mnemonic not in ("cmp", "test", "push")
                    xref[name.split("+")[0]].append((pc, "W" if write else "R", f"{ins.mnemonic} {ins.op_str}"))
        txt = f"{pc:04x}  {ins.bytes.hex():<14} {ins.mnemonic:<6} {ins.op_str}"
        if notes:
            txt = f"{txt:<60} ; {', '.join(notes)}"
        lines.append(txt)
        prev_end = pc + ins.size
    lines += ["", "; ---- cross-references to known variables (io/memory.LAYOUT) ----"]
    for name in sorted(xref):
        lines.append(f"; {name}:")
        for pc, rw, txt in xref[name]:
            lines.append(f";   {pc:04x} {rw}  {txt}")
    return lines


def main(argv):
    code = open(argv[0], "rb").read()
    insns, calls = disassemble_all(code)
    lines = listing(insns, calls)
    out = argv[1] if len(argv) > 1 else None
    covered = sum(i.size for i in insns.values())
    head = f"; {argv[0]}: {len(code)} bytes, {len(insns)} instructions ({covered} bytes) reached, {len(calls)} subroutines"
    text = "\n".join([head] + lines) + "\n"
    if out:
        open(out, "w").write(text)
        print(head, "->", out)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main(sys.argv[1:])
