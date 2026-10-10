# Paratrooper bot: process and handoff

State as of 2026-10-09 (branch `calibrated-phit-bombs-memory`, PR #1). How the
bot is built, step by step, with the code behind each step, what is known
for certain, what is still estimated, and what to do next. The trial and
error that led here is in [dev_history.md](dev_history.md).

All paths below are under `scripts/paratrooper/` unless noted.

---

## Step 0: Infrastructure

DOSBox-X runs the 1982 game in Docker under a virtual X display; the bot
runs in the same container, reading the screen and sending keys.

| what | code |
|---|---|
| container, emulator, virtual display, VNC (`./run.sh [n]`) | `docker/`, `run.sh` |
| screen grab (native 320x200 palette frame) | `io/screen.py` |
| key input (XTEST on a persistent X connection, non-blocking press/release) | `io/keys.py` |
| game memory (the game's 64 KB segment, read from the emulator process) | `io/memory.py` |
| main loop: grab -> barrel -> turret -> world -> policy, per distinct frame | `run.py` |
| telemetry (one jsonl per game) | `telemetry/writer.py` |

Run: `docker exec <env vars> paratrooper bash -c "cd /scripts && python3 -m paratrooper.run <n_games> /captures/<dir> [--record] [--mem]"`.
`--record` saves frames (`game_<t0>.npz`: `frames` uint8 palette indices
N x 200 x 320, `t` seconds since start), `--mem` the game's state bytes per
frame (`game_<t0>_mem.npz`). Both are held in RAM and written at game end.

## Step 1: Reading the game state

### 1a. Vision

| item | how | code |
|---|---|---|
| frames | native 320x200, **palette indices 0-3** (black, cyan, magenta, white) | `io/screen.py` |
| objects | connected components per colour + exact shape lookups: helicopter skids, trooper body, canopy, bomb disc, landed heads; a **bullet is one pixel** | `perception/sprites.py` |
| turret angle, score, game over | template reads | `perception/barrel.py`, `perception/hud.py` |
| tracking | lifecycles: trooper free fall -> canopy -> fate; aircraft; bombs; our bullets (a ledger: every Up press is a bullet) | `world/world.py` |
| game tick | inferred from sprite motion (phase-locked clock); a long-running error source (+-1 tick labels) | `world/clock.py` |
| shot outcomes | a bullet's disappearance matched to a target box on that tick; ~20% of kills uncredited ("lost") | `world/world.py` |
| landed / stacked troopers | heads that stay still; **over-counts landings and misses stack wipes** -- use frames or memory for landed state | `world/world.py` |

### 1b. Memory (ground truth)

The game's whole state lives in one 64 KB segment, found in the DOSBox-X
process by its code bytes and read through `/proc/<pid>/mem`.

| field | where (code's DS offset; memory offset = DS + 0x110) |
|---|---|
| tick counter | DS 1AB0 (u16, +1 per tick) |
| phase / phase timer | DS 1AB2 / 1AB3 |
| random seed | DS 1ADB (`seed = seed * 0x7781 + 0x64C9`, sub at CS:06F5) |
| helicopters | per-lane map: one nonzero byte per helicopter at column m (= x // 8); lane records at DS 1D91+bp (map base), 1D99+bp (band top row), 1D9D+bp (direction), bp = 2, 4 |
| troopers | per screen column c: screen address (sign = canopy) DS 1BA9 + 2c, top row DS 1C49 + 2c (= body y - 16, native), free-fall flag DS 1CE9 + 2c |
| bombs | slots at DS 1EF2: column at +0x3C, native row at +0x50 |
| bullets | 30 slots: x DS 1F59 + 2i, y +0x3C, direction +0x78 |
| barrel | DS 1F57: position = 19 - ptr / 2 (0, 0x28 = clamped sight lanes) |
| score | memory 0x2C10, digits least significant first |

Decoder: `perception/memstate.py` (same shapes as vision). Live feed:
`PARATROOPER_MEM_FEED=1` (world takes helicopters, troopers, bombs, bullets,
barrel and tick boundaries from memory; vision keeps planes, landed heads,
score, game over). Agreement with vision on a recorded round: barrel 99%,
troopers 553/559, helicopters 97%, bombs 94%. **Not yet A/B tested live.**

## Step 2: Calibration (the game model)

| item | state | code |
|---|---|---|
| turret | 19 positions + 2 clamped sight lanes, ~1 position/tick; **Up stops rotation and fires** (every stop costs a bullet); press -> bullet 3 ticks | `physics/model.py`, `control/turret.py` |
| bullets | no fixed fire rate: one bullet per Up, 30 slots; all spawn at native (160, 157) and move a fixed (dx, dy) per tick per direction (21 lanes) | `model.LANES` |
| target motion | helicopters/planes 8 px/tick on per-wave lanes; troopers 8 px/tick free fall, 4 under canopy; bombs: fixed height sequence, x moves with the plane (no "angle": they fall under it) | `model.py` |
| random processes | helicopter spawning (fitted), drops (**code: rand <= 0x99A, 3.75% per column pass**), chute opening (fitted hazard; the game has a threshold table at DS 0DF5 whose indexing is not yet confirmed), bomb release points | `sim/fit.py`, `sim/params.py`, `sim/sampler.py`, `physics/chute.py` |
| hitboxes | **all three confirmed from the code**: helicopter 6 columns (48 px) on the 8 px grid, band of 10 rows; trooper canopy/body/chuteless body = `fitted_parts`; bomb ~24 x 28 px tested before the bomb moves (`code`). All tested once per tick at the bullet's drawn position | `physics/model.py` |
| intercepts | precomputed meeting ticks for every lane x target phase; exact plans for deterministic targets, p(hit) over chute scenarios for free-fallers | `physics/tables.py`, `physics/plan.py`, `physics/prob_plan.py` |
| p(hit) vs outcomes | stop bullet, timing slip (17% one tick late), kill rate per contact (0.90 drawn, 0.12 between ticks), canopy share 0.75 | `sim/phit.py`, `PARATROOPER_PHIT=calibrated` |
| the game's code | full listing with labels and cross-references | `tools/disasm.py` -> `captures/derived/paratrooper.asm` |

## Step 3: Heuristics

`policy/heuristic.py` with `control/turret.py`. Every engagement is a
**job**: move -> (clamp) -> fire at planned ticks, timed off the target's
own tick counter, checked for abort every frame.

| rule | detail |
|---|---|
| priority | bombs preempt everything (any phase); planes in the plane phase; then troopers by urgency (ticks to ground - 6 x side crowding - 100 for a canopy over a landed trooper); pre-aim; helicopters; park for planes |
| crush shots | canopy-only plans for a trooper over a landed one |
| interrupts | helicopter jobs abort on a new/changed trooper; **trooper jobs abort only for bombs/planes** (no other trooper can preempt); pre-aim moves can't be interrupted |
| shots per target | troopers 1-2 (window width, p(hit) gain); bombs: a run of shots, <= 3 in flight; pair-aware (`PARATROOPER_BOMB_PAIRS=1`) and earliest intercept (`PARATROOPER_BOMB_BOX=code`) |

## Step 4: Expected value

`sim/value.py`: V(round, tick, landed left, landed right) by backward
induction from per-round point, landing, crush and bomb-death rates
(landings from recorded frames). Gives prices in points for a landing, a
crush, a trooper / helicopter / bomb kill, a bullet. Validated on the
current bot's games (expected 1,126 vs 1,176 observed; round-reach curve
within ~3 points). **Not wired into the policy.**

Diagnosis says the constraint is **gun time**: an engagement holds the gun
~18 ticks (~1 s), a trooper falls ~54 ticks (~3 s), and 56% of landings were
never engaged (35/40 while 1-6 other troopers were falling). EV targeting
must therefore score options in **points per tick of gun time** (or
schedule a short horizon), not argmax per frame.

## Step 5: Measurement

| tool | use |
|---|---|
| telemetry jsonl, `--record`, `--mem` | every game |
| A/B by environment variable, one container per arm | `run.py` logs every variable in `game_start` |
| `tools/mem_diag.py` | per trooper: hit / why the shot missed / never engaged, from memory |
| `tools/review_crush.py`, `tools/audit_misses.py` | frame-level shot reviews |
| `sim/telemetry.py` + notebook | loading games, fitting the sim |

Results (mean score): 1,049 baseline -> 1,211 fitted helicopter box (40
games each); 1,388 new settings, 1,608 diagnosis run (10 games each, within
noise; game-to-game sd ~550).

---

## Current switches

| variable | values (default first) | recommended |
|---|---|---|
| `PARATROOPER_HELI_BOX` | model, fitted | fitted (= the game's box) |
| `PARATROOPER_TROOPER_BOX` | model, fitted, fitted_parts | fitted_parts (= the game's box) |
| `PARATROOPER_BOMB_BOX` | model, code | code (= the game's box) |
| `PARATROOPER_PHIT` | model, calibrated | calibrated (needs fitted_parts) |
| `PARATROOPER_BOMB_PAIRS` | unset, 1 | 1 |
| `PARATROOPER_MEM_FEED` | unset, 1 | A/B first |
| `PARATROOPER_FREE` | unset, pointblank | unset (old A/B arm) |

## Known issues

- Telemetry over-counts landings; use frames/memory.
- `tools/check_tables` shows 2,042/3,000 helicopter plans matching its reference -- same on master, pre-existing.
- `tools/replay.py` downsamples frames for an older recording size (`[1::2, ::2]`); broken on current recordings.
- Chute table at DS 0DF5 gives ~1.0-1.7% per step vs a fitted 2.3%+ per tick: indexing not yet understood (check against `--mem` captures: free-fall flag flips + V).
- Collision code skips the trooper test in game phases 2-4; which phases those are is unconfirmed.
- Frames and memory recordings are written at game end; a crash loses the game.

## Next steps

**Refactor first (short):**
1. Make the code-confirmed settings the defaults; delete the `model`/`fitted` variants, their table caches and the switches above (git keeps the history).
2. Restructure `heuristic.py` as *enumerate candidate jobs -> score -> pick*. EV targeting needs this slot, and the measured fixes go there: earliest plan within ~0.05 of the best p(hit); priority by the **reach-floor deadline** instead of ticks to ground (the unengaged landings are at the outer columns, where the floor is ~264-288 px); preemptable trooper jobs.

**Then EV targeting:** score each candidate job as `plan_p x value price - bullets`, per tick of gun time, using `sim/value.py` prices; refit after each policy change (the values describe the current policy).

**Cleanup:** replace fitted estimates with the game's (chute table, drop rate); memory as the primary sensor (shrink world.py's vision workarounds); retire `sim/collision.py` box fitting, `tools/check_heli_box.py`, `check_trooper_box.py`, `show_trooper.py`, `mem_capture.py`; refit `sim/phit.py` under the new planner; archive old `captures/`.

**Later:** keys through the BIOS keyboard buffer (`PARATROOPER_KEYS=memory`, exact fire ticks); predict chute openings and drops from the random seed (the solver endgame).

**Scope note:** the original PRD forbids emulator memory reads ("the entire point of the POC"). Memory is now used for ground truth and an optional live feed; decide whether the live bot may use it, or whether it stays a calibration/measurement tool.
