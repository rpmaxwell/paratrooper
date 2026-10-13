# Paratrooper bot

A bot for Paratrooper (1982) running in DOSBox-X in Docker. It reads the game
(screen or memory), plans shots with the game's own physics, and records every
game so misses can be looked up later. How the bot works: [docs/process.md](docs/process.md).
How it got here: [docs/dev_history.md](docs/dev_history.md).

## Layout

To place a file, ask: **does a game run import it?** -> `paratrooper/`.
**Does it check something that must stay true?** -> `tests/`. Otherwise -> `lab/`.

```
paratrooper/            the product: everything a game run imports
  game/                 WHAT THE GAME IS: pure facts, no I/O, no clocks
    model.py            lanes, motion, hitboxes, bomb heights, reach floor
    geometry.py         screen / native / game coordinates, palette
    tables.py chute.py  intercept tables, chute-opening model
    data/               calibrated numbers (written by lab/calibration)
  emulator/             HOW THE GAME RUNS: docker.py (containers, speed), process_memory.py (read game RAM)
  observe/              HOW WE SEE IT -> Observation (base.py)
    vision.py memory.py the two observer kinds
    sprites.py barrel.py hud.py memstate.py screen.py   their building blocks
  act/                  HOW INPUT GETS IN: xtest.py (X keyboard events), fake.py (offline replay)
  control/turret.py     goto / clamp / fire -> key timing
  policy/               WHAT TO SHOOT: heuristic.py (+ Params), plan.py, prob_plan.py (shot planning)
  world/                tracking + outcome telemetry (to split: tracking -> observe, outcomes -> referee)
  record/               WHERE IT GOES: store.py (run layout), game.py (per game), writer.py
  sim/                  offline game simulator (spawns, drops, value of a landing)
  config.py             TOML -> dataclasses; KINDS = every swappable implementation
  runner.py             the game loop (runs in the container)
  cli/ __main__.py      host commands: up, down, play, ab, archive, replay, runs
configs/                base.toml (the champion), observer/, emulator/, exp/, archive/
tests/                  pytest; fixtures/ holds a short recorded game
lab/                    exploration: calibration, disasm, diagnostics, turbo, notebooks (lab/README.md)
docker/                 image, DOSBox-X config template, entrypoint
docs/                   process, history, PRD, old plans
runs/                   (gitignored) all game output
```

## Commands

```
uv venv && uv pip install -e '.[analysis,dev]'      # host environment
python -m pytest                                    # ~3 s
python -m paratrooper up [--instance N]             # emulator container (VNC on 5900+N)
python -m paratrooper play configs/base.toml --games 5
python -m paratrooper ab configs/base.toml configs/exp/<name>.toml --games 20
python -m paratrooper replay runs/<run>/<arm>/g000  # offline, any observer/policy config
python -m paratrooper archive list
```

Runs land in `runs/<date>_<name>/` (layout: `paratrooper/record/store.py`).
Query every game of every run: `duckdb -c "SELECT arm, avg(score) FROM read_json_auto('runs/*/*/*/summary.json') GROUP BY arm"`.
Captures from before 2026-10-13 are in `../paratrooper/captures/` in the old
layout (`game_<t0>.jsonl/.npz/_mem.npz`); `replay` and the lab diagnostics read it.

## Rules

1. **Variants are config, not copies.** A different parameter is a config value
   (`[policy.params]`). A different algorithm is a new kind registered in
   `config.KINDS`, named for what it does (`memory`, `bios`, `ev`), never `v2`/`new`.
   No env-var switches.
2. **Every finished experiment is archived.** `ab` writes both arms' resolved
   configs to `configs/archive/<run>__<arm>.toml` with `[meta]` (git sha, run,
   mean score, `result`, `status = "pending"`). Fill in `result` and
   `status` (`promoted` / `rejected`, and `control` for the arm it was
   compared against) and commit. A promoted arm's changes go into
   `base.toml` or the defaults. A rejected arm's *code* is deleted; its
   archived config stays. Loading an archived config whose kind is gone names
   the commit to check out.
8. **Judge A/Bs on what the change can move.** Score varies ~550 game to
   game: 6 games per arm can't detect anything smaller than a doubling (an
   A/B with identical play showed 767 vs 1,312). Prefer per-minute rates and
   the bullet audit (`ab` prints both, s.e. clustered by game), and when a
   change is meant to leave play alone, check that with `replay` (decisions
   and presses identical).
3. **Every run records what produced it**: resolved config per arm, git sha,
   uncommitted diff (`git.diff`). Commit before A/Bs you want to keep.
4. **Ticks inside, seconds only at the edges.** New code counts game ticks;
   wall-clock seconds belong only in `emulator/` and `act/`. Existing
   `*_S` constants (world.py, turret.py, xtest.py, heuristic.py, runner.py) are
   debt: they are why `play` refuses `emulator.speed = "turbo"`.
5. **Facts, hosting, choices.** `game/data/` holds facts about the game;
   emulator/actuator configs hold facts about how it's hosted (key latency in
   ticks depends on speed and actuator); policy params are our choices.
6. **Behaviour changes are deliberate.** `tests/test_replay.py` replays a
   recorded game and pins its outcome. A refactor keeps it green; a change
   that moves it regenerates `tests/fixtures/g_short/expected.json` and says
   why in the commit.
7. **Grow when needed.** A file becomes a folder at its second
   implementation or ~400 lines; an interface appears with the second
   implementation. Lab code moves into the package when a run needs it,
   with a test.
