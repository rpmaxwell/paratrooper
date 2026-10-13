# Refactor Plan: Fast, Verified, RL-Ready

## Context

The current bot (`scripts/bomb_defense.py` + `bomb_model.py`, `trooper_model.py`,
`heli_model.py`, `turret.py`) plays well on deterministic tick models: bombs
43/43 in validation, one-shot plane kills, mean score ~520, best 1124. It grew
ad hoc: ~45 scripts / 8.8k lines, a Python per-pixel pipeline, blocking control
loops, and logs that **disagree with ground truth** (the frame-by-frame shot
audit found 7 trooper misses where the bot logged 3; the helicopter "kills" in
later waves were once entirely false).

Next step is RL for target selection / priority (multi-target scheduling is
left to RL). This refactor must:

1. **Be fast**: under ~3 ms of processing per game tick (tick = 55 ms), leaving
   headroom for an emulator running faster than real time.
2. **Produce verified data**: every shot, target, and outcome logged from
   observed evidence and self-checked, because RL rewards and diagnostics
   come from it.
3. **Separate solved from learned**: perception, physics, aiming, and bombs
   are deterministic; RL chooses what to shoot and where to wait.
4. **Reach parity first**: the refactored heuristic bot must match today's
   results before RL starts.

## Measured bottlenecks (2026-09-27)

| Cost | Now | Fix | Expected |
|---|---|---|---|
| RGB -> palette | 13 ms/frame, done twice (player + recorder) | native 320x200 (crop at +1 row is an exact 2x upscale: 100% uniform 2x2 blocks), one bit-op on 2 channels | ~0.2 ms |
| Keypress | `xdotool` subprocess, 23-32 ms (median 27) | persistent python-xlib XTEST connection | <1 ms, low jitter |
| Sprite grouping | pure-Python connected components | OpenCV/SciPy labeling + exact shape table | ~0.3 ms |
| Planning | Python tick simulation per lane per target (~7.5 ms/trooper) | precomputed hit tables | microseconds |
| Loop | polls ~86 fps (mostly duplicate frames); blocking rotate/fire loops | once per game tick; non-blocking skills | ~4x less work |

The keypress jitter is also the likely cause of most remaining +-1 tick
timing misses.

## Architecture (`paratrooper/` package, replaces the scripts)

- **`io/`**: `screen.py` (region capture, native-res palette frames),
  `keys.py` (persistent XTEST input, every press timestamped), `clock.py`
  (game-tick detector; everything downstream counts ticks, not seconds).
- **`perception/`**: vectorized detection of all measured sprites: bomb (8x8
  disc), trooper body (8x12 + head) and canopy (24x12 dome), helicopter
  skids (32x4), plane (48x20), bullets (2x2), landed heads, barrel position,
  on-screen score.
- **`world/`**: entity tracker with persistent ids and states; our own
  bullets tracked individually (see Telemetry).
- **`physics/`**: the 19 barrel positions + the 2 limit-lane variants, and
  **precomputed intercept tables** built once and cached: troopers
  [position x x x y x fall speed x part], helicopters/planes [position x lane
  x direction x x], bombs [position x release x]. Planning = lookups plus
  reachability. The same tables are RL features ("which lanes hit target i,
  how soon").
- **`control/`**: non-blocking skills (goto, clamp, fire-at-plan, park),
  advanced once per tick; a **bomb reflex** preempts everything.
- **`policy/`**: heuristic policy (today's behavior: baseline + warm start)
  and the RL policy, both choosing from the same option list.
- **`env/`**: Gymnasium env. Step = decision point. Observation = fixed-size
  features (targets by urgency, per-target lane feasibility, turret state,
  landed counts per side, phase). Action = option (shoot target i / pre-aim
  position p / wait) with masking (MaskablePPO). Reward = verified score
  delta.
- **`telemetry/`**: see below.
- **`replay/`**: runs recorded frames (~150 existing files) through
  perception, tracking, and telemetry as offline regression tests.

## Telemetry: verified by construction

Principle: **an outcome is only logged when the frames show it.** Intentions
("we fired at trooper 17") and observations ("bullet 212 vanished inside
trooper 17's canopy box on tick 5031, and the canopy was gone next tick")
are separate records, joined by ids. Anything that cannot be explained is
logged as unexplained, never guessed.

### Trooper lifecycle (one record per trooper)

| Field | Meaning |
|---|---|
| `id`, `x` | column; x must stay constant, and any drift is flagged as a tracking error |
| `first_seen` | tick and y of first appearance, plus the helicopter it left (lane, x, direction) when attributable |
| `canopy_open` | tick and y when the canopy first appears (null if never) |
| `y_by_tick` | compact per-tick y trace (~50 ints) for fall-model checks and RL analysis |
| `engagements` | ids of every shot aimed at it (plan: position, lane variant, part, predicted meeting tick and point) |
| `fate` | exactly one of: `body_killed`, `chute_killed` (then `fell_dead`, and `crushed` = the landed trooper id it landed on, if any), `landed` (tick, x, side, stack height), `lost` (tick, reason) |
| `fate_evidence` | the bullet id and collision point that caused a kill, or `unexplained` |
| `missed_reason` | for any trooper that landed: never engaged / engaged and missed / unreachable (no lane could hit it) / turret busy |

### Bullet ledger (one record per bullet we fire)

- Every Up press is a bullet, including goto stops and clamps: `shot_id`,
  press tick, intent (`aimed` with target id and plan, `stop`, `clamp`).
- The bullet is found in the frames and followed tick by tick: spawn point,
  **measured velocity** (checked against the lane table; this catches
  lane-variant bugs like the limit lanes live), per-tick positions.
- End: `hit` (entity id + region: canopy box / body box / helicopter /
  plane / bomb) when it vanishes where its next position overlaps an entity
  that disappears or changes state that tick; `left_screen`; or
  `unexplained`.
- Prediction error per aimed shot: planned vs. actual meeting tick and
  point. This is the timing-accuracy metric.

### Self-checks (run live, summarized per game)

1. **Score reconciliation.** Every change in the on-screen score is
   explained by logged events (shot = -1, kills at measured point values).
   Point values are calibrated from data in Phase 2; unexplained deltas are
   counted and flagged. This is also the RL reward, so it must be right.
2. **Lifecycle closure.** Every trooper id ends in exactly one fate; open
   ids at game over are errors.
3. **Bullet closure.** Every press maps to exactly one bullet record (or
   `not_found`, counted).
4. **Audit agreement.** A sample of shots gets evidence clips, and the
   offline audit (`audit_trooper_shots.py`, moved into the package) is
   re-run against the logs. The agreement rate is tracked as a metric.

### Budget and storage

Tracking bullets and entities is part of the per-tick perception (~0.5 ms).
Records are buffered in memory and written per game as JSONL by a
background thread; the control loop never blocks on I/O. At roughly 50
troopers and 200 bullets per game this is ~200 KB per game.

## Phases (each gated before the next)

0. **Throughput + baseline.** Can DOSBox-X run the game faster than real
   time (fixed higher cycles / turbo), and does the 18.2 Hz game tick scale
   with it? Measure N parallel containers (`run.sh` supports instances).
   Record the parity baseline from the current bot: mean score, bomb %,
   audited trooper hit rate.
1. **I/O layer.** Native capture, XTEST input, tick clock. *Gate:* key
   latency and jitter measured; converted frames identical to today's on
   recordings.
2. **Perception, world model, telemetry.** *Gate (on replayed recordings
   and live):* detections match the current code; lifecycle and bullet
   closure 100%; audit agreement >= 98%; >= 99% of score changes explained;
   point values calibrated.
3. **Physics tables.** *Gate:* tables match the current simulators on a
   large randomized set; planning in microseconds.
4. **Skills + bomb reflex + heuristic policy (port of current behavior).**
   *Gate (live, 20+ games):* bombs >= 98%, mean score >= ~520, per-tick p99
   < 5 ms, zero turret-stop misses, telemetry self-checks passing.
5. **RL interface.** Env, feature observations, action masking, vectorized
   envs across containers. *Gate:* 100 random-policy episodes error-free;
   the heuristic policy through the env reproduces Phase 4's scores.
6. **Cleanup.** Archive the ~35 superseded scripts; calibration and audit
   tools become maintained package utilities.

## Decisions (recommended defaults)

1. **RL observations: computed features**, not raw pixels. They come only
   from the screen (no memory reads), so the PRD's vision-only rule holds,
   though the PRD's baseline was a CNN on frames.
2. **Action level: choose among precomputed options**, not raw
   rotate/fire.
3. **Bombs: a fixed reflex outside RL.**
4. **Language: Python** with numpy, OpenCV or SciPy, python-xlib. A
   compiled rewrite only if Phase 0 shows the emulator can outrun us.

## Riskiest unknown

Phase 0: if the game cannot run faster than real time, RL sample
throughput depends entirely on parallel containers, which changes how much
the rest of the speed work matters.

---

## Implementation status (2026-09-27/28)

Package: `scripts/paratrooper/` (old scripts untouched and still runnable).
Run: `docker exec paratrooper bash -c "cd /scripts && python3 -m paratrooper.run <n_games> <out_dir> [--record]"`.

| Phase | Status | Evidence |
|---|---|---|
| 1 I/O | done | capture identical to old path 40/40 frames; grab+convert 0.18 ms (was ~13); XTEST key->barrel 32 ms median (xdotool 61) |
| 2 perception | done | `tools/check_perception.py` over 10,962 recorded frames: bombs, bullets, score, game-over 0 disagreements; every other difference is the new detector seeing sprites the old scan band clipped |
| 2 world + telemetry | validating | `tools/replay.py`, `tools/check_telemetry.py` (closure, score reconciliation, frame audit) |
| 3 tables | done | `tools/check_tables.py`: bomb/trooper/plane plans identical to old planners (1000s of random cases); heli identical to the corrected simulator 4000/4000; 0.08-0.14 ms per plan (old 3.7-7.1 ms) |
| 4 control + heuristic policy | validating | loop p50 ~2.5 ms, p99 ~4.5-5 ms |

### Findings made while building it
- **Game tick clock** must handle wrap-around when phase-locking to sprite
  moves; otherwise tick numbering double-counts / skips (bullet steps).
- **Helicopter sprite is 48 px wide**, not 44 (old measuring window clipped
  the tail). Box: flying left x0..x0+47, right x0-16..x0+31.
- **Scoring (measured from telemetry):** shot -1 (not below 0), helicopter
  +10, plane +10, bomb +30, trooper body +5, chute kill +5 *when the
  trooper hits the ground*.
- **The turret-stop bullet** (every goto ends with Up, which fires) flies
  down the planned lane a few ticks before the aimed shot and very often
  makes the kill. The aimed shot is then redundant -- directly relevant to
  shot economy and to the RL action design.
- **Press -> spawn latency is 3 ticks** (91-98%) regardless of how soon
  after the previous press. An apparent "2 ticks when presses are 1 tick
  apart" was a bullet-tracker identity swap between two bullets on the same
  lane one tick apart; fixed by clock-exact tracking and constraining a
  press to claim only a bullet whose inferred spawn tick is T+2..T+4.

## Chute-opening model + hit-probability planner (2026-09-28)

- `physics/chute.py` -- fitted from telemetry (940 troopers seen from their
  drop, 449 random openings; Kaplan-Meier style, censoring troopers shot or
  lost before opening). Rule: the chute opens at a **uniformly random tick
  1..45 after the drop, but never lower than y = 241** (a trooper still in
  free fall there opens). Opening doesn't depend on screen column; drop
  height matters only via when the floor is reached. Validated per drop
  height: P(open at floor) 0.43 vs 0.41 observed (drop y 33), 0.50 vs 0.45
  (drop y 57); median opening tick 22 vs 22 / 21. Refit:
  `python3 -m paratrooper.physics.chute <run dirs>`.
- `physics/prob_plan.py` -- for a free-faller, P(hit) of a shot = sum over
  opening ticks k of P(k) * [bullet meets the body before k, or the
  chute/body after k, before landing], from a bitmask table of every bullet
  age that touches a box (`trooper_bits`, identical first hits to the
  first-hit table over all 6.7M entries). Checked against brute-force
  two-phase simulation: 2997/3000 identical (the 3 are zero-probability or
  transition-tick edge cases). ~2 ms per free-faller; also considers a pair
  of shots (one per likely phase) when it adds >= 0.15.
- Policy: free-fallers are engaged when the best shot has P(hit) >= 0.5
  (replaces "point-blank only"); `PARATROOPER_FREE=pointblank` restores the
  old rule for A/B tests. Most free-fallers turn out to have a P ~= 1 shot:
  the chute box is tall and the trooper slows, so one lane can cover every
  opening time.
- First game: 1732 points in 400 s (previous record 1124); landings 1.5/min
  (was ~3.3/min); 90% of engaged free-fallers did not land.
  `tools/audit_misses.py` now judges free-fallers by P(hit) too -- in that
  game no landing was "unreachable"; all were out-of-position (4),
  engaged-and-missed (4) or busy (2).

## Accuracy fixes and A/B (2026-09-29/30)

- **Hits register at drawn positions.** The game checks a bullet against a
  trooper where the bullet is drawn each tick, not along its path between
  ticks: fast bullets tunnel through a 16 px free-falling body. In 192
  logged trooper shots, when the only predicted contact was between ticks
  39% of targets still landed vs 9% with a drawn-position overlap. Trooper
  tables now come in both variants and the planners weight contact by its
  measured kill rate: drawn 0.91, between-tick only 0.61
  (`tables.P_DRAWN`, `P_SWEPT_ONLY`). Bombs/aircraft keep the swept rule.
- **Positions are stamped with the tick they first appear.** Sprites are
  redrawn one at a time after a tick, so a later frame can still show the
  old position after the clock moved on; the tracker could date a position
  a tick late, putting plans 8 px behind in free fall.
- **Immediate retargeting:** a trooper whose bullets all end without a kill
  is a target again at once (bullet ledger), not only after the planned
  meeting tick.
- **Stacks / targeting area:** a trooper that stops moving has landed (never
  targeted); per-column landing height; per-column lowest reachable height
  (`model.REACH_FLOOR`, y ~258 at the edges to ~311 beside the turret) --
  troopers below it are dropped from the candidates, so unreachable ones no
  longer crowd the 4 planning slots. The earlier hold-fire rule for
  troopers over a landed one was removed (it made things worse).
- Spray/debris: no measurable effect -- troopers undetected on 0.7% of
  ticks, mostly without debris nearby; bullets that vanished had debris
  ahead no more often than bullets that flew off-screen.
- **Interleaved A/B, 8 games each** (committed 65f0cff vs this code): mean
  787 -> 1277, median 1012 -> 1335, landings/min 2.06 -> 1.62, engaged-and-
  missed landings 32/56 -> 19/67. The earlier 1424 six-game mean for
  65f0cff was a lucky draw -- only side-by-side comparisons are reliable.
- Remaining landings: out of position 55%, turret busy 15%, engaged and
  missed 28% -- mostly placement / target choice, the RL work.
