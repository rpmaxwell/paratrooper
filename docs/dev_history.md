# Development history: how the bot got here

The trial and error behind [process.md](process.md), in order. Commit links
point at the code as it was then; files that later refactoring removes stay
reachable through them. Analyses run as one-off scripts (outside the repo)
are summarised here with their results.

Repository: https://github.com/rpmaxwell/paratrooper

---

## 0. Origins: a vision-only RL proof of concept (to mid-September 2026)

The project began as a precursor to an RL agent for F19 Strike Fighter,
which had stalled on reverse-engineering that game's memory. The
[PRD](https://github.com/rpmaxwell/paratrooper/blob/master/prd.md) scoped
Paratrooper as a test of the opposite approach: an agent that sees **only
pixels**, with "no emulator memory reads of any kind" as the point of the
exercise. A Gym-style environment (DOSBox-X under Xvfb, screen capture,
synthetic keys, score read by template matching) and PPO/DQN baselines were
built and beat a random agent.

What RL learned was the wrong game: it shot helicopters and ignored
paratroopers, because a hit's reward didn't distinguish target types and
aiming is a hard motor-control problem for trial and error. The plan
changed to **deterministic accuracy first, learned prioritisation later**
([plan_accuracy_and_priority.md](https://github.com/rpmaxwell/paratrooper/blob/master/plan_accuracy_and_priority.md)).

## 1. The accuracy grind (2026-09-18 to 09-27)

Calibration scripts measured rotation speed, fall speeds, helicopter
motion and the barrel's angles. Hit rate against "dead-on" shots stayed
around 11-14%, and repeated constant-tightening went nowhere. Several
lessons came out of this stretch (all in the plan document):

- **The measurement was the problem.** The ground truth itself (which shots
  hit) was unreliable. A target-free, gun-only test found the dominant
  error source once targets were taken out of the picture.
- **Watching beats aggregates.** The user watching over VNC repeatedly
  spotted what the statistics hid: clean misses on isolated shots, a delay
  between a drop and the gun's reaction, survival improving when per-shot
  numbers were ambiguous.
- **A wrong turn on bombs.** On 09-24 the bomb was judged not to be a real
  target ("shrapnel of the exploding turret"). On 09-25 frame-by-frame
  recordings showed the bomb is real, lethal and fully deterministic. It
  had looked canned because its physics never vary.
- **Tick models.** Targeting was rebuilt on the game's tick: troopers 8
  px/tick free fall, 4 under canopy; bombs on a fixed sequence; planes and
  the hidden clamped "limit lanes"; pixel-exact hitboxes from per-shot
  audits.

## 2. The refactor (09-27/28): fast, verified, RL-ready

[Commit 65f0cff](https://github.com/rpmaxwell/paratrooper/commit/65f0cff109cda0350172dd5ce23dc4f733e1d01c)
([refactor_plan.md](https://github.com/rpmaxwell/paratrooper/blob/master/refactor_plan.md)).
About 45 ad-hoc scripts became the `paratrooper/` package:
- I/O with grab+convert at 0.18 ms (was about 13 ms) and XTEST keys;
- a one-pass sprite detector;
- a world model with **verified telemetry** (logs had disagreed with ground truth);
- precomputed intercept tables (0.1 ms per plan, was 4–7 ms);
- a non-blocking heuristic policy.

Findings made while building it:
- the tick clock needs wrap-around handling;
- the helicopter is 48 px wide, not 44;
- scoring rules;
- press-to-bullet latency is 3 ticks;
- **the turret-stop bullet often makes the kill.**

The same days added a fitted chute-opening model and the probabilistic
free-fall planner (`physics/chute.py`, `physics/prob_plan.py`).

## 3. Accuracy fixes (09-29/30)

[Commit cba73d6](https://github.com/rpmaxwell/paratrooper/commit/cba73d6c300970dc2395c23e93a54841b31e6604):
- hits register where the bullet is **drawn** each tick, not along its path;
- positions are stamped with the tick they first appear;
- immediate retargeting after a miss;
- a per-column reach floor.

Interleaved A/B, 8 games each: mean 787 → 1,277. One lesson that kept
recurring: **only side-by-side comparisons are reliable.** An earlier
1,424 six-game mean had been luck.

## 4. The sim model and frame-level hitboxes (10-06)

[Commit f484926](https://github.com/rpmaxwell/paratrooper/commit/f484926cd6a5fcb23c17284a10b4e9c9737ca2d3)
added `sim/` and [the notebook](https://github.com/rpmaxwell/paratrooper/blob/f484926cd6a5fcb23c17284a10b4e9c9737ca2d3/notebooks/sim_spawn_drop.ipynb),
aiming at a Xiao-style solver: represent the game mathematically, then plan
against it.

- **Spawn and drop processes, fitted under censoring.** The bot's own
  shooting biases naive averages: helicopters that survived dropped 1.29
  troopers on average, but the true rate per crossing was 0.73. The model
  uses hazard estimation, per-lane spawn pools, burst clustering of drops
  and counterfactual landing times for shot troopers
  ([sim/fit.py](https://github.com/rpmaxwell/paratrooper/blob/f484926cd6a5fcb23c17284a10b4e9c9737ca2d3/scripts/paratrooper/sim/fit.py),
  [sim/sampler.py](https://github.com/rpmaxwell/paratrooper/blob/f484926cd6a5fcb23c17284a10b4e9c9737ca2d3/scripts/paratrooper/sim/sampler.py)).
- **Why helicopter shots missed.** Telemetry was too blurry (its ticks were
  ±1 often enough to smear 8–16 px), so
  [sim/collision.py](https://github.com/rpmaxwell/paratrooper/blob/f484926cd6a5fcb23c17284a10b4e9c9737ca2d3/scripts/paratrooper/sim/collision.py)
  fitted boxes from recorded frames, where bullet and target appear at the
  same instant. It found the helicopter box sits **+8 px** right, tested at
  drawn positions, with 96.9% of about 28k outcomes exact
  ([85dbd3f](https://github.com/rpmaxwell/paratrooper/commit/85dbd3f0624beef511cc18ab68e151507065e548),
  as an A/B switch).
- **Troopers.** The +8 px shift made trooper boxes *worse*; they are bigger,
  not shifted ([81d76da](https://github.com/rpmaxwell/paratrooper/commit/81d76da5be1a4c28097ed1bd7b78e285eaadd994)).
  The canopy/body split came from *which part* died
  ([428dc73](https://github.com/rpmaxwell/paratrooper/commit/428dc73e1c9bd2ddafa6d40115bf0846c32ea79f),
  `fitted_parts`).

All three boxes were later confirmed **exactly** by the disassembly
(section 9). The frame fitting was right, but it took about 10k labelled
outcomes to get there.

A/B, 40 games each: fitted helicopter box mean **1,211** vs 1,049.

## 5. Crush shots and calibrated p(hit) (10-06 to 10-07)

The user suspected crush logic "doesn't work at all".
[tools/review_crush.py](https://github.com/rpmaxwell/paratrooper/commit/4300ae8d6df85685416abfada5362cfcbba8e066)
showed the mechanic itself works: 97% of canopy kills over a landed trooper
crush it. But 86% of crush bullets were bets on *free-fallers*, planned at
p(hit) 0.56–0.94 and actually succeeding about 31–36%, flat across all
planned values. The narrow body box let the planner believe a bullet could
pass beside the body to reach the canopy.

That led to calibrating p(hit) from outcomes
([a8ebec4](https://github.com/rpmaxwell/paratrooper/commit/a8ebec4abdd362c420f8e7d4042a5fcca8e88a54),
`sim/phit.py`):
- **Replanning works.** Re-running the planner from the logged turret
  position reproduced 98% of logged plans, giving full plans even when the
  first bullet ended the job.
- **The stop bullet** makes about 40% of all trooper kills, and the planner
  ignored it.
- **Timing slip:** 17% of planned bullets spawn one tick late, both bullets
  of a pair together. That's an alignment error, not random jitter.
- **Free-fall kill shots** were calibrated on average (planned 0.89, got
  0.90) but had **no resolution**: every plan said about 0.9 and got about
  0.9. The remaining ~10% of failures weren't explained by chute timing,
  hitbox or execution data.
- **Crush shots** were fixed by `fitted_parts` plus the stop bullet:
  log-loss fell from 1.00 to 0.54 for free-fall crush bets, and from 2.73
  to 0.39 for canopy crush shots.

The result became `PARATROOPER_PHIT=calibrated`
([2b1fa2c](https://github.com/rpmaxwell/paratrooper/commit/2b1fa2ce60e4b54c07bec9fad9164e22abdd1131)).
The first version cost 6.5 ms per plan; summing over chute scenarios before
the pair search brought it back to 3.2 ms.

## 6. Decisions in points (10-07)

`sim/value.py` ([a8ebec4](https://github.com/rpmaxwell/paratrooper/commit/a8ebec4abdd362c420f8e7d4042a5fcca8e88a54))
values game states, V(round, tick, landed left, landed right). The first
fits were too pessimistic: the model gave 45% of games reaching round 3
against 55% observed. The causes:

- **Telemetry miscounts landed troopers.** It over-counted landings, and
  frames showed one falling body wiping out a whole two-trooper stack,
  which telemetry logged as at most one crush. Landed state is now fitted
  from frames (four on a side ends the game a median 367 ticks later;
  confirmed in 108 recorded games).
- **Mixing bot versions overstated the hazard.** Fitted on the current
  bot's games only, the model matched: 1,126 expected vs 1,176 observed,
  and the round-reach curve within about 3 points.

Two findings changed course:
- **The round-4 bomber phase killed about 70% of games that reached it.**
- **The points model reversed an earlier recommendation.** Removing a
  landed trooper is worth so much that a crush bet succeeding 31% of the
  time still beats a plain kill shot (214 vs 185 points, on a side holding
  two). The earlier advice to drop free-fall crush bets was wrong.

## 7. Round-4 bombers (10-07 to 10-08)

The user suspected a new launch angle. The data said otherwise: same fall
sequence, same release points. But 11% of round-4 bombs landed (against
under 1% in rounds 1–3), and many were first seen 3–9 ticks late. Round-4
planes drop **pairs** 2–4 ticks apart. Both bombs move with the plane, so
they fall stacked in one column, and the pair merges into a blob the
exact-size detector rejected. Neither bomb was seen until they separated.

- **The split detector**
  ([001e956](https://github.com/rpmaxwell/paratrooper/commit/001e956a39caf35e79ad7ffec739fee37563638b))
  first matched 129 of 285 merged blobs. The rest were **occlusion**: the
  bomb drawn second blanks its whole cell. With that handled it matched
  267 of 285. In replay, round-4 bombs first seen at k=1 went from 50 to
  75 of about 80.
- **A pair-aware plan**
  ([7de1d82](https://github.com/rpmaxwell/paratrooper/commit/7de1d828656945641430a07389273f046746341c),
  `PARATROOPER_BOMB_PAIRS`): planning the first bomb alone left the second
  out of reach in 33 of 42 cases.
- **The same commit includes** previously uncommitted turret/world work:
  key settling after Up, stuck-turn retry, bomb replanning.

The 10-game test of all new settings: mean 1,388 (±369 vs the 1,211
baseline, not significant). Crushes rose to 2.0 per game from 1.6 crush
bullets (was 4.5), and round-4 late sightings fell from 7 to 1.

A Docker crash left a stale X lock in each container's `/tmp`, killing
every restart ([33f2c5f](https://github.com/rpmaxwell/paratrooper/commit/33f2c5f1a384e0006dd90f9919bed2fe9f0aaf47)).

## 8. Reading memory (10-08)

Vision accuracy looked like it was plateauing, so the question became
whether to read the game's memory. **This departs from the PRD's founding
constraint;** the user chose to explore it. It turned out to be cheap. The
bot already runs as root in the emulator's container, so `/proc/<pid>/mem`
is readable, and the game's 64 KB segment is found by its code bytes in
about a second ([2bc23a1](https://github.com/rpmaxwell/paratrooper/commit/2bc23a175167ae294f54cceb94fb9e0cbf694d58)).
That's the opposite of the F19 experience that motivated the PRD.

Decoding by correlating memory with the vision stack
([d8235cc](https://github.com/rpmaxwell/paratrooper/commit/d8235ccc37aca05dcaac33fec8d12f0b362f8b4d)):
- **Score:** missed as a binary number, BCD and ASCII. It was found through
  the HI-SCORE: the HUD stores scores as digits, least significant first.
- **Tick counter:** found by looking for memory stepping +1 per tick.
- **Bullets, bombs, troopers:** coordinate matching found bullets and bombs
  (column units). Troopers were stored **per screen column** (y only).
- **Helicopters took longest.** No coordinate encoding matched. A 104-entry
  table moving one 8 px column per tick was a false lead (shot-helicopter
  debris). They turned out to be **a per-lane map with one byte per
  helicopter**.

## 9. Disassembly (10-08)

[tools/disasm.py](https://github.com/rpmaxwell/paratrooper/commit/5556ce6a3d92d474e6cc22d773a0c6f56f9ed253)
(capstone). The entry point is a trampoline into a code segment 0x2C40 into
the image. The first labels were wrong until it became clear the code sets
`DS = program + 0x11` paragraphs, so data offsets are memory offsets minus
0x110. What the code settled:

- **Bullets:** 30 one-pixel bullets from one spawn point, moved by a
  per-direction table, collision-checked once per tick at the drawn position.
- **Collision:** helicopters are 6 columns on the 8 px grid (the "+8 px"
  shift was grid alignment). Troopers match `fitted_parts` exactly. Bombs
  are ~24×28 px tested before the bomb moves; the box was **8×8** in the
  bot. That box explains 99.8% of 1.6k credited bomb kills, against 22%
  for the old one.
- **Randomness:** drops happen at 3.75% per column pass. All randomness
  comes from one LCG, `seed × 0x7781 + 0x64C9`. Chute opening uses a
  threshold table (indexing not yet confirmed).

The bomb box became `PARATROOPER_BOMB_BOX=code` with an earliest-intercept
planner ([4f5c064](https://github.com/rpmaxwell/paratrooper/commit/4f5c06466e781b677d93846749e15422aa288f02)).
The planned hit point moved from median k=29 to k=21 (of 33). Live it was
about 3 ticks earlier, since the old long bursts often scored early anyway.
The 2 of 10 games that reached round 4 both survived it, with no round-4
bombs landing.

## 10. Diagnosis from memory (10-08 to 10-09)

[4cd8c6d](https://github.com/rpmaxwell/paratrooper/commit/4cd8c6d2b4bc478b8b7b8344670f120392231cd4):
- `run.py --mem` records state bytes every frame;
- `perception/memstate.py` decodes them;
- `PARATROOPER_MEM_FEED` lets the world use them;
- `tools/mem_diag.py` labels every trooper's outcome.

The first diagnosis was wrong: killed troopers came out as "misses". Two
causes:
- bullets sit on a fixed lattice, which made tick alignment ambiguous, so
  it now aligns on trooper drops;
- "killed" had to mean killed by *our* bullet, so it now uses the world's
  attribution.

Results over 10 games (450 troopers, mean score 1,608):
- **Vision isn't the bottleneck.** It saw the trooper in 97.6% of the
  frames where memory has one.
- **56% of the 72 landings were never engaged,** 35 of 40 of them while 1–6
  other troopers were falling, at the outer columns where the reach floor
  comes early.
- **Each engagement holds the gun about 18 ticks** against about 54 in the
  air. The planner waits for the best p(hit) instead of shooting early.

The binding constraint is **gun time**. That's the starting point for
expected-value targeting.

---

## Lessons that kept coming back

1. **Check the ground truth before the model.** Wrong labels (logged hits,
   landed counts, tick stamps) caused more lost time than wrong physics.
2. **Only side-by-side comparisons count.** Scores swing by about 550 game
   to game, so 10-game means mislead.
3. **Frames beat telemetry; memory and code beat frames.** Each step down
   that ladder turned a fitted estimate into a fact.
4. **Watch it play.** Most breakthroughs started from something the user saw
   on VNC: clean misses, late bomb hits, the round-4 double bomb.
5. **Price decisions before judging them.** Crush bets looked bad by hit
   rate and good by points.
