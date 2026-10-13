# Plan: Deterministic Accuracy, Then Learned Prioritization

## Context

RL alone (PPO, various reward-shaping attempts) converged to a policy that
is good at shooting helicopters and effectively ignores paratroopers,
because a hit's reward didn't distinguish target type and the "aim
correctly" motor-control problem is hard for trial-and-error to solve on
its own. Investigation showed Paratrooper's physics are simple and
deterministic (constant-velocity rotation, constant-velocity fall, fixed
rotation limits) -- aiming is a solvable geometry problem, not something
that needs to be learned. This plan has two sequential phases:

1. **Make the gun itself accurate** -- a deterministic aim-and-fire solver
   that reliably hits whatever it's told to target.
2. **Decide what to target** -- a prioritization layer (rule-based and/or
   learned) implementing the risk-avoidance/scoring strategy already
   worked out from the game's actual mechanics.

Phase 2 only matters once Phase 1 is reliable -- a perfect priority list
firing through an inaccurate gun still misses. Do not start Phase 2 tuning
in earnest until Phase 1 hits its accuracy target.

---

## Phase 1: Deterministic Accuracy (target: ≥80% hit rate, one-shot)

### Status update (2026-09-18): Stages 1.1-1.4 done, target not yet met

Ran the full calibration pass against the idle `paratrooper-1` container
(the other, `paratrooper`, was mid-training the whole time and was never
touched). New scripts: `calibrate_rotation.py`, `calibrate_fall_speed.py`,
`capture_fall_traces.py`, `calibrate_helicopter.py`,
`calibrate_latency.py`, `calibrate_bullet_speed.py`,
`calibrate_hitbox.py`. All calibrated constants below are now live in
`aim_solver.py`.

**What was found and fixed:**

- **Rotation speed**: tightened from 121-129 (3-4 trials) to **128.23
  deg/sec, std 1.07** across 18 clean full-range sweeps (2 detection-
  glitch outliers excluded from 20). Well within the 1-2% validation gate.
- **Paratrooper fall speed is two-phase, not constant** -- this was the
  biggest surprise and likely the single largest pre-existing error
  source. A fast free-fall phase right after release at **~144.5 px/s**
  (matches the old single-speed assumption, which was apparently measured
  during this phase), transitioning to a much slower canopy-open descent
  at **72.92 px/s (std 0.36, n=17 -- the tightest measurement of the whole
  pass)**, with the transition happening around **y=422** (std ~14px).
  Since target selection prefers the freshest/topmost blob, most
  engagements were straddling or entirely inside the free-fall phase,
  where the old model was off by very close to 2x for the back half of
  the flight.
- **Helicopter speed**: 144.78 px/s (std 0.57, n=32), two fixed altitude
  bands (y=219.2 moving left, y=243.2 moving right) -- not previously
  measured at all.
- **Latency re-isolated cleanly**: 52.7ms (std 18.3ms, n=31), down
  slightly from the old indirectly-inferred 61ms. Confirmed reducing
  `actions.py`'s xdotool `--delay` from 50ms to 15ms does **not** reduce
  it (52.5ms, n=25) -- the dominant cost is subprocess spawn / X11
  delivery, not that flag, so the delay was left at the proven-reliable
  default (now overridable via `PARATROOPER_KEY_DELAY_MS` for future
  experiments).
- **Bullet travel confirmed hitscan, not just assumed**: frame-diffed a
  30-frame rapid burst in a clear angular corridor after firing and found
  zero pixel change at any point -- no projectile sprite is ever
  rendered. `LEAD_TIME_S` removed entirely.
- **New constant, previously invisible**: `DECISION_LATENCY_S` (31.8ms,
  std 3.9ms) -- the Python-side time between grabbing a frame and issuing
  the *first* action for it (detection, clustering, solving). Distinct
  from `LATENCY_OFFSET_S` (which only covers time *inside* the
  `do_action()` call). Folded into `solve_intercept()` internally.
- **Real bug found and fixed**: `solve_intercept()` capped its search
  window only at `GROUND_Y`, but the turret's own reachable range
  (25.71-155.10 deg, not 0-180) means an off-center falling target goes
  geometrically unreachable *before* it lands. The old code kept
  searching past that point and returned spurious "solutions" -- a barrel
  pinned at its rotation limit, matched against a target angle that had
  already swept past what the turret could ever point at -- producing
  aim errors of 20-100+ degrees on a meaningful fraction of shots. Fixed
  via `_max_reachable_t()`, solved in y-space to avoid atan2 wraparound.

**Where it stands:** with all of the above, measured hit rate is still
**~20-25%** against isolated falling paratroopers (several 30-100 shot
runs after the fixes), not meaningfully better than the ~28% pre-fix
baseline despite the rotation/fall/latency constants being far tighter
now and a real solver bug being fixed. Diagnostic findings from this
session:

- Aim error (actual barrel angle vs. model-predicted angle at fire time)
  shows **no correlation with hit/miss** on shots where it's small (0-8
  deg) -- hits and misses both occur across that whole range. This says
  the rotation/timing model is *not* the bottleneck; something else is.
- A deliberate controlled-offset hitbox sweep (`calibrate_hitbox.py`,
  offsets -10 to +10 degrees off the solved intercept, 8-10 trials per
  offset) showed **no falloff pattern at all** -- hit rate stayed flat
  (~20-40%) across the entire tested range, including at +-10 degrees.
  Two explanations remain live and weren't distinguished before this
  session ran long: (a) the real hitbox is wider than +-10 degrees, so
  the sweep never left the hittable zone, and the ~20-40% flat rate
  reflects some *other* uncontrolled noise source; or (b) target
  identity/position is getting corrupted between capture and fire often
  enough (wrong blob, multi-trooper confusion, a one-frame position
  lag against the game's actual internal tick) to swamp a real but
  narrower hitbox signal.
- The score-delta hit heuristic (`fire_cost = -1 if prev_score > 0`)
  was spot-checked and looks correct: idle score never drifts on its own
  (30s of no-op action, score held exactly at 0), and a "delta == 0 with
  prev_score > 0" case is consistent with a real 1-point kill netting
  against a 1-point ammo cost, not an artifact.

**Recommended next step for whoever picks this up**: don't re-run more
blind offset sweeps -- first distinguish (a) vs (b) above directly, e.g.
by overlaying the solved aim point on a saved frame at fire time and
visually confirming it lands on the sprite the score-delta says was hit,
across a dozen hits and misses. That will show directly whether it's a
targeting/tracking bug or a genuine geometry/hitbox constant still to be
measured, without burning more live-game trial budget guessing between
them.

### Status update (2026-09-21): two real bugs fixed, one hypothesis tried and reverted

Followed the recommended visual-inspection approach above, prompted by
the user watching over VNC and reporting two distinct failure modes:
"clean hitbox misses on an isolated, unhurried shot" and "a noticeable
delay between the paratrooper dropping and the gun reacting."

**Bug found and fixed -- ground-debris false targeting (the delay).**
`threat_points()` excludes only `y>=555`, but a landed walker's legs
merge into the ground line and render cyan pixels as high as y=553-554 --
just under that cutoff. `find_target()` across the practice/calibration
scripts was picking these up as if they were a still-falling target, and
`solve_intercept` correctly rejected them (nothing to intercept, they're
static), but the loop had no way to know that and just spun -- up to
9.6s and 167 rejected iterations observed in one case
(`diagnose_reaction_latency.py`) -- while a real paratrooper waited
unengaged. Fixed with a `TARGET_Y_MAX=540` cutoff in target selection
(not in `threats.py` itself, which the live-training container also
depends on and wasn't touched). Confirmed via before/after
`diagnose_reaction_latency.py` runs: multi-second stalls -> gone
entirely, only much smaller (~170-400ms) unrelated `detect_barrel_angle`
glitches remain. This did not move the aggregate hit-rate number much
(~24-27% before and after) -- it fixed responsiveness/smoothness, not
per-shot accuracy on shots that already had a valid solution, which
matches the user's framing of it as a separate issue from the hitbox
misses.

**Bug found and fixed -- merged multi-trooper blobs (part of the hitbox
misses).** Two troopers falling close enough vertically to land within
`cluster()`'s 15px radius get merged into one blob whose centroid sits in
the empty space between them, not on either trooper (one observed case:
a single "blob" 24px wide x 48px tall, ~2x taller than any real single
trooper's ~8x11px footprint). `characterize_sprite.py` confirmed the real
single-trooper shape is otherwise extremely consistent and symmetric
(identical spread-eagle pose -- arms bar, narrow torso, split legs --
across every clean sample), so the fix was rejecting oversized blobs
rather than hunting for a different reference point within a good one:
`MAX_BLOB_HEIGHT=20` added to target selection. Validated with a
purpose-built script, `calibrate_isolated_miss.py`, that only fires when
there's a single unambiguous blob both at aim-decision time and again at
the pre-fire instant (no relocate-by-distance step that could latch onto
a different trooper): measured miss-angle std dropped from **38.3deg**
(with an earlier, flawed relocate-based measurement contaminated by
exactly this merging problem) to **6.67-7.11deg** (two independent runs)
once blob identity was made unambiguous.

**Structural finding -- the game runs on a discrete ~50ms tick, not
continuous time.** Rapid polling (70+Hz) shows the barrel angle and the
falling target's y-position both hold *exactly* steady for a run of
frames, then jump all at once -- confirmed independently for both barrel
rotation and target fall, with a measured period (~50ms median) close to
the classic 18.2Hz PC BIOS timer rate common in 1982 DOS games. This
solidly explains where `LATENCY_OFFSET_S`'s 18.3ms std comes from (phase
uncertainty in when our do_action() call's ~50ms blocking delay
happens to land relative to the game's own tick boundary, not raw
subprocess jitter -- confirmed separately that an isolated xdotool call
has only ~1.4ms std). **A first attempt at exploiting this -- replacing
solve_intercept's continuous root-find with a discrete search over tick
indices, plus a mid-tick fire-time safety margin -- was implemented and
empirically tested, then reverted**: it made the measured miss-angle std
*worse* (6.67 -> 12.76deg on the same clean single-target test), most
likely because the ~50ms tick-period estimate carries too much
uncertainty (+-5-10ms from our own polling resolution) to reliably pick
the *correct* discrete tick, and/or there's a phase offset between "we
press rotate" and the game's tick boundary that isn't just the fixed
`LATENCY_OFFSET_S` constant. The continuous model, imprecise as it is,
apparently averages out better than a rigid but possibly-mis-aligned
discrete grid. `TICK_PERIOD_S` is left in `aim_solver.py` for reference
(unused, `TICK_SAFETY_MARGIN_S=0.0`) -- retrying this would need a much
tighter, phase-aware tick measurement first, e.g. correlating many
independent rotate-start timestamps against the tick clock directly
rather than inferring the period from free-running rotation traces.

**Remaining structural constraint, independent of any of the above**:
the target's angular half-width at typical shot distance (median ~270px
from the pivot) is only ~0.85-1.7deg (`calibrate_isolated_miss.py`
distance/sprite-size analysis) -- i.e. the whole hitbox is only
~1.7-3.4deg wide. That is comparable to or smaller than the combined
residual noise floor (~6.67-7.11deg std) even after both bugs above were
fixed, which is why even a same-frame, ground-truth "dead-on" shot
(measured miss <1deg) only hits ~11-14% of the time in the latest clean
runs. Getting materially past this now most likely requires tightening
that noise floor further -- the tick-phase measurement above is the most
promising remaining lead -- rather than more target-identification work,
which is now believed to be close to exhausted as a lever.

### Error-source audit and systematic elimination plan (2026-09-22)

Repeated end-to-end hit-rate/offset-sweep reruns have dead-ended: constants
are tight, two real bugs were found and fixed (ground-debris false
targeting, merged multi-trooper blobs), yet hit rate against a same-frame
"dead-on" shot is still only ~11-14%. That's a sign the remaining error is
somewhere an end-to-end test can't localize, not that there's no error left
to find. Before running more full-loop trials, decompose the pipeline into
independently-checkable error sources and measure each one in isolation.

**The full list of candidate error sources** (the original 5-item list was
incomplete -- it only covered modeling, not perception/instrumentation):

1. Paratrooper path model (free-fall + canopy-phase fall rate, transition
   point).
2. Projectile path model from a given barrel angle.
3. Gun-angle *prediction* model (rotation speed, limits, latency).
4. Projectile speed/travel-time model.
5. Intercept-solving math (`solve_intercept`) given the above.
6. **Perception/tracking**: measuring where the target actually is on
   screen (blob detection, clustering, identity across frames) --
   independent of #1's motion *model*. The merged-blob bug lived here.
7. **Current-angle measurement**: whether `detect_barrel_angle` accurately
   reports where the barrel *is right now*, independent of #3's
   prediction of where it *will be*.
8. **Pixel-to-angle geometric calibration**: the turret pivot's pixel
   location and the px-to-degree scale factor underlying #1, #3, #6, and
   #7 alike. Never directly isolated -- only rotation *speed* has been
   validated, not the coordinate transform itself.
9. **Tick-phase timing model**: the game runs on a discrete ~50ms tick,
   confirmed directly by polling. A first discrete-solver attempt made
   things worse, most likely due to phase (not just period) uncertainty,
   per the 2026-09-21 note -- this is unresolved, not ruled out.
10. **Ground-truth validity**: the score-delta hit detector (spot-checked,
    looks fine) and the true hitbox width (the +-10 deg offset sweep
    showed no falloff at all -- inconclusive, not "wide hitbox confirmed").

**Why more sweeps won't help**: hit-rate and offset sweeps are end-to-end
tests -- they show total error is too large but can't attribute it across
10 stacked sources. The plan below isolates each source with its own
zero-or-minimal-dependency measurement, ordered so later stages can trust
earlier ones' ground truth.

- **Stage A -- ground-truth instrumentation.** Extend the visual-overlay
  harness (`capture_annotated_shots.py`, already built and run once
  2026-09-21) into the standard diagnostic tool. Separately, re-run the
  offset sweep (`calibrate_isolated_miss.py`'s method: unambiguous
  single-blob-only trials, no relocate-by-distance step) with a much wider
  offset range than +-10 deg, since the existing sweep never left the
  hittable zone and therefore never measured a real falloff -- this is a
  prerequisite for interpreting every other stage's results against a
  known hitbox bound.
- **Stage B -- instrument-only checks (source #7, #8).** Measure
  `detect_barrel_angle` noise on a *stationary* barrel (isolates detector
  noise from rotation-model error). Separately, regress detector readings
  at several known/commanded angles (including the two hard limits,
  25.71/155.10 deg) to fit the pivot pixel location and px-to-deg scale
  independently, rather than assuming the current transform is exact.
- **Stage C -- tick-phase timing (source #9).** Re-attempt tick
  measurement with a phase-aware method: timestamp many independent
  rotate-*start* commands against the polling loop directly, rather than
  inferring period alone from free-running rotation traces (what the
  reverted attempt did).
- **Stage D -- re-audit path models (source #1) with better ground truth.**
  More independent fall traces across waves/difficulty levels (currently
  only one clean isolated trace backs the "no acceleration" assumption per
  the confidence table below). Low priority for #2/#4 -- hitscan is
  already well-confirmed by frame-diffing.
- **Stage E -- intercept math as pure logic (source #5).** Unit-test
  `solve_intercept`/`_max_reachable_t` against synthetic, hand-computed
  scenarios (near turret limits, near the free-fall/canopy transition,
  near ground) with no live game involved, the way the reachable-range bug
  was originally found by inspection rather than by a sweep.
- **Stage F -- perception under realistic conditions (source #6).**
  Re-validate the merged-blob fix in busier multi-trooper scenes, not just
  the isolated single-blob scenes `calibrate_isolated_miss.py` currently
  restricts itself to.
- **Stage G -- reconcile.** Combine each stage's independent std/bias in
  quadrature and compare to Stage A's hitbox bound. A predicted total that
  matches the observed ~6.67-7.11 deg noise floor confirms the sources; a
  mismatch means something is still unmeasured.

#### Stage A results (2026-09-22): the ground-truth measurement itself is unreliable, not just the hitbox

Ran `calibrate_hitbox_wide.py` -- same unambiguous-single-blob methodology
as `calibrate_isolated_miss.py`, but offsets widened to
+-2/4/6/8/10/12/15/20/25/30/40 deg (23 values, 6 trials each, 138 total,
82 with an unambiguous target at both decision- and fire-time) against the
idle `paratrooper` container (`paratrooper-1` was mid-run on a long
`practice_deterministic_aim.py` job and was left untouched).

**Result: still no usable falloff pattern, even across a much wider range
than the hitbox could plausibly be.** Hit rate by |measured true-miss
angle|: 0% at [0,1) deg (n=4), 20% at [1,2) (n=5), 0% at [2,7) (n=20), 11%
at [7,10) (n=9), 0% at [10,15) (n=15), 7% at [15,20) (n=15), 0% beyond
that (n=14). Overall clean hit rate ~4%. The single most important data
point: **a same-frame measured miss under 1 degree still missed all 4
times.** That rules out "the sweep just never widened enough" -- the
signal doesn't correlate with this metric at any offset, not just within
+-10 deg. This means the true-miss-angle ground truth itself (used here
and in the earlier `calibrate_isolated_miss.py` 6.67-7.11 deg std result)
is not a reliable predictor of hit/miss, which is a stronger and more
useful negative result than "hitbox is narrow."

**Hypothesis tried and NOT confirmed**: `actions.py` documents that the
fire key ("Up") both stops the turret's rotation and fires in one
keypress -- so if rotation is still ongoing when `do_action(3)` is called,
the barrel keeps moving for another ~LATENCY_OFFSET_S (~53ms, ~6.76 deg at
128.23 deg/s) before the keypress actually lands and the turret truly
stops. Every ground-truth script (`calibrate_isolated_miss.py`,
`calibrate_true_miss_angle.py`, `calibrate_hitbox_wide.py`) captures
`prefire_frame` *before* calling `do_action(3)`, so `actual_barrel_at_fire`
may be read before the barrel has actually finished rotating -- a
plausible several-degree stale-read bug in the ground truth itself, not
just in the model being tested. Tested by inferring rotation direction
from `sign(actual_barrel_at_fire - barrel0)` in the 82 clean trials and
adding/subtracting the predicted 6.76 deg correction: **this made the
scatter worse, not better** (raw std 15.95 deg -> 16.44 deg (add) / 17.67
deg (subtract)), so the hypothesis as tested is not confirmed. Caveat: the
sign-of-net-rotation proxy can't distinguish "still rotating at the exact
prefire-frame instant" from "already finished rotating several frames
earlier" (`b_direction`/`b_hold_s` weren't logged in this script), so this
is a weak test of the hypothesis, not a clean refutation of it.

**Recommended immediate next step (supersedes starting at Stage B as
planned above)**: instrument the fire event directly rather than
inferring it. Log `b_direction`/`b_hold_s` per trial (cheap, already
computed) so trials can be split by "was rotation still active going into
the fire call" vs. not; and/or burst-capture frames spanning the
`do_action(3)` call itself (same technique as the bullet hitscan check) to
directly observe, frame by frame, exactly when the barrel visually stops
moving relative to when the fire key was issued -- rather than assuming
the pre-fire frame already shows the final state. Until the ground-truth
capture timing is itself verified, no offset-sweep or hitbox-width
conclusion drawn from `true_miss_deg` (from any of the three scripts using
this pattern) should be trusted as more than suggestive.

#### Direct fire-tick instrumentation (2026-09-22): real bug found, but not the dominant one

Built `calibrate_fire_tick.py`, which does NOT infer or correct the fire
timing -- it observes it. `actions.py` documents that the fire key ("Up")
both stops rotation and fires in one press, so a fast enough barrel-angle
burst capture spanning the keypress must show a clear signature: readings
that change tick-to-tick while rotating, then an abrupt switch to a
constant reading once stopped. That transition tick *is* the real fire
moment, directly observable, no latency constant involved. (Two
implementation notes worth keeping: (1) `threats.detect_barrel_angle`
scans a huge 1024x584 strip and measured at ~12.6 ms/frame in this
container -- almost certainly x86-on-arm emulation overhead scaling with
array size, not real CPU cost -- which is too slow to resolve a 50ms tick;
cropping tightly to the known pivot+radius before the same mask/where
logic cut that to ~0.5 ms/frame and was verified to return identical
results. (2) The first version only burst-captured *after* calling the
fire keypress and got "no transition seen" on most trials; monitoring a
window before the intended fire moment too, not just after, was needed to
actually see what was happening.)

**Result, 66 trials (offsets +-2 to +-10, same range as
calibrate_isolated_miss.py): confirmed, directly, that the barrel usually
finishes rotating and stops changing *before* `do_action(3)` (fire) is
even called** -- not during its dispatch latency as the current model
assumes. Across the 46 trials where the transition was located: fire-stop
time relative to the fire keypress call, mean=-12.6ms, std=18.3ms,
range -55ms to +26ms, and 31/46 (67%) stopped early (negative). The std
here (18.3ms) matches `LATENCY_OFFSET_S`'s own previously-measured std
(18.3ms, from `calibrate_latency.py`, a completely independent
measurement) almost exactly -- strong evidence this is the same
underlying subprocess-spawn/X11-dispatch jitter, not a new, separate noise
source. This means the current `sleep_s = b_hold_s - LATENCY_OFFSET_S`
formula is over-sleeping by roughly 12.6ms on average before firing --
plausibly because `LATENCY_OFFSET_S` was calibrated for the *rotate*
key's dispatch-to-effect latency and is being reused for the *fire* key's
without that being separately verified.

**This confirms a real, previously-unknown timing bug -- but, matching
exactly the pattern from the reverted discrete-tick attempt and the
untested fire-latency correction before it, fixing the measurement did
NOT shrink the aim-error noise floor**: std on this run's `true_miss_deg`
was 6.23 deg (n=18 clean, both barrel and target trackable at the located
tick) -- essentially the same as `calibrate_isolated_miss.py`'s earlier
6.67-7.11 deg over the same +-10 deg offset range, not meaningfully
better. Per the framing that prompted this work: a correction for a real
effect should only ever help once applied correctly, so a fix that
doesn't move the number isn't evidence the effect is fake -- it's evidence
this isn't the dominant term. The ~12.6ms mistiming is real (worth fixing
in `aim_solver.py` on its own merits) but small relative to whatever is
actually producing the 6-7 deg noise floor.

**Two reliability caveats before trusting this method further**: (1) the
hit/blob-disappearance cross-check (does the target blob vanish at the
same tick the barrel-stop transition was detected, for trials that hit)
only had 2 qualifying hits this run -- one matched exactly (0ms
difference, a strong positive signal for the method), one was off by
217.9ms (either an animation delay or a misidentified transition tick;
n=2 is nowhere near enough to tell which). (2) Only 18/66 trials were
"clean" (fire tick located AND target unambiguously trackable there) --
the rest were lost to stretches of invalid barrel-angle readings (~None
for the whole window, cause not yet diagnosed -- possibly frame capture
landing mid-redraw) or "no transition seen" (barrel already constant more
than 150ms before the intended fire moment, some of which are legitimate
turret-limit stops and some of which may mean the real stop happened
earlier than this run's monitoring window reached). Before using this
script's output to draw further conclusions, both issues need
attention: a larger batch to get the cross-check n up, and either a wider
pre-fire monitoring window or a diagnosis of the all-invalid-reading
stretches.

#### Target-free gun-only test (2026-09-22): the real dominant error source, most likely found

Prompted by watching the gun over VNC looking "much more accurate" than
the aggregate numbers suggested -- to check that directly, target
detection was removed from the test entirely. Two new scripts:

- **`discover_firing_angles.py`**: sweeps the turret full-range with the
  same fast pivot-cropped polling as calibrate_fire_tick.py and records
  every genuinely distinct angle reading encountered (collapsing
  consecutive near-identical readings). **The turret can only ever be
  observed resting at 19 fixed angles** between the two hard limits
  (25.71 to 155.10 deg) -- confirmed byte-for-byte identical across two
  independent full sweeps, so this is real, repeatable geometry (most
  likely quantization of the rendered barrel-tip's integer pixel position
  at a fixed radius from the pivot), not sampling noise. Spacing between
  consecutive angles is NOT uniform -- 3.45 to 13.73 deg, mean 7.19 deg --
  so this is a real, empirically-measured list, not a computed grid.
- **`test_gun_open_loop_accuracy.py`**: with no target on screen at all,
  repeatedly commands the turret from its current resting angle to a
  chosen angle from that discovered list (varying tick-distance, both
  directions), using the *exact same* rotate/sleep/fire control path as
  real play (`ROTATION_SPEED_DEG_S`, `LATENCY_OFFSET_S`, `execute_intercept`'s
  formula) -- the only thing removed is target detection. Ground truth for
  where it actually landed uses the same validated fire-tick burst-capture
  method. Handles game-over the same way every other script here does
  (detect + press space + keep going) -- with nothing being shot, the
  episode ends on its own before long, so this runs across as many
  episodes as it takes, per the plan for this test.

**Result, 100 trials: the gun itself is good.** 86/100 clean
(fire tick located). Of those, **73% landed on exactly the intended
discrete tick, 97% within +-1 tick, 98% within +-2**. Two large outliers
(one -5 tick, one +16 tick / 113.5 deg) are very likely a distinct bug --
one of them shows the barrel moving the *wrong direction* for several
real tick-transitions before firing, which no timing-jitter explanation
accounts for; most likely a stale/incorrect assumed starting angle
carried over from the previous trial (this test's own bookkeeping, not
necessarily a game-input bug) -- flagged for follow-up, excluded from the
headline stats below. Excluding those two: n=84, mean error=+0.19 deg,
std=3.11 deg -- tightly clustered around zero, an order of magnitude
tighter than every target-based test this whole investigation has run.

**This reframes the entire investigation.** The real hitbox was
separately estimated at only ~1.7-3.4 deg wide (calibrate_isolated_miss.py's
distance/sprite-size analysis) -- *narrower than the 7.19 deg average gap
between adjacent achievable discrete angles*. That means even a
hypothetically perfect execution (always landing on tick_offset=0, the
73% case) is not sufficient on its own: `solve_intercept` does continuous
root-finding as if any angle were reachable, then whatever discrete tick
the timing naturally lands nearest to is what actually fires --
`solve_intercept` has never known about this 19-point grid, so the
*intended* continuous target itself is generically up to ~half a tick
(3.6 deg) away from anything the turret could ever actually stop at, an
error comparable to or larger than the entire hitbox width, structurally
independent of how tight the rotation-speed/latency/fall-rate constants
get. This is very likely the dominant term the whole error-source audit
above was looking for -- and notably, it's the mechanism the reverted
discrete-tick attempt (2026-09-21) was trying to exploit, but that
attempt computed a *uniform* grid from `TICK_PERIOD_S` (assumed ~6.41 deg
spacing) rather than using the real, non-uniform, empirically-measured
19-point list now available in `/captures/firing_angles.json` -- worth
retrying with the real grid before concluding the idea itself doesn't
work.

**Recommended next step**: modify `solve_intercept` (or add a
post-processing step) to snap its continuous solution to the nearest
angle in the real measured discrete grid *before* computing hold_s --
i.e. solve for the intercept the way it already does, then choose
whichever of the 19 real achievable angles is closest to the target's
predicted angle at the time the turret could reach it, and aim for that
exact discrete angle instead of the unreachable continuous ideal. Given
the numbers above, this is the most promising lever found in this entire
investigation -- but per the standing rule this session has been
enforcing throughout, it should be validated against live hit rate before
being trusted, not assumed to work just because the reasoning is sound.

#### Implemented and live-validated (2026-09-22): inconclusive, not a clear win

`solve_intercept` was rewritten to scan `FIRING_ANGLES_DEG` (the 19
measured angles) directly -- for each, compute the time to rotate there
from the current barrel angle and where the target would really be at
that moment, and return whichever minimizes the miss. Only 19 candidates,
so this replaced the old continuous bisection outright rather than
snapping its output after the fact. Synthetic sanity checks (static
target, already-aligned case, unreachable case) all behaved correctly.

**Live validation (`calibrate_isolated_miss.py`, 88 shots, same harness
used for the earlier 6.67-7.11 deg baseline) did not show a clear
improvement**: overall hit rate 15.9% (14/88), unbiased offset=0 subset
25% (2/8, too small to read much into). `true_miss_deg` std was 10.23 deg
excluding two obviously-bogus readings (barrel angle -159.44 deg -- outside
the physical range entirely; `calibrate_isolated_miss.py` never applies
`valid_barrel_angle()` to the *prefire* reading, only the initial
decision-time one, a pre-existing gap in that script unrelated to this
change) -- worse than the 6.67-7.11 deg baseline, not better. The
bin-by-bin hit-rate-vs-miss-angle table is still noisy and non-monotonic
(0% at <1 deg, 29% at 3-4 deg, 0% beyond), the same pattern seen all
session.

**Read on this**: not a refutation of the discrete-grid theory -- the
measurement this comparison relies on (`true_miss_deg` via the prefire-frame
method) is the same one already shown to be contaminated by the fire-timing
bug found earlier today, and n=88 (52-54 "clean") is small given how noisy
every hit-rate measurement in this file has been. But it means the fix
should not be presented as validated. Before trusting or reverting it:
re-run a live comparison using the fire-tick direct-observation method
(the one thing this session validated as trustworthy) against *real*
targets rather than `calibrate_isolated_miss.py`'s known-contaminated
prefire-frame ground truth, at a larger sample size, and fix the missing
`valid_barrel_angle()` check on the prefire reading so obviously-impossible
readings can't corrupt the comparison either way.

#### Bomb shooter added and 30-minute live session (2026-09-22): strong result on the metric that actually matters

Per-shot hit-rate stats kept coming back ambiguous/noisy all session (see
above), but survival had visibly and dramatically improved watching over
VNC -- consistently reaching the bomb-drop stage instead of losing in the
first round. That's a much harder-to-fake aggregate signal (it requires
many consecutive correct decisions, not one lucky shot), so the natural
next validation step was survival/score over many full episodes rather
than another isolated per-shot measurement, plus removing the
last un-handled lethal threat (bombs) so episodes could run long enough
to actually measure that.

**Bomb characterized empirically** (`hunt_bomb.py`, plays paratroopers
normally while scanning every frame for small non-helicopter, non-walker
WHITE blobs): a small filled-circle sprite, bounding box exactly 7x7px /
~48 filled pixels every time, moving at constant velocity in both x and y
(~146, 151 px/s in the one fully-captured 1.7s/16-point trajectory) --
confirmed as a straight line (constant dy/dx ratio across consecutive
points), matching the description of watching it over VNC. A separate,
similarly-sized but *purely horizontal* white blob was also found near
the helicopter altitude bands and correctly identified as a helicopter
fragment, not a bomb -- excluded via a same-shape-profile-plus-both-axes-
moving check (`BOMB_MIN_AXIS_SPEED`) in the detector.

**`solve_bomb_intercept` added to aim_solver.py**: same discrete
19-angle-grid scan as the rewritten `solve_intercept`, generalized to a
target moving linearly in both x and y (estimated fresh from two
consecutive detections -- no calibrated fixed bomb velocity yet, only one
directly-observed trajectory so far) instead of falling straight down
from a fixed x.

**`play_with_bombs.py`**: paratroopers (priority, existing logic) plus
bombs (new logic), nothing else -- no helicopter targeting. Run live for
the full 1800s against the idle `paratrooper` container, restarting on
death same as every other script here (no way found to reliably extend
an episode beyond shooting the threats down).

**Result**: 30 completed episodes, final scores
`[74, 99, 14, 29, 39, 16, 42, 181, 48, 36, 96, 145, 71, 50, 56, 18, 52,
113, 128, 68, 55, 12, 71, 66, 37, 57, 14, 79, 53, 64]` -- mean 62.8,
median 55.5, **new all-time high score 181** (beating the previous
168 record). 127 bomb-intercept attempts, 29 confirmed hits (score jumps
of a consistent, repeated +29/+30 -- almost certainly a fixed bomb-kill
bonus, distinct from the smaller per-paratrooper/walker point values) for
a 22.8% bomb hit rate -- lower than ideal (the detector's false-positive
filtering likely still costs some attempts, and bomb geometry/timing
hasn't been tuned at all yet, unlike the paratrooper solver), but every
confirmed hit is real signal: destroying even ~1 in 4 bombs measurably
extends survival, and the mean/median scores here are well above anything
seen in this file's per-shot accuracy tests, consistent with the original
VNC observation that prompted this whole detour.

#### Target identification during clutter (2026-09-23): two real bugs found and fixed, live-validated

User report watching over VNC: the gun keeps shooting at "the spray" after
a hit, many paratroopers get ignored entirely, easy shots are missed, and
all of this gets worse with clutter. Built `diagnose_target_id.py` to
instrument every raw cyan blob in the target y-range (not just the one
selected), log "ignored" events where visible threats exist but every
blob got rejected, and burst-capture frames after each shot to see
directly what happens post-hit rather than guessing.

**Bug 1 -- oversized-blob rejection was discarding entire real target
clusters, not just merged noise.** `find_paratrooper_target`'s existing
`MAX_BLOB_HEIGHT=20` filter (added earlier to reject a merged-centroid
artifact) rejects the *whole* blob when several vertically-stacked
troopers chain together under `cluster()`'s radius=15 grouping. Visually
confirmed: a frame with two live clusters of 4-5 stacked troopers each
(h=45, h=41) produced zero targets -- the gun did nothing despite several
killable troopers on screen. This is exactly "many paratroopers ignored
entirely," and gets worse with clutter because chaining is more likely
the more troopers are on screen at once. **Fix**: `split_oversized_blob`
re-segments an oversized cluster by y-gap (12px, between the single-
trooper's own ~11px point spread and the 15px cluster radius that caused
the merge) instead of discarding it outright, recovering the individual
troopers.

**Bug 2 -- a post-hit debris effect gets picked up as a fake target ("the
spray").** Visually confirmed in a saved frame: after a hit, a scattered
burst of magenta (+ some cyan) pixels appears above the turret and
lingers; the cyan fraction of it clears the height filter and gets
selected -- directly reproducing "the gun is constantly shooting at the
spray after a hit." Measured the distinguishing signal directly: a real
trooper's cyan footprint is dense (55-65% of its own bounding box filled,
a solid ~8x11 sprite) while debris fragments from the same frame measured
4-29% (scattered dots with gaps) -- a wide, clean margin. **Fix**:
`is_real_trooper_shape` rejects any candidate blob under 35% fill density
or under 15 total pixels.

Both fixes applied to `play_with_bombs.py`'s `find_paratrooper_target`.

**Live-validated, 600s session (merge-split fix; the density/spray fix
landed after this run started, so it reflects bug 1's fix only)**: 253
shots, **85 hits = 34% hit rate** -- and critically, **hit rate now rises
with clutter instead of falling**: 22% at clutter=1, 24% at clutter=2, 28%
at clutter=3, **48% at clutter=4+**. That inversion is strong direct
evidence the fix addresses the reported symptom, not just a proxy metric.
"Ignored" events dropped to 12 over the full 10-minute session (previously
this triggered on essentially any busy screen) -- the residual cases are
extreme piles (h up to 164-228) where troopers are packed too tightly for
even a 12px y-gap to separate, or a near-miss just over the height
threshold (h=23 vs the 20px limit) -- both rare edge cases, not the
dominant failure mode anymore.

#### Bomb performance investigation (2026-09-23): two structural bugs found by code inspection, one confirmed fixed, one unproven

User report watching live: the gun usually doesn't move or fire at a
bomb at all, and when it does it's often too late (blown up regardless).
Noted this got worse as bomb encounters became more frequent from the
target-ID fixes above -- a real clue, not a coincidence.

Found by reading `play_with_bombs.py`'s control flow (no new live capture
needed, the bug was visible in the code):

1. **Paratrooper engagements fully block bomb tracking.** Committing to a
   paratrooper shot does a single blind `time.sleep(sleep_s)` while
   rotating -- which can take over a second for a big sweep -- during
   which `find_bomb()` is never called at all. Since the merge-split fix
   makes paratrooper engagements far more frequent, a bomb appearing
   during that window goes completely unwatched until the rotation
   finishes, by which point it's either badly late or the tracking
   baseline is stale. This directly explains why bomb performance got
   *worse* as the other fixes made paratrooper engagement more frequent.
2. **Zero-tolerance tracking reset.** A single frame with no bomb-shaped
   blob detected (transient occlusion by another sprite, a one-frame
   detection blip) reset `bomb_prev` to `None` instantly, discarding
   tracking progress and forcing a full 2-point re-acquisition.

**Fixes**: paratroopers rotations now poll for a bomb every 20ms during
the wait and abandon the paratrooper shot (cheap -- just a missed shot,
no explosion) the instant one appears, so bomb handling is never blocked
behind a paratrooper engagement. Tracking loss on a bomb-not-detected
frame now only clears `bomb_prev` after `BOMB_MAX_TRACK_GAP_S` (0.4s) of
no detection, not instantly.

**Live-validated, 900s session**: 18 episodes, scores `[70, 44, 72, 57,
41, 42, 48, 73, 56, 56, 60, 44, 51, 60, 80, 60, 44, 42]` (mean 55.6,
median 56, max 80 -- consistent with prior sessions, no regression). 17
bomb shots, 6 hits = **35% bomb hit rate** -- up from the 22.8-29% blended
baseline before today's bomb fixes, though not dramatically higher, and
still clearly short of what "the gun should reliably shoot down bombs"
would look like.

**Important caveat, stated plainly**: the "abort paratrooper shot for
bomb" path logged **zero** triggers across the full 900s session -- the
blocking-engagement scenario, while a real and correctly-fixed bug, does
not appear to have been the dominant contributor in this particular run
(no bomb happened to arrive during an active paratrooper rotation this
time). The observed improvement is better attributed to the tracking
grace-period fix. This doesn't mean fix 1 was unnecessary -- it's still a
real structural risk worth having fixed -- but it means the "still doesn't
move/fire most of the time" complaint is not fully explained yet. Not yet
built: a funnel diagnostic counting, for every raw bomb-shaped detection,
how many survive each successive filter (shape match, gap timeout,
diagonal-speed check, angle check, altitude gate) to see exactly where
the remaining attrition happens -- needed before further bomb-detection
changes should be trusted.

#### The "bomb" is not a real, interceptable threat (2026-09-24): the whole bomb-shooting workstream was aimed at the wrong thing

User, watching live with the newly-calibrated acceleration model running:
"0/3 on bomb hits in exactly the same way -- the turret moves right
before impact and _maybe_ gets a single shot off." Then, decisively:
"you're shooting at a piece of the shrapnel of the exploding turret...
it's obvious they aren't hits because the game ends immediately after
each event."

Checked directly: pulled every "BOMB shot" and the next "GAME OVER" from
the live log. **All 14 bomb-shot attempts in that session -- hits and
misses alike -- were followed by GAME OVER 5.4-5.7 seconds later, with
no exception.** That gap is far too tight (a ~0.3s spread across 14
independent events) to be coincidence; it's a fixed-length sequence.

Built `diagnose_bomb_vs_doom.py` to test this directly: play normally,
defend paratroopers as usual, but **never fire at the bomb-shaped blob at
all** -- if the game still ends ~5.5s after every sighting regardless,
that proves the outcome doesn't depend on shooting it (ruling out "we're
just too slow/inaccurate to stop it in time"). Confirmed, twice: sighting
at 40.0s -> GAME OVER at 46.7s (6.0s later); sighting at 86.3s -> GAME
OVER at 93.0s (6.0s later). Not shooting it changes nothing.

Also checked the obvious candidate cause -- `paratrooper_env.py` already
documents a real doom mechanic (`DOOM_LANDED_THRESHOLD=4`: a side's 4th
landing freezes its walkers, then they "assemble and end the episode" a
few seconds later, matching this timing suspiciously well). But
`count_walkers_per_side` at the moment of every sighting read (0,1) and
(1,1) -- nowhere near 4 on either side. So it's very likely *not* that
specific mechanic, or at least not visible-walker-count-driven doom as
currently instrumented. **The true trigger is still unidentified** -- what's
confirmed is only that by the time this white 7x7 accelerating shape
appears, the episode's end is already locked in, unaffected by anything
we do afterward.

**This retroactively explains today's entire bomb investigation**: the
"deterministic trajectory" (exact same y=276 start, same acceleration,
same ~56-63deg angle progression across every instance) was never
surprising bomb physics -- it's a canned end-of-episode animation, which
is *supposed* to look identical every time. The "hits" (consistent +29/30
score jumps) are very likely an automatic end-of-episode score bonus that
lands at a fixed point in this sequence, unrelated to whether our shot
was well-aimed -- which is also why hit rate never climbed past ~35-50%
regardless of which targeting model was used: we were never the variable
that mattered.

**Consequence**: none of today's bomb-accuracy fixes (merge-split
targeting was real and validated; the bomb-specific ones -- density
filter, interrupt/tracking-tolerance fix, calibrated acceleration model
-- were built and tuned against something that was never a live,
stoppable threat) should be trusted as "improving survival." They may
still be harmless (a wasted shot during an already-lost sequence costs
nothing that matters), except for the interrupt logic, which by design
*abandons a real paratrooper shot* to chase this non-threat -- that part
is actively counterproductive and should be reverted or disabled.
**Recommended next step**: stop treating this shape as a target entirely;
investigate what actually triggers the sequence (score threshold? wave
count? something else in `paratrooper_env.py` not yet cross-checked?)
before deciding whether it's preventable at all, rather than continuing
to refine how we shoot at it.

**Not yet done**: the density/spray fix (bug 2) hasn't had its own
dedicated live validation run yet -- it's implemented and reasoned from
directly-measured pixel data, but per this session's standing rule,
that's not the same as confirming it live. Also unresolved: the rare
residual "ignored" cases from very dense multi-trooper piles that a
simple y-gap split can't fully separate.

**Not yet done**: no dedicated bomb accuracy calibration (this reused the
paratrooper `LATENCY_OFFSET_S`/`TICK_SAFETY_MARGIN_S` constants and the
same rotate/sleep/fire path uncritically); no cross-check of the
`solve_bomb_intercept` discrete-grid logic against ground truth the way
`solve_intercept` was; the two-observation velocity estimate is noisy
(only ~0.3-0.4s between samples) and could benefit from a third
confirmation point; false-positive filtering against helicopter fragments
is heuristic, not verified against a large clean sample the way the
paratrooper detector was.

#### Bomb-shot root-cause analysis and two fixes (2026-09-23)

Asked directly: was the ~23-29% blended bomb hit rate a hard geometry
problem (bombs dropped at inherently varied/difficult angles) or an
execution problem (can't fire accurately even at a known angle)? Neither,
exactly -- pulled all 161 logged bomb-shot lines from both prior sessions
and cross-referenced trajectory angle, altitude, rotation distance, and
hit/miss:

- **38% of all "bomb" shots (61/161) were a false positive**, not bombs at
  all: a fragment of the helicopter sprite near its known altitude bands
  (219.2/243.2) clears the shape+diagonal-motion filter. It has a
  distinctly shallow trajectory angle (max observed 36.8 deg from
  horizontal, tightly clustered at y=224-268) and hit only 1/61 (1.6%) --
  because it was never a real target. Real bombs were never observed
  shallower than 45 deg -- a clean, wide gap.
- Among the **real** (steep-trajectory) shots, hit rate climbed
  monotonically with how far the bomb had already fallen when fired at:
  0% at y~250, 14% at y~300-350, 62% at y~400, 100% at y~500 -- while
  showing almost no dependence on how much rotation was needed (27-40%
  across small/big holds). That rules out an execution/timing story (like
  the paratrooper gun's tick-quantization issue) -- it's lead-time error:
  the 2-frame velocity estimate's noise compounds over the predicted
  flight time, so a shot committed while the bomb is still high up (long
  `hold_t`) misses far more than one taken once it's already low (short
  `hold_t`).

**Two fixes applied to `play_with_bombs.py`**: `BOMB_MIN_TRAJ_ANGLE_DEG =
40` rejects the helicopter-fragment false positive outright (clean margin
between the observed 36.8 deg max false-positive and 45 deg min real
bomb); `BOMB_MIN_FIRE_Y = 380` withholds fire while a confirmed real bomb
is still high up, continuing to re-track it with fresher, shorter-baseline
velocity estimates each frame until it crosses that altitude (chosen at
the data's inflection point between the 14% and 62% bins).

**Live-validated, 900s session**: 12 bomb shots total (all real -- every
single one landed in the 56-60 deg range, zero false-positive shots fired,
confirming the angle filter works), **5 hits = 42%** (up from the 36%
real-bomb-only baseline, and far above the previous 22.8-29% blended
figures that were being dragged down by wasted false-positive shots).
18 completed episodes, scores `[68, 49, 65, 42, 35, 16, 29, 31, 6, 67, 17,
29, 45, 70, 44, 62, 62, 42]`, mean 43.3, median 43.0, max 70 -- lower than
the two prior sessions' means (62.8, 60.9), most likely ordinary
episode-to-episode variance (paratrooper performance dominates overall
score far more than the dozen bomb encounters per session do) rather than
a regression, but worth keeping an eye on across further runs rather than
assuming the bomb fix explains it.

#### Phase 2 solved from frame-by-frame recordings (2026-09-25): the bomb IS real, and it is fully deterministic

**Correction to the 2026-09-24 entry above.** The 8x8 white disc that
appears ~6s before GAME OVER *is* the real bomb -- it looked like a
"canned doom animation" because bomb physics really are identical every
time, and not shooting it "changed nothing" because it is lethal. The
old `hunt_phase2.py` never found phase 2 because its `heli_seen` counted
the 48px-wide plane sprite as a helicopter. Built `record_episode.py`
(background thread records every frame, palette-indexed, dumped on game
over) and read the phase-2 sequence directly off the frames.

**Game facts (all in game-area coords = Xvfb frame minus (192, 200)):**
- Everything moves on the ~18.2 Hz game tick with integer steps.
- Phase 1 (helicopters) always ends at ~38.0s after episode start; first
  plane enters at 38.2-38.3s (10/10 episodes). Helicopter = cyan skids
  32x4 at y0 17 or 41; plane = cyan body 48x20 at y0 1.
- Bomb = white disc, bbox exactly 8x8, 48 px. x: exactly 8 px/tick (the
  plane's speed). y: the fixed discrete-gravity sequence `BOMB_YS`
  (vy grows 2 px/tick every 4 ticks) -- ~45 deg mid-flight, ~65 deg at the
  turret, so "the few launch angles" are really one parabola translated
  in x. Release x varies with the plane (seen 24-80 from the left);
  every bomb lands on the turret ~33 ticks (1.8s) after release. Since y
  identifies the tick, ONE detection gives the whole future path.
- Bullets are 2x2 white dots, NOT hitscan: a fixed integer step of 16-21
  px/tick per barrel position (`calibrate_bullets.py`, 90 trials). The
  barrel only rests at 19 discrete positions and moves ~1 position per
  tick. Up stops it exactly where it is (`turret.py` calibration: lead 0
  exact 16/17, lead 1 always one short).
- A killed bomb breaks into small white dashes; +30 score.

**Solver (`bomb_model.py`, `bomb_defense.py`):** simulate every barrel
position's bullet tick by tick against the known bomb path, pick the
reachable position with the widest window of hitting spawn ticks, `goto`
it closed-loop, then fire exactly when the bomb's own tick counter says
the bullet will spawn in the window (+-1 tick margin). Bombs from the left
come nearly straight at the turret, so the best shot is head-on along
their line (usually 117-126 deg barrel) with a forgiving multi-tick window.
The shrapnel problem is solved by construction: only the exact 8x8/48px
disc on the known y-sequence with a plausible release x is ever a target.

**Results:** run 2 (with a since-fixed goto bug): 19/20 bombs killed,
new record score 348 (218s, two full plane waves). See run 3 notes below.

#### Follow-ups and helicopter shooting (2026-09-25/26)

- Later helicopter waves fly LOWER (skids y0 29/65 vs 17/41 in wave 1).
  Hardcoded lanes left the bot stuck in "phase 2" (turret idle) after the
  second plane wave, and low skids (32x4, dense) passed the paratrooper
  shape filter so the phase-1 solver chased helicopters. Fixed in
  `bomb_model.sky_sprites` (deep band) and `play_one_game.is_real_trooper_shape`
  (reject flat/wide blobs).
- Bomb bursts: cap 3 bullets in flight, stop once a bullet's meeting tick
  has passed and the bomb is gone (previously emptied whole windows, up
  to 21 shots, into the debris).
- Open-ended run before helicopters: 82 games, bombs 250/254, median 121,
  mean 130, best 394 (739 in an earlier run). Nearly every death is to
  paratroopers in later helicopter waves.
- **Helicopters** (`heli_model.py`): exactly 8 px/tick, upper lane flies
  left, lower lane right; sprite = 44x20 box at a fixed offset from the
  32x4 skids. Same lane simulation as bombs; engaged ONLY when there is
  no paratrooper target, rotation aborts if a trooper/plane/bomb appears,
  max 2 bullets each. First 4 games: mean 437, new record 811. Kills
  verified visually; kill metric counts helicopters that left the screen
  as "unresolved", not kills.
- Still open: paratrooper solver assumes hitscan bullets (aim_solver
  docstring) -- wrong, bullets are ~16-21 px/tick; target selection
  picks the highest trooper rather than the most urgent.

#### Paratrooper targeting rebuilt on the tick model (2026-09-26)

Measured from recordings (`trooper_model.py` docstring has the details):
body = cyan 8x12 + white 4x4 head; canopy = cyan 24x12 at exactly
(body.x-8, body.y-32); x never changes (identity key); 8 px/tick free fall,
4 px/tick under canopy, 8 px/tick again once the chute is shot (so speed
alone can't tell a chute-shot faller from a fresh jumper -- history can).
Canopies open anywhere from y 73 to 349 (the old solver assumed one fixed
altitude) after 100-204 px of free fall; 88/89 troopers open one. Landed
troopers = white heads at ground level; they stack (pyramid).

Policy: per-column tracker with states free/canopy/doomed; shoot the
CANOPY (3x the area, half the speed, and a chute kill over a landed trooper
takes both out); free-fallers only with short-horizon body shots, otherwise
pre-aim at the predicted canopy point. One bullet when the hit window is
wide, two when narrow, then no more shots at that trooper until its
meeting tick resolves hit/miss -- no spray, never shoots a doomed faller.

Bugs found on the way: helicopter direction must be measured from motion
(lane heights are reused with REVERSED directions in later waves -- the
earlier "kills" in those waves were false and cost points); goto must check
the barrel before doing per-frame bookkeeping (the extra processing made
stops land a position late and cost 2 bombs).

Result (8 fresh games, trooper5): scores 392-701, mean 579, median 592
(helicopter-only bot: mean ~330). Bombs 33/33. Trooper engagements: 59
kills (40 chute, 19 body) vs 8 misses (88%); 7 chute kills landed on a
waiting trooper. Helicopters: 95% of fired engagements confirmed down by an
independent frame check. The paratrooper solver's hitscan assumption is
retired (bomb_defense no longer uses aim_solver for troopers).

#### Planes, the hidden limit lanes, and pixel-exact hitboxes (2026-09-27)

Overnight continuous run (125 games, pre-planes): mean 447, median 433,
best 1103, bombs 413/425. Lost bombs: 6 goto-off-by-one (processing speed),
2 unreadable barrel, 3 model-precision (release x~48, only window meets
the bomb at the turret).

**Planes** (`heli_model.plane_box`): 48x20 hollow outline, always y 1..20,
8 px/tick. Engaged only with no bomb in flight, from positions within 5 of
the bomb park position, aborting on any bomb. One shot when the window
has >=3 ticks. Verified from frames: single-bullet kills.

**Evidence clips**: the player queues ~1.5-2.5s frame clips of notable
events (plane kill/miss, trooper miss, bomb landed) and writes them only
when idle -- writing inline blocked the loop ~0.3s and cost a bomb.

**Trooper misses were mostly one hidden bug, not the hitbox.** 10 of 11
missed-trooper clips were fired from the rotation-limit positions, and 7
of those bullets flew (+-20,-6)/tick instead of the table's (+-20,-4),
passing 10-16 px over the canopy. `calibrate_limit_lanes.py`: stopping the
moment the limit sprite appears -> (+-20,-6) in 20/21 shots; pushing into
the stop for 0.2s first -> (+-20,-4) in 23/23. So each limit position is
two lanes that look identical on screen. `bomb_model.lane()` picks the
right one from the turret state tracked by `turret.goto`/`turret.clamp`;
troopers may plan a clamp (flatter lane, reaches lower side targets,
~5 ticks slower). The remaining miss was a 1 px graze on the canopy's
empty top corner -> canopy/body now use pixel-exact masks.

Result (6 games, hitbox1): trooper hit rate 72% -> 100% (37/0), mean 614,
new record 1124.

#### Trooper hitbox, measured from a per-shot audit (2026-09-27)

`audit_trooper_shots.py` + `AUDIT_TROOPER_SHOTS=<n>` (a clip around every
planned trooper shot + shots_<t0>.json of every Up press and its target).
A bullet is never drawn overlapping what it hits -- it vanishes on the
tick it would enter it -- so the audit classifies the *next* position.

- **Two adjacent solid boxes**, not pixel masks: canopy box = dome AND the
  whole string area down to the head (x-8..x+15, y-32..y-5 in body
  coords); trooper box = head + body (x..x+7, y-4..y+11). 12 of 14 bullets
  that reached the black space between the strings killed the canopy.
- **Free-faller body shots are a gamble**: 3/15 hit; 5 misses were the
  canopy opening mid-flight (trooper halves speed, bullet passes under).
  Now point-blank only (<=4 ticks); otherwise pre-aim and wait.
- **Tracker bug**: a missed trooper whose canopy then opened stayed
  "pending" forever and was never targeted again (and never counted as a
  miss). Fixed: any surviving, non-doomed trooper resolves as a miss.
- The tracker's own miss count still undercounts (3 logged vs 7 audited).

Audit run 2 (6 games): 29 kills / 7 clear misses (81%, same rate as
before, different mix); remaining misses are timing (+-1 tick of fire-key
latency, 4 cases), long flights (meet at 36 ticks, 2 cases), and one
bullet inside the rectangle's lower corner beside the strings without a
kill (box may narrow at the bottom -- 1 case, unconfirmed).

### Current state (as of this plan)

| Constant | Value | Confidence |
|---|---|---|
| Turret rotation speed | ~124 deg/sec | Low -- only 3-4 trials, spread 121-129 deg/sec |
| Rotation limits | 25.71 deg (right) to 155.10 deg (left) | High -- directly measured, repeatable |
| Paratrooper fall speed | ~138 px/sec, zero horizontal drift | Medium-high -- one clean isolated trace, consistent throughout descent |
| Post-destruction free-fall speed | Assumed same as canopy-open (~138 px/s), per direct observation of the game, not yet independently measured | Unverified |
| Bullet travel time | Unknown -- treated as instantaneous (hitscan) | Unmeasured; grid search over a bundled `LEAD_TIME_S` fudge factor (0.0-0.1s) was inconclusive at current sample sizes |
| System latency (xdotool round-trip) | ~61ms, empirically fit and now compensated via `LATENCY_OFFSET_S` | Medium -- fit from noisy practice data, not a clean isolated measurement |
| Helicopter speed/altitude | Not yet measured | Unmeasured |
| Hitbox / hit tolerance (px or deg) | Not yet measured | Unmeasured |

**Current measured hit rate: ~28% (14/50) against isolated falling
paratroopers**, up from ~13% before the latency fix. Not close to the
target yet.

### Stage 1.1 -- Tighten the core physics constants

- Re-run rotation-speed calibration with many more trials (10+ full
  sweeps in each direction, not 3-4) to shrink the 121-129 deg/sec spread
  to a tight, trusted number. Check for non-uniformity (does it decelerate
  near the limits, or is it truly constant edge-to-edge?).
- Re-verify paratrooper fall speed across multiple independent drops (not
  just one), and check whether it changes across waves/difficulty levels
  as score increases (classic arcade games often speed up over time).
- Measure helicopter horizontal speed and altitude band the same way fall
  speed was measured (passive frame tracking, no action needed).
- **Validation gate**: constants stable to within ~1-2% across repeated
  independent trials before moving on.

### Stage 1.2 -- Isolate and fix latency at the source

- `actions.py`'s `do_action()` calls `xdotool key --delay 50 ...`, which
  blocks for the full 50ms before returning -- this is the dominant
  identified latency source. Test reducing that delay (e.g. to 10-15ms)
  directly at the source rather than only compensating for it after the
  fact via `LATENCY_OFFSET_S`; confirm the game still reliably registers
  the keypress at a shorter delay before relying on it.
- Re-measure `LATENCY_OFFSET_S` cleanly: a dedicated calibration script
  that repeatedly rotates for a fixed short duration and directly compares
  predicted vs. actual barrel angle, not inferred indirectly from
  hit/miss outcomes in practice runs (too noisy to isolate this term
  precisely).
- **Validation gate**: predicted vs. actual barrel angle at fire time
  agrees within ~1-2 degrees across repeated trials.

### Stage 1.3 -- Measure bullet travel time directly

- Fire from a fixed angle with no target in the way, and frame-diff
  consecutive captures to isolate and track the projectile itself (a
  moving pixel/line separate from any threat or the barrel) to get a
  real px/sec bullet speed, rather than inferring it indirectly through
  a bundled `LEAD_TIME_S` fudge factor. If bullets turn out fast enough
  to be reasonably approximated as instantaneous, document that and drop
  `LEAD_TIME_S` entirely; if not, replace the fudge factor with a real
  ballistic lead calculation (bullet-speed-aware intercept, not just a
  flat time shift).

### Stage 1.4 -- Determine the real hitbox

- Deliberately fire at small, controlled angular offsets from the
  computed target center and observe the largest offset that still
  registers a hit, to map out the effective hitbox width in degrees/px.
  This tells us how much residual error (from Stages 1.1-1.3) we can
  actually tolerate -- may reveal that some of the current miss rate is
  from a genuinely narrow hitbox rather than remaining calibration error.
- Repeat for helicopters, which likely have a different hitbox/aim-point
  given horizontal motion at a fixed altitude band.

### Stage 1.5 -- Rebuild and validate against the target

- With tightened constants, re-run `practice_deterministic_aim.py` (or
  its successor) at a much larger sample size (100+ shots) to get a
  statistically solid hit-rate read, iterating on whichever constant
  the miss pattern implicates.
- **Validation gate**: ≥80% one-shot hit rate against isolated falling
  paratroopers over 100+ attempts, and a comparable check against
  helicopters, before starting Phase 2 in earnest.
- Confirm the solver still performs acceptably when the target is
  embedded in a busier, multi-threat screen (not just the clean isolated
  scenarios used for calibration) -- this is where target *detection*
  (picking the right blob out of a crowd) interacts with aiming
  *precision*, and is worth a dedicated check before assuming Phase 1 is
  fully done.

---

## Phase 2: Learned Prioritization (risk-avoidance-driven scoring)

Once the gun reliably hits what it's told to shoot at, the remaining
problem is purely a decision problem: **which visible target, if any,
should be engaged right now** to maximize survival time (and therefore
score), given the actual game mechanic that a side reaching 4 cumulative
landings triggers an unavoidable, fixed-length end-of-episode countdown.

### The priority list (established from direct game-mechanics analysis)

1. **A paratrooper is on screen and reachable** -> shoot it. Among
   multiple, prioritize by a combination of (a) ease/probability of a
   hit and (b) how close that side already is to the 4-landing doom
   threshold -- a side at 3 landed outranks a side at 0, regardless of
   which is angularly closer.
   - **Special case**: if a falling paratrooper's trajectory is aligned
     to land on an existing walker's position, prioritize it specifically
     -- this is the only way to remove an already-landed paratrooper (a
     2-for-1 clear), and can pull a side back from the brink of doom.
2. **No paratroopers on screen** -> shoot helicopters. Fewer helicopters
   in the air means fewer future paratroopers.
3. **No paratroopers or helicopters on screen** -> hold a default
   position good for incoming bombs, which happens to also work well for
   the next helicopter's likely entry angle. Low priority in practice
   until survival time improves enough for bomber-plane waves to appear
   regularly (confirmed rare in current play).
4. **A side has already reached the doom threshold** -> abandon
   prevention entirely for the remainder of that episode (nothing further
   changes the outcome) and switch to pure opportunistic scoring with
   whatever time is left -- matches the `_doomed` flag already implemented
   in `paratrooper_env.py`'s v3 reward.

### Implementation approach

- **Start rule-based, not learned.** The priority list above is already
  fully specified and deterministic enough to implement directly as a
  target-selection function on top of the Phase 1 solver -- no trial and
  error needed for the parts we already understand precisely (doom
  threshold, alignment-clear detection, doomed-side shutoff).
- **Reserve learning/tuning for what's genuinely underspecified**: the
  exact tradeoff weight between "closer to doom" vs. "easier to hit" when
  both differ across two live threats, timing nuances around the
  alignment-clear opportunity window, and any interaction effects that
  only show up under real, busy multi-threat play. This is a much smaller
  and more tractable learning problem than raw pixel-to-action control,
  since Phase 1 has already solved execution -- it only has to learn how
  to weigh a handful of already-identified factors, not how to aim at all.
- Concretely: implement the rule-based version first, measure it against
  real play (score, survival time, frequency of reaching doomed state,
  frequency of successfully exploiting the alignment-clear), then decide
  whether a lightweight learned layer on top is actually needed or
  whether the hand-specified rules already capture the strategy well
  enough.

### Validation gates

- Rule-based target selector implemented and running against live play
  before any tuning/learning layer is added.
- Compare against the current RL baseline on: average score per episode,
  average survival time (steps or seconds to doomed/game-over), frequency
  of ever reaching a doomed side, and (if measurable) frequency of
  successful alignment-clears.
- Only add a learned tuning layer if the rule-based version's failure
  modes point at a genuinely underspecified tradeoff rather than an
  execution (Phase 1) problem -- re-check Phase 1 accuracy first if
  Phase 2 isn't performing as expected, since a prioritization bug and an
  aiming-precision bug can look identical from the outside (both show up
  as "missed a paratrooper it should have hit").
