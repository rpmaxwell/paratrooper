#!/usr/bin/env python3
"""Stage 6 (attempt 3): PPO training run, post-mortem on why attempt 1
(3072 steps) and attempt 2 (10240 steps) both converged to a "never
fire" policy that scored worse than random.

Root cause (confirmed empirically against the live game, not just
inferred from behavior): every fire keypress costs exactly -1 off the
displayed SCORE, deterministically and immediately, floored at 0; a
hit adds a separate, larger, delayed bonus on top of that -1. Training
against raw score delta means the agent's dominant, easiest-to-learn
signal is "firing loses points" -- it correctly learns to stop firing.
ParatrooperEnv now defaults to reward_mode="hit", which strips the -1
back out of the training signal (raw score is still tracked in `info`
for real evaluation). ent_coef is raised so the policy doesn't
re-collapse into a low-entropy no-fire habit before it has enough
signal to learn aim.

Checkpointed and resumable: attempt 3's first run (100k steps) was left
running detached via a backgrounded `docker exec` and was lost entirely
-- both training containers exited a few minutes short of completion
with no model ever written, and since nothing was checkpointed there
was nothing to resume from. This version saves a checkpoint every
CHECKPOINT_FREQ steps and resumes from the latest one on startup, so an
unattended multi-day run only loses a few minutes of progress if
interrupted, not the whole run.

Post-mortem on attempt 3's completed 850k-step run: ent_coef=0.02 wasn't
enough entropy to stop the policy from settling into a *different* local
optimum than "never fire" -- it converged to "hold a fixed ~45deg angle
near whichever side already has a cluster of targets and fire there,"
ignoring the other side entirely. That's a cheap, low-variance policy to
converge to and 0.02 of entropy bonus decayed away before the policy had
sampled the (expensive, multi-step, blind) "sweep across to the other
side" behavior enough times to learn it pays off. ent_coef is raised
again here, and paratrooper_env.py now adds reward shaping (aim-tracking
distance to the nearest airborne threat, and an escalating per-side
penalty for letting paratroopers land) so there's a training signal that
disfavors camping one side even before a full sweep is executed. IMPORTANT:
because the reward function changed, resuming from an *old* checkpoint
(trained under the previous reward) would carry over a value function
calibrated to a different reward scale -- move or clear
CHECKPOINT_DIR/MODEL_PATH before starting this run, don't resume in place.
"""
import glob
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from paratrooper_env import ParatrooperEnv

TOTAL_TIMESTEPS = 850_000  # ~48h at the observed ~5 steps/sec, with headroom
N_STEPS = 1024
ENT_COEF = 0.05  # up from attempt 3's 0.02, which wasn't enough to stop the
                 # policy from collapsing onto a fixed-angle single-side
                 # camping strategy -- see the docstring above.
MONITOR_PATH = "/captures/train_monitor.csv"
MODEL_PATH = "/captures/ppo_paratrooper"
CHECKPOINT_DIR = "/captures/checkpoints_ppo"
CHECKPOINT_FREQ = 10_000
KEEP_LAST_N_CHECKPOINTS = 3  # bound disk use -- host disk is very tight (~3GB free)
COMPLETE_SENTINEL = "/captures/ppo_training_complete"
INFO_KEYWORDS = ("score", "shots_fired", "hits", "misses", "moves", "landed_left", "landed_right", "doomed")


class PruningCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that deletes all but the N most recent checkpoints
    after each save, so a multi-day run can't fill up a nearly-full disk."""

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            files = sorted(
                glob.glob(os.path.join(self.save_path, "*_steps.zip")),
                key=os.path.getmtime,
            )
            for stale in files[:-KEEP_LAST_N_CHECKPOINTS]:
                os.remove(stale)
        return result


os.makedirs(CHECKPOINT_DIR, exist_ok=True)
if os.path.exists(COMPLETE_SENTINEL):
    os.remove(COMPLETE_SENTINEL)

env = Monitor(ParatrooperEnv(), filename=MONITOR_PATH, info_keywords=INFO_KEYWORDS, override_existing=False)

checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*_steps.zip")), key=os.path.getmtime)
if checkpoints:
    latest = checkpoints[-1]
    print(f"resuming from checkpoint {latest}")
    model = PPO.load(latest, env=env, device="cpu")
else:
    print("no checkpoint found, starting fresh")
    model = PPO("CnnPolicy", env, n_steps=N_STEPS, ent_coef=ENT_COEF, verbose=1, device="cpu")

remaining = max(TOTAL_TIMESTEPS - model.num_timesteps, 0)
print(f"{model.num_timesteps} steps already done, {remaining} remaining toward {TOTAL_TIMESTEPS}")

if remaining > 0:
    callback = PruningCheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=CHECKPOINT_DIR, name_prefix="ppo_ckpt")
    model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=False)

model.save(MODEL_PATH)
env.close()
with open(COMPLETE_SENTINEL, "w") as f:
    f.write("done\n")
print(f"\nsaved model to {MODEL_PATH}.zip and monitor log to {MONITOR_PATH}")
