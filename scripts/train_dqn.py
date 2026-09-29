#!/usr/bin/env python3
"""Stage 6 (attempt 3): DQN training run, post-mortem on why attempt 2
(10240 steps) converged to a "never fire" policy that scored worse
than random -- same root cause as the PPO side (see train_baseline.py):
raw score delta conflates a deterministic -1 fire cost with a rare,
delayed hit bonus, so optimizing it directly teaches "don't fire".
ParatrooperEnv now defaults to reward_mode="hit" to fix that. Exploration
is widened (exploration_fraction/exploration_final_eps) and buffer/
learning_starts/target_update sized up now that the budget is ~10x
larger, so epsilon doesn't collapse to its floor a few episodes in.

Checkpointed and resumable: see train_baseline.py's docstring -- attempt
3's first 100k-step run was lost entirely when the detached `docker
exec` session died a few minutes short of completion with nothing ever
saved. Checkpoints every CHECKPOINT_FREQ steps and auto-resume from the
latest one on startup bound the damage from a repeat to a few minutes.
Note the replay buffer isn't checkpointed (it would dwarf the ~3GB free
disk) -- a resume restarts with an empty buffer, refilling over the
next `learning_starts` steps, and the exploration schedule restarts
over `remaining` steps too. Both are minor, acceptable costs of a
resume compared to losing the learned Q-network entirely.
"""
import glob
import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from paratrooper_env import ParatrooperEnv

TOTAL_TIMESTEPS = 850_000  # ~48h at the observed ~5 steps/sec, with headroom
MONITOR_PATH = "/captures/train_monitor_dqn.csv"
MODEL_PATH = "/captures/dqn_paratrooper"
CHECKPOINT_DIR = "/captures/checkpoints_dqn"
CHECKPOINT_FREQ = 10_000
KEEP_LAST_N_CHECKPOINTS = 3  # bound disk use -- host disk is very tight (~3GB free)
COMPLETE_SENTINEL = "/captures/dqn_training_complete"
INFO_KEYWORDS = ("score", "shots_fired", "hits", "misses", "moves")


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
    model = DQN.load(latest, env=env, device="cpu")
else:
    print("no checkpoint found, starting fresh")
    model = DQN(
        "CnnPolicy",
        env,
        learning_starts=2_000,
        buffer_size=50_000,
        target_update_interval=2_000,
        exploration_fraction=0.3,
        exploration_final_eps=0.07,
        verbose=1,
        device="cpu",
    )

remaining = max(TOTAL_TIMESTEPS - model.num_timesteps, 0)
print(f"{model.num_timesteps} steps already done, {remaining} remaining toward {TOTAL_TIMESTEPS}")

if remaining > 0:
    callback = PruningCheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=CHECKPOINT_DIR, name_prefix="dqn_ckpt")
    model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=False)

model.save(MODEL_PATH)
env.close()
with open(COMPLETE_SENTINEL, "w") as f:
    f.write("done\n")
print(f"\nsaved model to {MODEL_PATH}.zip and monitor log to {MONITOR_PATH}")
