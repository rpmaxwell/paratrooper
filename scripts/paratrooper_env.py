#!/usr/bin/env python3
"""Stage 4: gymnasium.Env wrapper for Paratrooper.

Screen-only: observations come from mss screen capture, actions go out
via xdotool (actions.py), reward is the raw on-screen score delta
(read_score.py), done comes from the title-screen template check
(read_score.is_done). No emulator memory is read anywhere.
"""
import subprocess
import time
from collections import deque

import gymnasium as gym
import mss
import numpy as np
from gymnasium import spaces
from PIL import Image

from actions import do_action
from read_score import is_done, read_score
from threats import count_walkers_per_side, detect_barrel_angle, nearest_threat_angle_gap_by_side

OBS_SIZE = 84
FRAME_STACK = 4
STEP_SETTLE_S = 0.1  # let the game render the effect of an action before capturing
RESET_SETTLE_S = 0.3
MAX_EPISODE_STEPS = 10_000  # safety valve only, not a designed episode limit

# Bounding box of the DOSBox-X window within the 1024x768 Xvfb frame,
# excluding its own menu-bar strip -- measured empirically from
# captures/*.png (nonblack bbox x:[192,831] y:[184,598]; menu bar occupies
# y:184-200). The old crop-free obs resized the *entire* 1024x768 monitor
# grab, including the vast majority of that frame that is unused black
# desktop -- most of the CNN's 84x84 input was wasted on that.
GAME_RECT = (192, 201, 832, 599)  # (x0, y0, x1, y1)

FIRE_ACTION = 3

# v3 reward redesign, driven by a key mechanical fact about the actual game
# (confirmed by direct observation, not inferred from pixels): a side isn't
# gradually more "at risk" as paratroopers land on it -- once the 4th one
# lands on a side, that side is *doomed*: the walkers there freeze for a few
# seconds (occasionally one more drops during this window), then assemble
# and end the episode. It's a hard threshold, not a smooth continuum, and
# after it's crossed nothing done on either side can prevent the episode
# from ending soon. So the actual objective is closer to survival-time (=
# score accumulation window) than raw score: avoid the 4th landing on
# either side for as long as possible, and while that's still avoidable,
# prioritize whichever side is closer to it over any other target,
# including a helicopter that's easier to hit.
DOOM_LANDED_THRESHOLD = 4

# Escalating cost for landings 1-3 (still recoverable -- discourage but
# don't panic), then a large, mostly-fixed penalty exactly at the 4th
# landing, representing "this side just became unrecoverable." Landings
# beyond the 4th (the occasional extra drop during the freeze) cost nothing
# further -- by then nothing done on this side matters anymore (see
# `_doomed` below).
LANDING_PENALTIES = {1: -2.0, 2: -4.0, 3: -8.0, DOOM_LANDED_THRESHOLD: -100.0}

# Dense per-step shaping toward the *priority* airborne threat, on top of
# the sparse hit_reward: -AIM_SHAPING_COEF * (angle_gap_deg / 180). Target
# selection is urgency-weighted (see step()), not just nearest-by-angle --
# a side with more cumulative landings is prioritized over a side that's
# merely closer in angle, since it's more urgent per the doom logic above.
# The coefficient is a *starting point*, not tuned -- it accrues every
# single step (unlike a hit, which is rare), so if it's too large it can
# dominate the objective and the agent ends up just chasing the shaping
# term instead of firing. Watch "aim_shaping" vs "hit_reward" magnitudes in
# info/Monitor logs and lower this if per-episode shaping totals dwarf
# per-episode hit totals.
AIM_SHAPING_COEF = 0.05

# Degrees of "priority" credited per existing landing on a side when
# choosing which side's nearest threat to shape aim toward. Deliberately
# larger than the max possible real angle gap (180deg) so a side with more
# cumulative landings *always* outranks a side that's merely closer in
# angle -- angle only breaks ties between sides with equal landing counts.
URGENCY_DEG_PER_LANDING = 200.0

# Direct bonus for a hit that happens while the barrel was already aimed
# close to a paratrooper (helicopters aren't cyan and never appear in the
# threat mask at all, so a close angle gap at the moment of a hit is a
# reasonable proxy for "that was a paratrooper, not a helicopter"). Without
# this, a helicopter hit and a paratrooper hit pay the same, so raw
# hit_reward alone gives no reason to prefer the harder, smaller,
# faster-moving target -- observed in practice as the agent becoming
# proficient at helicopters while ignoring paratroopers entirely. This is a
# starting point, not tuned; watch "paratrooper_bonus" vs "hit_reward" in
# logs.
PARATROOPER_HIT_BONUS = 3.0
PARATROOPER_HIT_ANGLE_THRESHOLD = 10.0  # degrees

# Confirmed empirically (see plan notes): every fire keypress costs exactly
# -1 off the displayed SCORE, floored at 0 -- independent of whether it hits
# anything. A hit adds a separate, larger bonus on top of that -1. Raw score
# delta therefore conflates a certain, immediate -1 with a rare, delayed
# positive -- which is what previously drove both PPO and DQN to collapse
# onto a "never fire" policy (guaranteed 0 beats an action whose naive
# expected value looks negative). `reward_mode="hit"` strips the -1 back out
# so the RL objective is just the hit bonus; raw score/cost are still
# exposed via `info` for real evaluation and logging.
FIRE_COST = -1.0


def _to_gray84(frame_rgb: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = GAME_RECT
    cropped = frame_rgb[y0:y1, x0:x1]
    img = Image.fromarray(cropped).convert("L").resize((OBS_SIZE, OBS_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


class ParatrooperEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reward_mode: str = "hit"):
        super().__init__()
        assert reward_mode in ("hit", "raw")
        self.reward_mode = reward_mode
        self.action_space = spaces.Discrete(4)  # 0=noop 1=rotate_left 2=rotate_right 3=fire
        # Channel-last (H,W,C): matches gym/SB3 image-observation convention
        # so stable-baselines3's CnnPolicy autodetects and transposes it
        # (via VecTransposeImage) without extra plumbing.
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(OBS_SIZE, OBS_SIZE, FRAME_STACK), dtype=np.uint8
        )
        self._sct = mss.mss()
        self._monitor = self._sct.monitors[1] if len(self._sct.monitors) > 1 else self._sct.monitors[0]
        self._frames = deque(maxlen=FRAME_STACK)
        self._last_score = 0
        self._steps_this_episode = 0
        self._moves = 0
        self._shots_fired = 0
        self._hits = 0
        self._misses = 0
        self._landed_left = 0
        self._landed_right = 0
        self._doomed = False

    def _grab(self) -> np.ndarray:
        img = np.array(self._sct.grab(self._monitor))
        return img[:, :, :3][:, :, ::-1]  # BGRA -> RGB

    def _stacked_obs(self) -> np.ndarray:
        return np.stack(self._frames, axis=-1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        frame = self._grab()
        if is_done(frame):
            # Only the title/game-over screen responds to this; pressing it
            # mid-episode is untested and intentionally avoided -- reset()
            # is expected to be called after a done=True step, per the
            # standard gym usage pattern.
            subprocess.run(["xdotool", "key", "space"], check=True)
            time.sleep(RESET_SETTLE_S)
            frame = self._grab()

        gray = _to_gray84(frame)
        self._frames.clear()
        for _ in range(FRAME_STACK):
            self._frames.append(gray)
        self._last_score = read_score(frame)
        self._steps_this_episode = 0
        self._moves = 0
        self._shots_fired = 0
        self._hits = 0
        self._misses = 0
        # Seed from the actual on-screen count rather than assuming 0 -- a
        # fresh game starts empty in practice, but this avoids a spurious
        # landing-penalty burst on step 1 if that ever isn't true.
        self._landed_left, self._landed_right = count_walkers_per_side(frame)
        self._doomed = (
            self._landed_left >= DOOM_LANDED_THRESHOLD or self._landed_right >= DOOM_LANDED_THRESHOLD
        )
        return self._stacked_obs(), {"score": self._last_score}

    def step(self, action: int):
        action = int(action)
        do_action(action)
        time.sleep(STEP_SETTLE_S)
        frame = self._grab()

        prev_score = self._last_score
        score = read_score(frame)
        raw_delta = float(score - prev_score)
        self._last_score = score

        is_fire = action == FIRE_ACTION
        # The -1 fire cost is floored at 0 by the game's own display (score
        # can't render negative), so a miss at prev_score==0 shows up as
        # raw_delta==0 -- identical to "no shot fired". Only assume the cost
        # actually applied if there was score for it to come out of;
        # otherwise assuming a -1 that wasn't visibly there turns every
        # floor-clipped miss into a phantom +1 "hit".
        fire_cost = FIRE_COST if (is_fire and prev_score > 0) else 0.0
        hit_reward = raw_delta - fire_cost
        if is_fire:
            self._shots_fired += 1
            if hit_reward > 0:
                self._hits += 1
            else:
                self._misses += 1
        else:
            self._moves += 1

        barrel_angle = detect_barrel_angle(frame)
        left_gap, right_gap = nearest_threat_angle_gap_by_side(frame, barrel_angle)

        # Paratrooper-hit bonus: helicopters aren't cyan and never appear in
        # the threat mask, so a small gap at the moment of a hit is a good
        # proxy for "that hit was a paratrooper." Computed from this step's
        # gaps (post-action, same frame the hit itself is read from).
        paratrooper_bonus = 0.0
        if is_fire and hit_reward > 0:
            candidate_gaps = [g for g in (left_gap, right_gap) if g is not None]
            if candidate_gaps and min(candidate_gaps) <= PARATROOPER_HIT_ANGLE_THRESHOLD:
                paratrooper_bonus = PARATROOPER_HIT_BONUS

        base_reward = (hit_reward if self.reward_mode == "hit" else raw_delta) + paratrooper_bonus

        left_n, right_n = count_walkers_per_side(frame)
        landing_penalty = 0.0
        for _ in range(max(0, left_n - self._landed_left)):
            self._landed_left += 1
            if not self._doomed:
                landing_penalty += LANDING_PENALTIES.get(self._landed_left, 0.0)
        for _ in range(max(0, right_n - self._landed_right)):
            self._landed_right += 1
            if not self._doomed:
                landing_penalty += LANDING_PENALTIES.get(self._landed_right, 0.0)
        if self._landed_left >= DOOM_LANDED_THRESHOLD or self._landed_right >= DOOM_LANDED_THRESHOLD:
            self._doomed = True

        # Once doomed, the episode is ending soon regardless of what happens
        # on either side -- no more aim-shaping toward "preventing" anything.
        if self._doomed:
            aim_shaping = 0.0
        else:
            left_score = (
                self._landed_left * URGENCY_DEG_PER_LANDING - left_gap if left_gap is not None else None
            )
            right_score = (
                self._landed_right * URGENCY_DEG_PER_LANDING - right_gap if right_gap is not None else None
            )
            if left_score is None and right_score is None:
                chosen_gap = None
            elif right_score is None or (left_score is not None and left_score >= right_score):
                chosen_gap = left_gap
            else:
                chosen_gap = right_gap
            aim_shaping = -AIM_SHAPING_COEF * (chosen_gap / 180.0) if chosen_gap is not None else 0.0

        reward = base_reward + aim_shaping + landing_penalty

        terminated = is_done(frame)
        self._steps_this_episode += 1
        truncated = self._steps_this_episode >= MAX_EPISODE_STEPS

        self._frames.append(_to_gray84(frame))
        obs = self._stacked_obs()
        info = {
            "score": score,
            "raw_score_delta": raw_delta,
            "fire_cost": fire_cost,
            "hit_reward": hit_reward,
            "paratrooper_bonus": paratrooper_bonus,
            "aim_shaping": aim_shaping,
            "landing_penalty": landing_penalty,
            "landed_left": self._landed_left,
            "landed_right": self._landed_right,
            "doomed": self._doomed,
            "moves": self._moves,
            "shots_fired": self._shots_fired,
            "hits": self._hits,
            "misses": self._misses,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        self._sct.close()
