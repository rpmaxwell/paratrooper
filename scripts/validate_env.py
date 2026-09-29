#!/usr/bin/env python3
"""Stage 4 validation gate: run a random-action agent for several
hundred steps and check the environment is actually sound before any
RL algorithm touches it:
  - reward is nonzero at expected moments (kills/misses)
  - done fires correctly on death
  - reset() produces a clean state with no leakage from the previous episode
"""
from paratrooper_env import ParatrooperEnv

N_STEPS = 500


def main():
    env = ParatrooperEnv()
    obs, info = env.reset()

    assert obs.shape == env.observation_space.shape, obs.shape
    assert obs.dtype == env.observation_space.dtype, obs.dtype
    print(f"initial obs shape={obs.shape} dtype={obs.dtype} score={info['score']}")
    assert info["score"] == 0, f"expected clean reset score=0, got {info['score']}"

    nonzero_rewards = 0
    positive_rewards = 0
    negative_rewards = 0
    episode_count = 0
    episode_lengths = []
    steps_since_reset = 0
    done_events = []
    reset_scores = []

    for step in range(N_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps_since_reset += 1

        assert obs.shape == env.observation_space.shape, (step, obs.shape)
        assert obs.dtype == env.observation_space.dtype, (step, obs.dtype)

        if reward != 0:
            nonzero_rewards += 1
            if reward > 0:
                positive_rewards += 1
            else:
                negative_rewards += 1

        if terminated or truncated:
            done_events.append((step, info["score"], terminated, truncated))
            episode_count += 1
            episode_lengths.append(steps_since_reset)
            steps_since_reset = 0

            obs, info = env.reset()
            reset_scores.append(info["score"])
            assert info["score"] == 0, f"leakage: reset score={info['score']} after episode end"

    env.close()

    print(f"\nran {N_STEPS} steps, {episode_count} episodes completed")
    print(f"episode lengths: {episode_lengths}")
    print(f"nonzero rewards: {nonzero_rewards} (positive={positive_rewards}, negative={negative_rewards})")
    print(f"done events (step, score_at_done, terminated, truncated): {done_events}")
    print(f"reset scores after each episode end (should all be 0): {reset_scores}")

    assert nonzero_rewards > 0, "reward was never nonzero -- environment is likely broken"
    assert episode_count > 0, "no episode completed in this many steps -- done signal may be broken"
    assert all(s == 0 for s in reset_scores), "reset() leaked score from previous episode"
    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()
