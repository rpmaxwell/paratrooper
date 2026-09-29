#!/usr/bin/env python3
"""Stage 6 definition-of-done check for the DQN attempt: random-action
agent vs trained DQN agent, same number of episodes each."""
import numpy as np
from stable_baselines3 import DQN

from paratrooper_env import ParatrooperEnv

MODEL_PATH = "/captures/dqn_paratrooper"
N_EPISODES = 20


def run_episodes(env, policy_fn, n, label):
    rewards = []
    fire_stats = []
    for i in range(n):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            done = terminated or truncated
        shots, hits = info["shots_fired"], info["hits"]
        hit_rate = hits / shots if shots else 0.0
        fire_stats.append((shots, hits, hit_rate))
        print(
            f"  [{label}] episode {i+1}/{n}: reward={ep_reward:.0f} steps={ep_steps} "
            f"final_score={info['score']} shots={shots} hits={hits} hit_rate={hit_rate:.2f}"
        )
        rewards.append(ep_reward)
    return rewards, fire_stats


def main():
    env = ParatrooperEnv()

    print(f"running {N_EPISODES} random-action episodes...")
    random_rewards, random_fire = run_episodes(env, lambda obs: env.action_space.sample(), N_EPISODES, "random")

    print(f"\nloading trained DQN model from {MODEL_PATH}...")
    model = DQN.load(MODEL_PATH, device="cpu")

    def trained_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    print(f"running {N_EPISODES} trained-agent episodes...")
    trained_rewards, trained_fire = run_episodes(env, trained_policy, N_EPISODES, "trained")

    env.close()

    r_mean, r_std = float(np.mean(random_rewards)), float(np.std(random_rewards))
    t_mean, t_std = float(np.mean(trained_rewards)), float(np.std(trained_rewards))
    r_shots = np.mean([s for s, h, hr in random_fire])
    r_hitrate = np.mean([hr for s, h, hr in random_fire])
    t_shots = np.mean([s for s, h, hr in trained_fire])
    t_hitrate = np.mean([hr for s, h, hr in trained_fire])

    print(f"\nrandom  episode rewards: {random_rewards}  mean={r_mean:.2f} std={r_std:.2f} "
          f"mean_shots={r_shots:.1f} mean_hit_rate={r_hitrate:.2f}")
    print(f"trained episode rewards: {trained_rewards}  mean={t_mean:.2f} std={t_std:.2f} "
          f"mean_shots={t_shots:.1f} mean_hit_rate={t_hitrate:.2f}")

    if t_mean > r_mean:
        print(f"\nPASS: trained ({t_mean:.2f}) > random ({r_mean:.2f})")
    else:
        print(f"\nFAIL: trained ({t_mean:.2f}) did not beat random ({r_mean:.2f})")


if __name__ == "__main__":
    main()
