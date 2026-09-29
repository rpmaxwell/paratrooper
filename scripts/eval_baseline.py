#!/usr/bin/env python3
"""Stage 6 definition-of-done check: run a pure random-action agent and
the trained PPO agent for the same number of episodes each, and confirm
the trained agent's average episode reward is measurably higher.
"""
import numpy as np
from stable_baselines3 import PPO

from paratrooper_env import ParatrooperEnv

MODEL_PATH = "/captures/ppo_paratrooper"
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

    print(f"\nloading trained model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH, device="cpu")

    # Report both: deterministic (argmax) is the usual "polished" eval
    # convention, but a short/undertrained PPO policy can still assign
    # real probability mass to the useful action without it being the
    # single top pick everywhere -- stochastic sampling matches how the
    # policy actually generated its (positive) training-time rewards.
    def trained_policy_stochastic(obs):
        action, _ = model.predict(obs, deterministic=False)
        return int(action)

    def trained_policy_deterministic(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    print(f"running {N_EPISODES} trained-agent episodes (stochastic)...")
    trained_stoch_rewards, stoch_fire = run_episodes(env, trained_policy_stochastic, N_EPISODES, "trained-stochastic")

    print(f"running {N_EPISODES} trained-agent episodes (deterministic)...")
    trained_det_rewards, det_fire = run_episodes(env, trained_policy_deterministic, N_EPISODES, "trained-deterministic")

    env.close()

    def summarize(rewards, fire_stats):
        mean, std = float(np.mean(rewards)), float(np.std(rewards))
        shots = float(np.mean([s for s, h, hr in fire_stats]))
        hit_rate = float(np.mean([hr for s, h, hr in fire_stats]))
        return mean, std, shots, hit_rate

    r_mean, r_std, r_shots, r_hitrate = summarize(random_rewards, random_fire)
    ts_mean, ts_std, ts_shots, ts_hitrate = summarize(trained_stoch_rewards, stoch_fire)
    td_mean, td_std, td_shots, td_hitrate = summarize(trained_det_rewards, det_fire)

    print(f"\nrandom               episode rewards: {random_rewards}  mean={r_mean:.2f} std={r_std:.2f} "
          f"mean_shots={r_shots:.1f} mean_hit_rate={r_hitrate:.2f}")
    print(f"trained (stochastic) episode rewards: {trained_stoch_rewards}  mean={ts_mean:.2f} std={ts_std:.2f} "
          f"mean_shots={ts_shots:.1f} mean_hit_rate={ts_hitrate:.2f}")
    print(f"trained (deterministic) episode rewards: {trained_det_rewards}  mean={td_mean:.2f} std={td_std:.2f} "
          f"mean_shots={td_shots:.1f} mean_hit_rate={td_hitrate:.2f}")

    for label, mean in [("stochastic", ts_mean), ("deterministic", td_mean)]:
        if mean > r_mean:
            print(f"\nPASS ({label}): trained ({mean:.2f}) > random ({r_mean:.2f})")
        else:
            print(f"\nFAIL ({label}): trained ({mean:.2f}) did not beat random ({r_mean:.2f})")


if __name__ == "__main__":
    main()
