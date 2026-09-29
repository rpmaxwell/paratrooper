#!/usr/bin/env python3
"""Continuously play with a random-action agent, episode after episode,
so the game can be watched live over VNC. Not a trained agent (that's
Stage 6) -- just Stage 4's env driven forever instead of for a fixed
step count, for demo/observation purposes.
"""
import time

from paratrooper_env import ParatrooperEnv

env = ParatrooperEnv()
obs, info = env.reset()
episode = 1
ep_steps = 0
ep_reward = 0.0

print(f"episode {episode} start")
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    ep_steps += 1
    ep_reward += reward

    if terminated or truncated:
        print(f"episode {episode} end: steps={ep_steps} total_reward={ep_reward:.0f} final_score={info['score']}")
        episode += 1
        ep_steps = 0
        ep_reward = 0.0
        obs, info = env.reset()
        print(f"episode {episode} start")
        time.sleep(0.5)
