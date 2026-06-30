"""Headless evaluation of the trained conv DQN (no render window)."""
import argparse, numpy as np, torch
from Agent_Implementation import AdvancedDQNAgent
from Atari_Environment import AtariWrapper

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=20)
p.add_argument("--weights", default="model_weights.pth")
args = p.parse_args()

env = AtariWrapper(env_name="BreakoutNoFrameskip-v4", frame_skip=4, rendering_mode="rgb_array")
state = env.reset()
agent = AdvancedDQNAgent(state_shape=state.shape, action_size=env.action_space.n)
agent.load_model(filepath=args.weights)
agent.epsilon = 0.0
agent.epsilon_min = 0.0

rewards = []
for ep in range(args.episodes):
    state = env.reset()
    done = False
    ep_reward = 0.0
    while not done:
        action = agent.select_action(state)
        state, reward, done, truncated, _ = env.step(action)
        ep_reward += reward
        if truncated:
            break
    rewards.append(ep_reward)
    print(f"  ep {ep+1:>3}/{args.episodes}: reward={ep_reward:.0f}", flush=True)
env.close()

r = np.array(rewards)
print("-" * 44, flush=True)
print(f"mean reward : {r.mean():.2f}", flush=True)
print(f"max reward  : {r.max():.0f}", flush=True)
print(f"min reward  : {r.min():.0f}", flush=True)
print(f"std reward  : {r.std():.2f}", flush=True)
