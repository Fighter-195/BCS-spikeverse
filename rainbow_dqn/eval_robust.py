"""Robustness eval: random no-op starts + small epsilon (standard Atari protocol).
Reveals whether the policy generalizes or just replays one deterministic trajectory."""
import argparse, numpy as np, torch
from Agent_Implementation import AdvancedDQNAgent
from Atari_Environment import AtariWrapper

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=15)
p.add_argument("--epsilon", type=float, default=0.05)
p.add_argument("--max-noop", type=int, default=30)
args = p.parse_args()

env = AtariWrapper(env_name="BreakoutNoFrameskip-v4", frame_skip=4, rendering_mode="rgb_array")
state = env.reset()
agent = AdvancedDQNAgent(state_shape=state.shape, action_size=env.action_space.n)
agent.load_model("model_weights.pth")
agent.epsilon = args.epsilon
agent.epsilon_min = args.epsilon

rewards = []
for ep in range(args.episodes):
    state = env.reset()
    # random no-op start to break determinism
    for _ in range(np.random.randint(1, args.max_noop + 1)):
        state, _, done, trunc, _ = env.step(0)
        if done or trunc:
            state = env.reset(); break
    done = False; ep_r = 0.0
    while not done:
        a = agent.select_action(state)
        state, r, done, trunc, _ = env.step(a)
        ep_r += r
        if trunc: break
    rewards.append(ep_r)
    print(f"  ep {ep+1:>3}: reward={ep_r:.0f}", flush=True)
env.close()
r = np.array(rewards)
print("-"*40, flush=True)
print(f"eps={args.epsilon}, no-op starts 1..{args.max_noop}", flush=True)
print(f"mean {r.mean():.1f} | max {r.max():.0f} | min {r.min():.0f} | std {r.std():.1f}", flush=True)
