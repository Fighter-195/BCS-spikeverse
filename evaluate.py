"""
evaluate.py — quick, configurable evaluation of the spiking agent.

Unlike Test.py (which is hard-wired to 100 episodes at 500 time-steps), this
script exposes the knobs that matter so you can measure results in minutes:

    python evaluate.py --episodes 10 --time-steps 200 --seed 0
    python evaluate.py --mode fixed         # compare action-selection variants

It also reports the mean output spike-count, which tells you whether the
spiking network is actually firing (i.e. contributing) or staying silent.
"""
import argparse
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description="Evaluate the SNN Breakout agent.")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--time-steps", type=int, default=200, help="spike-train length per state")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=0.1, help="exploration during eval")
    p.add_argument("--method", choices=["Binary", "Greyscale"], default="Binary")
    p.add_argument("--weights", default="weights/binary_model_weights.pth")
    p.add_argument("--scale1", type=float, default=20.0)
    p.add_argument("--scale2", type=float, default=100.0)
    p.add_argument("--mode", choices=["original", "fixed"], default="original",
                   help="'original' = repo's mean()/argmax greedy; "
                        "'fixed' = argmax over per-action spike counts")
    args = p.parse_args()

    # Override the spike-train length *before* importing the networks, so the
    # SNN default time_steps binds to our value.
    from config import config
    config["time_steps"] = args.time_steps
    from Environment import BreakoutEnv
    from Agent import SNNAgent, poisson_spike_encoding

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = BreakoutEnv(processing_method=args.method)
    agent = SNNAgent(state_shape=env.reset().shape, action_size=env.action_space.n)
    agent.load_ann_weights(filepath=args.weights, scale_1=args.scale1, scale_2=args.scale2)
    net = agent.policy_net
    device = net.device
    T = args.time_steps

    def choose(state):
        """Return (action, total_output_spikes) for one state."""
        if np.random.rand() < args.epsilon:
            return np.random.randint(agent.action_size), None
        enc = poisson_spike_encoding(state[None, :], time_steps=T)[0].float().to(device)
        net.reset()
        with torch.no_grad():
            q = net(enc)                       # [1, action_size] = spike counts per action
        if args.mode == "fixed":
            action = int(q.argmax(dim=1).item())       # pick the highest-firing action
        else:
            action = int(q.mean(dim=1).argmax().item())  # repo's original logic
        return action, float(q.sum().item())

    rewards, spike_totals = [], []
    print(f"Evaluating: mode={args.mode}  episodes={args.episodes}  T={T}  "
          f"eps={args.epsilon}  method={args.method}")
    for ep in range(args.episodes):
        state = env.reset()
        ep_reward = 0.0
        while True:
            action, spikes = choose(state)
            if spikes is not None:
                spike_totals.append(spikes)
            state, reward, done, trunc, _ = env.step(action)
            ep_reward += reward
            if done or trunc:
                break
        rewards.append(ep_reward)
        print(f"  ep {ep+1:>3}/{args.episodes}: reward={ep_reward:.1f}")
    env.close()

    rewards = np.array(rewards)
    mean_spikes = np.mean(spike_totals) if spike_totals else 0.0
    print("-" * 48)
    print(f"mean reward : {rewards.mean():.2f}")
    print(f"max reward  : {rewards.max():.0f}")
    print(f"std reward  : {rewards.std():.2f}")
    print(f"mean output spikes / greedy step : {mean_spikes:.2f}  "
          f"({'network is firing' if mean_spikes > 0 else 'network SILENT — Q all zero'})")


if __name__ == "__main__":
    main()
