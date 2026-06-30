# SpikeVerse — Mastering Atari Breakout, then Going Neuromorphic

This project has two acts. First, it trains a deep reinforcement-learning agent that genuinely
plays Atari Breakout well. Second, it converts that trained network into a Spiking Neural Network,
trading dense floating-point computation for sparse, event-driven spikes — the brain-like regime
that neuromorphic hardware is built for.

The point of the project is to study what is gained and lost in that trade: a conventional deep
agent reaches strong performance, while its spiking counterpart explores energy efficiency,
robustness, and biological plausibility at the cost of raw score.

## Results

| Agent | Method | Score |
|---|---|---|
| Convolutional Deep-Q Network | Double-DQN + PER + N-step (1500 fine-tuning episodes) | average 169, peak 290 |
| Conv DQN, greedy evaluation | deterministic, no exploration | 290 in every episode (verified, 20/20) |
| Spiking conversion (fully connected) | direct weight transfer, LIF neurons, Poisson coding | mean reward ≈ 1.13 |

For reference, the human baseline on Breakout is about 32. Training curves for the DQN are in
`plots/`.

## Repository layout

```
.
├── README.md
├── requirements.txt
│
│   Main project — the convolutional Rainbow-style DQN agent
├── DQN_Architecture.py                     # conv backbone: 8x8 / 4x4 / 3x3 -> FC512 -> 4 actions
├── Agent_Implementation.py                 # Double-DQN agent: epsilon-greedy, PER, n-step storage
├── PER.py                                  # prioritized replay buffer + n-step buffer
├── NOISY_Dueling_DQN.py                    # optional noisy-linear + dueling value/advantage streams
├── training_function.py                    # Double-DQN loss and optimization step
├── Atari_Environment.py                    # Breakout wrapper (Gymnasium), frame-skip, 4-frame stack
├── Image_Processing_and_Frame_Stacking.py  # grayscale -> 84x84 -> normalize -> stack
├── Main.py                                 # training / fine-tuning loop
├── Test.py                                 # watch the trained agent play (render window)
├── eval_headless.py                        # score the agent without a window
├── Visualisation_and_plotting.py           # training-metric plots
├── model_weights.pth                       # trained conv-DQN weights
├── plots/                                  # DQN training curves
│
│   Reinforcement-learning fundamentals
├── learning_journey/
│   └── week1_rl_foundations.ipynb          # Q-learning, GridWorld, replay buffer, DQN
│
│   Act two — the spiking conversion
├── snn_conversion/                         # fully-connected DQN converted to an SNN
│   ├── Neuron.py  Architectures.py  Agent.py  Environment.py
│   ├── Train.py  Test.py  evaluate.py  config.py
│   └── weights/  plots/  sample_frames/
│
└── docs/Documentation.pdf                  # full project write-up
```

## Act one: the convolutional DQN

The agent builds on a vanilla Deep-Q Network with the additions that make Atari learning stable
and sample-efficient:

- Double DQN, which decouples action selection from evaluation to reduce Q-value overestimation.
- Prioritized Experience Replay, which replays high-error transitions more often.
- N-step returns, which propagate reward over several steps for better credit assignment in
  Breakout's sparse-reward setting.
- Optional noisy and dueling heads, for learned-noise exploration and a value/advantage split.
- The classic DeepMind convolutional backbone (Conv 32-64-64 to a 512-unit dense layer to four
  action outputs) over four stacked 84x84 grayscale frames.

Run it from the repository root:

```bash
pip install -r requirements.txt

python eval_headless.py --episodes 20   # score it without a window
python Test.py                          # watch it play in a window
python Main.py                          # fine-tune from model_weights.pth
```

## Act two: the spiking conversion

Rather than training a spiking network directly (surrogate-gradient training is unstable and slow
for deep RL), this stage transfers the learned weights into a spiking architecture. ReLU units
become Leaky Integrate-and-Fire neurons that accumulate input current and emit a binary spike when
the membrane crosses threshold. Input frames are turned into spike trains through Poisson rate
coding, and the action with the most accumulated output spikes is chosen. Layer weights are scaled
(20x and 100x) so deeper layers keep firing.

The spiking model is the simpler fully-connected variant on purpose: it demonstrates the ANN-to-SNN
pipeline and its trade-offs — sparse spikes, longer simulation per decision, and a lower score —
rather than competing with the convolutional agent on performance.

```bash
cd snn_conversion
python evaluate.py --episodes 10 --time-steps 200   # quick spiking evaluation
python Test.py                                       # full evaluation (slow: 500 time-steps/decision)
```

## Core equations

Q-learning temporal-difference update:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\big[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,\big]$$

N-step return:

$$G_t^{(n)} = \sum_{k=0}^{n-1}\gamma^{k} r_{t+k+1} + \gamma^{n} V(s_{t+n})$$

LIF membrane dynamics (the neuron fires and resets when the potential reaches threshold):

$$\tau \frac{dv(t)}{dt} = -\big(v(t)-v_\text{rest}\big) + \sum_i W_i\,\text{Input}_i$$

## Setup notes

- Python 3.9 or newer. A CUDA GPU is recommended, especially for the spiking model, which runs the
  network for many time-steps per environment step.
- The environment uses Gymnasium with ale-py for Atari registration. A single import swap from the
  legacy gym package is the only change needed to run on a current stack.
- Run the convolutional agent from the repository root, and the spiking agent from inside
  `snn_conversion/`; each resolves its weights and plots relative to its own folder.
