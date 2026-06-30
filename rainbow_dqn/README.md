# 🎯 Rainbow-style Conv DQN — Atari Breakout (avg 169 / peak 290)

This folder contains the **convolutional** Deep-Q agent that actually masters Breakout — the
strong model behind the SpikeVerse project's headline numbers. It's a Nature-DQN backbone with
several **Rainbow** components layered on top.

> **Credit:** this agent is the work of teammate **[@adianandgit](https://github.com/adianandgit/Atari-Breakout-SNNs)**
> (group project). Included here under its original **MIT License** (see `LICENSE`). The only change
> from the source is the `gym → gymnasium` import in `Atari_Environment.py` (required to run on
> modern `ale-py`); all model logic is unchanged.

---

## ✅ Verified result

Running the trained `model_weights.pth` greedily (ε = 0, deterministic `BreakoutNoFrameskip-v4`):

```
20/20 episodes → reward = 290 each   (mean 290, std 0)
```

290 is the deterministic optimal trajectory of this policy; the README's "avg 169" is the
*training* average measured **with** exploration. Either way, this agent far exceeds the human
baseline of 32.

## 🚀 What's implemented
- **Double DQN (DDQN)** — decoupled action selection/evaluation (`training_function.py`, `Agent_Implementation.py`)
- **Prioritized Experience Replay (PER)** — TD-error-weighted sampling (`PER.py`)
- **N-step returns** — better credit assignment in sparse-reward Breakout (`PER.py: NStepBuffer`)
- **Noisy / Dueling networks** — optional exploration & value/advantage decomposition (`NOISY_Dueling_DQN.py`)
- **Conv DQN backbone** — `8×8/4×4/3×3` conv → FC 512 → 4 actions (`DQN_Architecture.py`)
- **Standard Atari preprocessing** — RGB→gray→84×84→[0,1], 4-frame stack (`Image_Processing_and_Frame_Stacking.py`)

## 🧩 Architecture
```
4 × 84×84 stacked frames
   → Conv2d(4→32, k8,s4) → ReLU
   → Conv2d(32→64, k4,s2) → ReLU
   → Conv2d(64→64, k3,s1) → ReLU
   → Flatten(3136) → Linear(512) → ReLU → Linear(4)   → Q-values [NOOP, FIRE, RIGHT, LEFT]
```

## ▶️ Usage (run from this folder)
```bash
pip install -r requirements.txt

# Headless evaluation (no render window), prints mean/max over N episodes
python eval_headless.py --episodes 20

# Watch it play (opens a render window)
python Test.py

# Train / fine-tune from the existing weights
python Main.py     # remove lines 12-13 of Main.py to train from scratch
```

## 📊 Training curves
See `fine_tuned_results.png` and `finetuned_results_overview.png` — the DDQN reached a trailing
50-episode average of ~81.6, then N-step fine-tuning pushed it to **avg 169 / peak 290**.

## 🔗 Relation to the SNN part of this repo
This is the **ANN** that plays well. The repo's root contains the separate **FC → SNN conversion**
experiment (mean ~1.13) — the spiking, neuromorphic side of the project. Converting *this*
convolutional model to a spiking network (spiking conv layers over a time window) is an open
extension, not yet done.
