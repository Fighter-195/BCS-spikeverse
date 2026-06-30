import numpy as np
from collections import deque
from Agent_Implementation import AdvancedDQNAgent
from Atari_Environment import AtariWrapper
from config import config
from tqdm import tqdm

# Initialize the Atari environment wrapper and DQN agent.
env = AtariWrapper(env_name="BreakoutNoFrameskip-v4", frame_skip=4)
state=env.reset()
state_shape = state.shape
action_size = env.action_space.n
def train_agent(agent=AdvancedDQNAgent(state_shape, action_size, config=config),env=env):
    """Main training loop with comprehensive monitoring"""
    

    # Set up tracking variables for metrics and timing.
    episode_rewards = []
    mean_scores = []
    losses = []
    scores_window = deque(maxlen=50)
    total_steps = 0

    # Implement the main episode loop with proper environment interaction.
    for i in range(config['max_episodes']//config['logging_interval']):
        print(f"Training for episode {i*config['logging_interval']+1} to {i*config['logging_interval']+config['logging_interval']}")
        for episode in tqdm(range(config['logging_interval'])):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_loss = []

            while not done:
                # Agent selects action using epsilon-greedy strategy.
                action = agent.select_action(state)

                # Environment takes a step and returns the next state, reward, done flag, truncated flag, and info.
                next_state, reward, done, truncated, info = env.step(action)

                # Handle experience storage, agent updates, and target network synchronization.
                agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                total_steps += 1

                loss = agent.update()
                if loss is not None:
                    episode_loss.append(loss)

                # Synchronize target network weights at regular intervals.
                if total_steps % agent.update_target_freq == 0:
                    agent.update_target_network()
            # agent.decay_epsilon()

            # Monitor training progress with comprehensive logging and statistics.
            scores_window.append(episode_reward)
            mean_score = np.mean(scores_window)
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            episode_rewards.append(episode_reward)
            mean_scores.append(mean_score)
            losses.append(avg_loss)


        print(f"Episode {(i+1)*config['logging_interval']} | Reward: {episode_reward:.2f} | "
              f"Mean Score: {mean_score:.2f} | Loss: {avg_loss:.4f}")

        # Implement early stopping when target performance is achieved.
        if mean_score >= config['target_score']:
            print(f"\nAchieved the Human Average in {((i+1)*config['logging_interval'])} episodes! Average score: {mean_score:.2f}")
            print
    
    return agent,episode_rewards, mean_scores, losses