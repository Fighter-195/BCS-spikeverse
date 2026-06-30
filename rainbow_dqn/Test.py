from Agent_Implementation import AdvancedDQNAgent
from Atari_Environment import AtariWrapper


env = AtariWrapper(env_name="BreakoutNoFrameskip-v4", frame_skip=4,rendering_mode='human')
state=env.reset()
state_shape = state.shape
action_size = env.action_space.n
agent=AdvancedDQNAgent(state_shape=state_shape,action_size=action_size)
print("Loading Model")
agent.load_model(filepath="model_weights.pth")
print("Model Loaded")
agent.epsilon=0.0
agent.epsilon_min=0.0


for _ in range (5):
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        # Agent selects action using epsilon-greedy strategy.
        action = agent.select_action(state)

        # Environment takes a step and returns the next state, reward, done flag, truncated flag, and info.
        next_state, reward, done, truncated, info = env.step(action)

        episode_reward += reward
        state = next_state

    print(f"Episode Reward: {episode_reward}")