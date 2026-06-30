import numpy as np
from typing import Tuple
import gymnasium as gym
from gymnasium import spaces
from Image_Processing_and_Frame_Stacking import FrameStack, preprocess_frame
import ale_py
import os

# Modern ale-py registers Atari envs with Gymnasium, not the unmaintained gym.
gym.register_envs(ale_py)

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

class AtariWrapper:
    def __init__(self, env_name: str = "BreakoutNoFrameskip-v4", frame_skip: int = 4,rendering_mode: str = "rgb_array"):
        """Initialize environment and frame stack"""
        # Create the Gym environment and frame stack manager.
        self.env = gym.make(env_name,render_mode=rendering_mode)
        self.frame_stack = FrameStack(maxlen=4)

        # Define the action space mapping for Breakout game.
        self.action_space = spaces.Discrete(4)     # 0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT
        self.observation_space = self.env.observation_space

        # Store frame skipping parameter for temporal efficiency.
        self.frame_skip = frame_skip
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial stacked state"""
        # Reset both the environment and frame stack.
        initial_obs,_=self.env.reset()
        self.frame_stack = FrameStack(maxlen=4)

        # Process the initial observation and create the first state stack.
        processed_frame = preprocess_frame(initial_obs)

        self.frame_stack.push(processed_frame)
        state = self.get_state()

        # Return the properly formatted state for the agent.
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action with frame skipping"""
        # Convert agent action to environment action.
        agent_action = action
        # Execute the action multiple times to skip frames.
        total_reward = 0.0
        done = False
        for _ in range(self.frame_skip):
            obs, reward, done, truncated, info = self.env.step(agent_action)
            total_reward += reward
            
            # Process the observation and add it to the frame stack.
            processed_frame = preprocess_frame(obs)
            self.frame_stack.push(processed_frame)
            
            # If the episode is done, break out of the loop.
            if done or truncated:
                break
            

        # Accumulate rewards and process the final frame.
        self.state= self.get_state()
        # Return the new state, total reward, done flag, truncated, and info.
        return self.state, total_reward, done, truncated, info
        
    
    def get_state(self) -> np.ndarray:
        """Return current stacked state"""
        # Returning the current state stack from the frame stack.
        return self.frame_stack.get_stack()
    
    def render(self):
        self.env.render()
    
    def close(self):
        """Close the environment"""
        # Properly close the Gym environment to free resources.
        self.env.close()