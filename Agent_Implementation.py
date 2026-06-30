from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from config import config
from DQN_Architecture import DQN
from PER import PrioritizedReplayBuffer
from PER import NStepBuffer

def compute_double_dqn_loss(policy_net, target_net, states, actions, 
                             rewards, next_states, dones, gamma, is_weights):
    """Compute Double DQN loss with importance sampling"""

    # Current Q-values from policy network for the taken actions
    q_values = policy_net(states).gather(1, actions)

    # Select next actions using the policy network (Double DQN trick)
    next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)

    # Evaluate selected actions using the target network
    next_q_values = target_net(next_states).gather(1, next_actions)

    # Bellman target with detached target network
    target_q_values = rewards + (1 - dones) * gamma * next_q_values
    target_q_values = target_q_values.detach()  # Detach to stop gradients

    # Compute TD error (target - prediction)
    td_errors = target_q_values - q_values
    
    # Apply importance sampling weights if provided to correct sampling bias.
    if is_weights is not None:
        loss = (td_errors ** 2 * is_weights).mean()
    else:
        # Compute the mean squared error loss without importance sampling.
        loss = (td_errors ** 2).mean()
    # Return both the loss value and TD errors for priority updates.
    return loss, td_errors


class AdvancedDQNAgent:
    def __init__(self, state_shape, action_size, config=config):
        """Initialize agent with networks and replay buffer"""
        # Store network dimensions and configuration parameters.
        self.state_shape = state_shape
        self.action_size = action_size
        self.config = config
        # Create policy and target networks with identical architectures.
        self.policy_net = DQN(self.state_shape, self.action_size)
        self.policy_net.to(self.policy_net.device)
        self.target_net = DQN(self.state_shape, self.action_size)
        self.target_net.to(self.target_net.device)
        # Initialize the target network with policy network weights.
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Set up the optimizer and prioritized replay buffer.
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'])
        self.replay_buffer = PrioritizedReplayBuffer(config['buffer_size'], 
                                                     alpha=config['alpha'], 
                                                     beta_start=config['beta_start'])
        # Configure exploration parameters for epsilon-greedy strategy.
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = (1.0 - self.epsilon_min) / (config['max_episodes'])
        self.update_target_freq = config['target_update_freq']

        self.n_step = config['n_step']
        self.n_step_buffer = NStepBuffer(self.n_step, config['gamma'])

    
    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        self.decay_epsilon()
        # Choose between random exploration and greedy exploitation.
        if np.random.rand() < self.epsilon:
            # Random action for exploration.
            return np.random.randint(self.action_size)
        # For greedy actions, use the policy network to select best action.
        else:
        # Convert state to tensor and move to the appropriate device.
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state_tensor = state_tensor.to(self.policy_net.device)
            # Get Q-values from the policy network and select action with max Q-value.
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()


    def store_transition(self, state, action, reward, next_state, done):
        transition = self.n_step_buffer.push(state, action, reward, next_state, done)

        if transition:
            self.replay_buffer.push(*transition)

        if done:
            # flush remaining n-step transitions
            for t in self.n_step_buffer.flush():
                self.replay_buffer.push(*t)
    
    def update(self) -> Optional[float]:
        """Perform learning update"""
        # Check if sufficient experiences are available for training.
        if len(self.replay_buffer) < self.config['initial_replay_size']:
            return None

        # Sample a batch from the prioritized replay buffer.
        experiences, indices, weights = self.replay_buffer.sample(self.config['batch_size'])
        if not isinstance(experiences, (list, tuple)) or not all(isinstance(e, (list, tuple)) and len(e) == 5 for e in experiences):
            return None
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert experience components to tensors and move to device.
        device = self.policy_net.device
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

        
        loss, td_error = compute_double_dqn_loss(self.policy_net,self.target_net,states,actions,rewards,next_states,dones,self.config['gamma'],weights)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        new_priorities = td_error.abs().detach().cpu().numpy().squeeze() + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)

        return loss.item()

    
    def update_target_network(self):
        """Copy weights from policy to target network"""
        # Synchronize target network weights with policy network.
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay epsilon for exploration"""
        # Reduce epsilon value according to decay schedule.
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def save_model(self, filepath: str="model_weights.pth"):
        """Save model weights to file"""
        # Save the policy network state dictionary to the specified filepath.
        torch.save(self.policy_net.state_dict(), filepath)

    def load_model(self, filepath: str="model_weights.pth"):
        """Load model weights from file"""
        # Load the policy network state dictionary from the specified filepath.
        self.policy_net.load_state_dict(torch.load(filepath))
        self.update_target_network()