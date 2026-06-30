import numpy as np
import pandas as pd
from typing import Tuple, List
import random
from config import config

class SumTree:
    def __init__(self, capacity: int):
        """Initialize tree structure for efficient priority sampling"""
        # Setup binary tree arrays and tracking variables
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write_idx = 0
        

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority changes up the tree"""
        # Update parent nodes recursively to root
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
        

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve leaf index for given priority value"""
        # Navigate tree based on cumulative priorities
        left = 2 * idx + 1
        right = 2 * idx + 2

        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])


    def total_priority(self) -> float:
        """Return total priority (root value)"""
        return self.tree[0]

    def add(self, priority: float, data: object) -> None:
        """Store experience with priority"""
        # Add data to tree and propagate changes
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx += 1
        if self.write_idx >= self.capacity:
            self.write_idx = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1
        

    def update(self, idx: int, priority: float) -> None:
        """Update priority of existing experience"""
        # Update node and propagate change
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
        

    def get_leaf(self, s: float) -> Tuple[int, float, object]:
        """Sample leaf based on priority value s"""
        # Find leaf and return index, priority, data
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
    

from collections import deque

class NStepBuffer:
    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.buffer = deque()

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) < self.n:
            return None

        return self._get_n_step_transition()

    def _get_n_step_transition(self):
        state, action = self.buffer[0][0], self.buffer[0][1]
        R = 0
        for i in range(self.n):
            r, d = self.buffer[i][2], self.buffer[i][4]
            R += (self.gamma ** i) * r
            if d:
                break
        next_state, done = self.buffer[-1][3], self.buffer[-1][4]
        self.buffer.popleft()
        return state, action, R, next_state, done

    def flush(self):
        transitions = []
        while len(self.buffer) > 0:
            trans = self._get_n_step_transition()
            if trans:
                transitions.append(trans)
        return transitions

    
class PrioritizedReplayBuffer:
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4):
        """Initialize prioritized replay buffer"""
        # Set up the SumTree and store prioritization parameters.
        # Initialize beta annealing schedule and numerical stability constants.
        # Set up frame counting for importance sampling weight calculation.

        self.tree = SumTree(capacity)
        self.capacity = capacity

        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment_per_sampling = (1.0 - beta_start) / capacity
        self.frame_count = 0

        self.epsilon = 1e-6  # Small value for numerical stability

    
    def _get_beta(self) -> float:
        """Calculate current beta value (anneals to 1.0)"""
        # Implement linear annealing from beta_start to 1.0 over training.
        beta = min(1.0, self.beta + self.beta_increment_per_sampling * self.frame_count)
        self.frame_count += 1
        return beta
    
    def push(self, state, action, reward, next_state, done) -> None:
        """Store experience with maximum priority"""
        # Package the experience tuple and assign appropriate priority.
        # Use maximum existing priority for new experiences to ensure sampling.
        experience = (state, action, reward, next_state, done)

        if self.tree.n_entries == 0:
            max_priority = 1.0
        else :
            leaf_start = self.capacity - 1
            leaf_end = leaf_start + self.tree.n_entries
            max_priority = np.max(self.tree.tree[leaf_start:leaf_end])
    
            if max_priority == 0:
                max_priority = 1.0

        self.tree.add(float(max_priority), experience)
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights"""
        # Divide priority range into segments for stratified sampling.
        # Calculate importance sampling weights to correct for sampling bias.
        # Return experiences, tree indices, and normalized weights.

        batch = []
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)

        total_priority = self.tree.total_priority()
        segment = total_priority / batch_size

        beta = self._get_beta()

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx, p, data = self.tree.get_leaf(s)
            batch.append(data)
            indices[i] = idx
            priorities[i] = p

        N = self.tree.n_entries
        if N == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        
        priorities = np.maximum(priorities, 1e-6)  # Ensure no zero priorities for numerical stability
        probs = priorities / total_priority
        weights = (N * probs) ** (-beta)
        max_weight = np.max(weights)
        if max_weight > 0:
            weights /= max_weight
        else :
            weights.fill(1.0)
        
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities) -> None:
        """Update priorities based on TD errors"""
        # Convert TD errors to priorities using alpha exponent.
        # Add small epsilon for numerical stability.
        # Update tree nodes with new priority values.

        for idx, error in zip(indices, priorities):
            new_priority = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, new_priority)
    
    def __len__(self) -> int:
        """Return current buffer size"""
        # Return the number of experiences currently stored.
        return self.tree.n_entries
