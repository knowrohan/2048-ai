"""
Experience replay buffer for storing self-play game data.
Stores states, policies, and values, and allows sampling batches for training.
"""
import random
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, policy, value):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, policy, value)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        # Convert values to float32 explicitly before converting to tensor
        return (
            torch.stack(states),
            torch.from_numpy(np.array(policies, dtype=np.float32)),
            torch.from_numpy(np.array(values, dtype=np.float32)).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)
