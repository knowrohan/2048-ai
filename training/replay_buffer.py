import random
import torch

class ReplayBuffer:
    def __init__(self, capacity=10000):
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
        return torch.stack(states), torch.tensor(policies, dtype=torch.float32), torch.tensor(values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.buffer)
