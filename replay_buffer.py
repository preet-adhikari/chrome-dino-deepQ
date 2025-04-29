import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        if state is None or next_state is None:
            print("⚠️ Warning: Tried to add None frame to buffer.")
        self.buffer.append((state, action, reward, next_state, done))
       

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
