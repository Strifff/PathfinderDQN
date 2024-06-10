from collections import deque
import numpy as np


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        if len(self.memory) >= batch_size:
            indices = np.random.choice(len(self.memory), batch_size, replace=False)
        else:
            indices = np.random.choice(
                len(self.memory), len(self.memory), replace=False
            )

        sampled_transitions = [self.memory[i] for i in indices]
        # for i, transition in enumerate(sampled_transitions):
        #     print(f"Sampled element {i}: {transition}")

        return sampled_transitions

    def __len__(self):
        return len(self.memory)
