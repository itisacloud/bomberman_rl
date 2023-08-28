import numpy as np


class Memory:
    def __init__(self, input_dim: tuple[int, int, int], size: int):
        self.size = size
        self.index = 0

        self.states = np.zeros((size, *input_dim), dtype=np.float32)
        self.next_states = np.zeros((size, *input_dim), dtype=np.float32)
        self.actions = np.zeros((size), dtype=np.str_)
        self.rewards = np.zeros((size), dtype=np.int32)

    def cache(self, state: np.ndarray, next_state: np.ndarray, action: str, reward: int):
        if self.index >= self.size:
            self.index = 0

        self.states[self.index] = state
        self.next_states[self.index] = next_state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.index += 1

    def sample(self, batch_size: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(self.size, batch_size, replace=False)
        return self.states[indices], self.next_states[indices], self.actions[indices], self.rewards[indices]
