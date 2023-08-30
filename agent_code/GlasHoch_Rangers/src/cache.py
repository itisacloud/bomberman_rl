import torch

class Memory:
    def __init__(self, input_dim: tuple[int, int, int], size: int):
        self.size = size
        self.index = 0

        self.states = torch.zeros((size, *input_dim), dtype=torch.float32)
        self.next_states = torch.zeros((size, *input_dim), dtype=torch.float32)
        self.actions = torch.zeros((size), dtype=torch.int32)
        self.rewards = torch.zeros((size), dtype=torch.int32)
        self.done = torch.zeros((size), dtype=torch.bool)

    def cache(self, state: torch.Tensor, next_state: torch.Tensor, action: int, reward: int, done: bool):
        if self.index >= self.size:
            self.index = 0

        self.states[self.index] = state
        self.next_states[self.index] = next_state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done

        self.index += 1

    def sample(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, self.size, (batch_size,))
        return self.states[indices], self.next_states[indices], self.actions[indices], self.rewards[indices], self.done[indices]
