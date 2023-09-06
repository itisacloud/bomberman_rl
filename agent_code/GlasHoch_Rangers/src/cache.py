import torch

class Memory:
    def __init__(self, input_dim: tuple[int, int, int], size: int):
        print("Memory")
        self.size = size
        self.counter = 0
        self.index = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.states = torch.zeros((size, *input_dim), dtype=torch.float32).to(self.device)
        self.next_states = torch.zeros((size, *input_dim), dtype=torch.float32).to(self.device)
        self.actions = torch.zeros((size), dtype=torch.int32).to(self.device)
        self.rewards = torch.zeros((size), dtype=torch.int32).to(self.device)
        self.done = torch.zeros((size), dtype=torch.bool).to(self.device)


    def cache(self, state: torch.Tensor, next_state: torch.Tensor, action: int, reward: int, done: bool):
        if self.index >= self.size:
            self.index = 0

        self.states[self.index] = state
        self.next_states[self.index] = next_state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done

        self.index += 1
        self.counter += 1


    def sample(self, batch_size: int = 1) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, min(self.counter, self.size), (batch_size,))
        return (
            self.states[indices].squeeze().to(self.device),
            self.next_states[indices].squeeze().to(self.device),
            self.actions[indices].squeeze().to(self.device),
            self.rewards[indices].squeeze().to(self.device),
            self.done[indices].squeeze().to(self.device)
        )