import torch

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

rotated_actions = {
    0: 1,
    1: 2,
    2: 3,
    3: 0,
    4: 4,
    5: 5
}

class Memory:
    def rotateFeature(self, rots, feature):
        return torch.rot90(feature, rots, (0, 1))

    def rotateFeatures(self, rots, features):
        return torch.stack([self.rotateFeature(rots, feature) for feature in features])

    def rotateAction(self, rots, action):
        for _ in range(rots):
            action = rotated_actions[action]
        return action

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

        rotation = torch.randint(0, 4, (1,)).item()  # Generate a random rotation
        rotated_state = self.rotateFeatures(rotation, state)
        rotated_next_state = self.rotateFeatures(rotation, next_state)
        rotated_action = self.rotateAction(rotation, action)

        self.states[self.index] = rotated_state
        self.next_states[self.index] = rotated_next_state
        self.actions[self.index] = rotated_action
        self.rewards[self.index] = reward
        self.done[self.index] = done

        self.index += 1
        self.counter += 1

    def sample(self, batch_size: int = 1) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, min(self.counter, self.size), (batch_size,), device=self.device)

        return (
            self.states[indices],
            self.next_states[indices],
            self.actions[indices],
            self.rewards[indices].squeeze(),
            self.done[indices].squeeze()
        )
