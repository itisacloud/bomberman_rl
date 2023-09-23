import numpy as np
import torch

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', "BOMB"]

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
        return torch.stack([self.rotateFeature(rots, features[idx]) for idx in range(features.shape[0])])

    def rotateAction(self, rots, action):
        action = action.clone()
        for _ in range(rots):
            action = rotated_actions[int(action)]
        return torch.tensor(action)  # Convert back to a tensor

    def __init__(self, input_dim: tuple[int, int, int], size: int):
        print("Memory")
        self.size = size
        self.counter = 0
        self.index = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.features = torch.zeros((size, *input_dim), dtype=torch.float32).to(self.device)
        self.actions = torch.zeros(size, dtype=torch.int32).to(self.device)
        self.rewards = torch.zeros(size, dtype=torch.int32).to(self.device)
        self.log_pis = torch.zeros(self.worker_steps, dtype=torch.float32).to(self.device)
        self.advantages = torch.zeros(self.worker_steps, dtype=torch.float32).to(self.device)
        self.returns = torch.zeros(self.worker_steps, dtype=torch.float32).to(self.device)
        self.done = torch.zeros(size, dtype=torch.bool).to(self.device)

    def cache(self, value, action, log_pis, reward, feature, done):
        self.features[self.index] = feature
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.log_pis[self.index] = log_pis
        self.values[self.index] = value
        self.done[self.index] = done

        self.index += 1

    def sample(self):
        return {
            'features': self.features,
            'actions': self.actions,
            'values': self.values,
            'log_pis': self.log_pis,
            'advantages': self.advantages,
            'returns': self.returns
        }
