import numpy as np
import torch
actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT',"BOMB"]

rotated_actions = {
    0:1,
    1:2,
    2:3,
    3:0,
    4:4,
    5:5
}


class Memory:
    def rotateFeature(self, rots,feature):
        return torch.rot90(feature, rots, (0, 1))

    def rotateFeatures(self, rots,features):
        return torch.stack([self.rotateFeature(rots, features[idx]) for idx in range(features.shape[0])])

    def rotateAction(self, rots,action):
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
        indices = torch.randint(0, min(self.counter, self.size), (batch_size,),device=self.device)
        rotation = torch.randint(0, 4, (batch_size,),device=self.device)

        if np.random.rand() < 0.5:
            rotated_states = self.states[indices]
            rotated_next_states = self.next_states[indices]
            rotated_actions = self.actions[indices]
        else:
            rotated_states = torch.stack([self.rotateFeatures(rot, self.states[idx]) for idx, rot in zip(indices, rotation)])
            rotated_next_states = torch.stack([self.rotateFeatures(rot, self.next_states[idx]) for idx, rot in zip(indices, rotation)])
            rotated_actions = torch.tensor([self.rotateAction(rot, self.actions[idx]) for idx, rot in zip(indices, rotation)], dtype=torch.int32)

        return (
            rotated_states,
            rotated_next_states,
            rotated_actions,
            self.rewards[indices].squeeze(),
            self.done[indices].squeeze()
        )