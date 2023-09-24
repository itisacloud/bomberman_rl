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




class cache:
    def rotateFeature(self, rots, feature):
        return torch.rot90(feature, rots, (0, 1))

    def rotateFeatures(self, rots,features):
        return torch.stack([self.rotateFeature(rots, features[idx]) for idx in range(features.shape[0])])

    def rotateAction(self, rots,action):
        action = action.clone()
        for _ in range(rots):
            action = rotated_actions[int(action)]
        return torch.tensor(action)  # Convert back to a tensor


    def __init__(self, input_dim: tuple[int, int, int], size: int,rotation_augment=False,rotation_augment_prob=0.5):
        print("Memory")
        self.size = size
        self.counter = 0
        self.index = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.features = torch.zeros((size, *input_dim), dtype=torch.float32).to(self.device)
        self.new_features = torch.zeros((size, *input_dim), dtype=torch.float32).to(self.device)
        self.actions = torch.zeros((size), dtype=torch.int32).to(self.device)
        self.rewards = torch.zeros((size), dtype=torch.float32).to(self.device)
        self.done = torch.zeros((size), dtype=torch.bool).to(self.device)
        self.rotation_augment = rotation_augment
        self.rotation_augment_prob = rotation_augment_prob

        self.priorities = torch.zeros((size), dtype=torch.float32).to(self.device)



    def cache(self, state: torch.Tensor, next_state: torch.Tensor, action: int, reward: int, done: bool):
        if self.index >= self.size:
            self.index = 0
        self.features[self.index] = state
        self.new_features[self.index] = next_state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done

        self.index += 1
        self.counter += 1


    def sample(self, batch_size: int = 1) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, min(self.counter, self.size), (batch_size,),device=self.device)
        rotation = torch.randint(0, 4, (batch_size,),device=self.device)

        if np.random.rand() < self.rotation_augment_prob and self.rotation_augment:
            rotated_features = torch.stack([self.rotateFeatures(rot, self.features[idx]) for idx, rot in zip(indices, rotation)])
            rotated_new_features = torch.stack([self.rotateFeatures(rot, self.new_features[idx]) for idx, rot in zip(indices, rotation)])
            rotated_actions = torch.tensor([self.rotateAction(rot, self.actions[idx]) for idx, rot in zip(indices, rotation)], dtype=torch.int32)
        else:
            rotated_features = self.features[indices]
            rotated_new_features = self.new_features[indices]
            rotated_actions = self.actions[indices]

        return (
            rotated_features,
            rotated_new_features,
            rotated_actions,
            self.rewards[indices].squeeze(),
            self.done[indices].squeeze()
        )