
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import optim

from .BomberNet import BomberNet
from .expert import Expert

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT',"BOMB"]

reversed = {"UP": 0,
            "RIGHT": 1,
            "DOWN": 2,
            "LEFT": 3,
            "WAIT": 4,
            "BOMB": 5
            }

class Agent():
    def __init__(self, AGENT_CONFIG, REWARD_CONFIG, training):
        # ... (same initialization code as before)
        self.training = training
        # Define a policy network
        self.policy_net = BomberNet(input_dim=self.state_dim, output_dim=self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=AGENT_CONFIG["learning_rate"])

    def learn(self, memory):
        if self.curr_step < self.burnin:
            return None, None

        new_state, old_state, action, reward, done = memory.sample(self.batch_size)

        # Compute the policy loss
        action_probs = self.policy_net(old_state)
        selected_action_probs = action_probs.gather(1, action.unsqueeze(1))
        policy_loss = -torch.log(selected_action_probs) * reward

        # Compute and apply gradients
        self.optimizer.zero_grad()
        policy_loss.mean().backward()
        self.optimizer.step()

        return None, policy_loss.mean().item()

    def act(self, features, state=None):
        if self.training:
            action_probs = self.policy_net(features)
            action_distribution = torch.distributions.Categorical(action_probs)
            action_idx = action_distribution.sample().item()
        else:
            # During evaluation, select the action with the highest probability
            action_probs = self.policy_net(features)
            action_idx = torch.argmax(action_probs, dim=1).item()

    def save(self,save_config = True):

        if self.save_dir is None:
            print("Cannot save model. No save directory given.")
            return
        self.since_last_save += 1

        if self.curr_step % self.save_every != 0 and self.since_last_save <= self.save_every:
            return
        self.since_last_save = 0
        print(self.__str__())

        # Save the model
        model_save_path = (
                self.save_dir + f"/{self.agent_name}_{self.config_name}_{int(self.curr_step)}.pth"
        )
        torch.save(self.net.state_dict(), model_save_path)

        # Save the configuration
    def load(self, model_path):
        print(model_path)
        print(os.path.abspath(model_path))

        if not Path(model_path).is_file():
            print(f"Model file not found at {model_path}. Cannot load the model.")
            return
            # Sync the target network with the loaded model's parameters

