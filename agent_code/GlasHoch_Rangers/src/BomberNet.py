import copy
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .cache import Memory



class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(512 * (h // 4) * (w // 4), 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        input = input.unsqueeze(0)

        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


class Agent():
    def __init__(self,AGENT_CONFIG,REWARD_CONFIG,training):

        self.training= training

        self.exploration_rate_min = AGENT_CONFIG["exploration_rate_min"]
        self.exploration_rate_decay = AGENT_CONFIG["exploration_rate_decay"]
        self.exploration_rate = AGENT_CONFIG["exploration_rate"]
        self.state_dim = AGENT_CONFIG["state_dim"]
        self.action_dim = AGENT_CONFIG["action_dim"]
        self.batch_size = AGENT_CONFIG["batch_size"]
        self.memory_size = AGENT_CONFIG["memory_size"]
        self.save_every = AGENT_CONFIG["save_every"]
        self.curr_step = 0

        # Hyperparameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = "./models"

        # setting up the network
        self.net = DQNetwork(input_dim=self.state_dim,output_dim=self.action_dim).float()
        self.net.to(self.device)

        self.burnin = AGENT_CONFIG["burnin"]  # min. experiences before training
        self.learn_every = AGENT_CONFIG["learn_evry"] # no. of experiences between updates to Q_online
        self.sync_every = AGENT_CONFIG["sync_evry"]  # no. of experiences between Q_target & Q_online sync

        self.gamma = AGENT_CONFIG["gamma"]

        # discount factor
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.memory = Memory(AGENT_CONFIG["state_dim"],self.memory_size)

        self.REWARD_CONFIG = REWARD_CONFIG

        if AGENT_CONFIG["load"] == True: # load model :D
            self.load(AGENT_CONFIG["load_path"])

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        new_state, old_state, action, reward, done = self.memory.sample(self.batch_size)


        # Get TD Estimate
        td_est = self.td_estimate(old_state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, new_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    # tutorial with Double DQN
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q_online = self.net(next_state, model="online")
        best_action_online = torch.argmax(next_state_Q_online, axis=1)

        next_state_Q_target = self.net(next_state, model="target")
        next_Q_values = next_state_Q_target[np.arange(0, self.batch_size), best_action_online]

        td_targets = (reward + (1 - done.float()) * self.gamma * next_Q_values).float()

        return td_targets

    def act(self,features):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            action_values = self.net(features, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

            # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    def save(self):
        if self.save_dir is None:
            print("Cannot save model. No save directory given.")
            return
        if self.curr_step % self.save_every != 0:
            return
        save_path = (
                self.save_dir + f"/agent_{int(self.curr_step)}.pth"
        )
        torch.save(self.net.state_dict(), save_path)

    def load(self, model_path):
        print(model_path)
        print(os.path.abspath(model_path)
)

        if not Path(model_path).is_file():
            print(f"Model file not found at {model_path}. Cannot load the model.")
            return

        self.net.load_state_dict(torch.load(model_path))
        self.sync_Q_target()  # Sync the target network with the loaded model's parameters



