import torch
from torch import nn, optim

import numpy as np
from pathlib import Path
import random, datetime, os, copy

from cache import Memory



class DQNetwork(nn.Module):
    '''mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    '''

    def __init__(self, input_dim, extra_dim, action_dim):
        super().__init__()
        c, h, w = input_dim

        self.extra_dim = extra_dim
        self.action_dim = action_dim


        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

class Agent():
    def __init__(self,state_dim, extra_dim, action_dim, AGENT_CONFIG):
        # Hyperparameters
        super().__init__(state_dim, extra_dim, action_dim, AGENT_CONFIG)
        self.save_every = None
        self.curr_step = None
        self.network_arch = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = DQNetwork()
        self.save_path = Path(__file__).parent / "model.pt"

        #setting up the network
        self.net = DQNetwork()
        self.net.to(self.device)

        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.memory = Memory(AGENT_CONFIG["state_dim"], AGENT_CONFIG["extra_dim"], AGENT_CONFIG["action_dim"], int(AGENT_CONFIG["memory_size"]))


    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        new_state,old_state,action,reward,done = self.memory.sample(self.batch_size)

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

    #tutorial with Double DQN
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q_online = self.net(next_state, model="online")
        best_action_online = torch.argmax(next_state_Q_online, axis=1)

        next_state_Q_target = self.net(next_state, model="target")
        next_Q_values = next_state_Q_target[np.arange(0, self.batch_size), best_action_online]

        td_targets = (reward + (1 - done.float()) * self.gamma * next_Q_values).float()

        return td_targets

    def save(self):
        if self.save_dir is None:
            print("Cannot save model. No save directory given.")
            return
        if self.curr_step % self.save_every != 0:
            return
        save_path = (
                self.save_dir / f"agent_{int(self.curr_step // self.save_every)}.pth"
        )
        torch.save(self.net.state_dict(), save_path)


