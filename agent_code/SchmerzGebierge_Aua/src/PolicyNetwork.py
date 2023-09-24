import copy
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.cuda import device
from torch.distributions import Categorical
from torch.nn.parallel import DataParallel  # Import DataParallel
from Memory import Memory

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', "BOMB"]

reversed = {"UP": 0,
            "RIGHT": 1,
            "DOWN": 2,
            "LEFT": 3,
            "WAIT": 4,
            "BOMB": 5
            }


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, len(actions))
        )
        self.critic = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, features):
        return Categorical(logits=self.actor(features)), self.critic(features).reshape(-1)
