import copy
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DataParallel  # Import DataParallel

from .cache import cache

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT',"BOMB"]

reversed = {"UP": 0,
            "RIGHT": 1,
            "DOWN": 2,
            "LEFT": 3,
            "WAIT": 4,
            "BOMB": 5
            }

class bomberNet(nn.Module):
    def __init__(self, input_dim, output_dim, precision=torch.float32):
        super().__init__()

        c, h, w = input_dim

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * (h // 2) * (w // 2), 512),  # Adjusted input size for the linear layer
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

        self.online = DataParallel(self.online)  # Wrap the online network in DataParallel
        self.target = DataParallel(self.target)  # Wrap the target network in DataParallel

        self.loss_fn = nn.MSELoss() #default loss
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=0.00001) #default optimizer

    def forward(self, input, model):
        input = input.unsqueeze(0)
        if len(input.shape) == 5:
            input = input.squeeze(0) # this feels wrong but it works
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        else:
            raise Exception("Invalid model name")


