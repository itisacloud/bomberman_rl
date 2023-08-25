import torch
from torch import nn, optim
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class DQNetwork(nn.Module):
    '''mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    '''

    def __init__(self, input_dim, extra_dim, action_dim, learnin_rate, network_arch=None, is_training=False):
        super().__init__()
        c, h, w = input_dim

        self.network_arch = network_arch
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

        self.optimizer = optim.RMSprop(self.parameters(), lr=learnin_rate)
        self.optimizer = optim.RMSprop(self.parameters(), lr=learnin_rate)
        self.loss = nn.MSELoss()


    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)