
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import optim, nn
from torch.cuda import device

from .PolicyNetwork import PolicyNetwork
from .State import State
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
        # Define a policy network

        self.agent_name = AGENT_CONFIG["agent_name"]
        self.config_name = AGENT_CONFIG["config_name"]


        self.training = training

        self.state_dim = AGENT_CONFIG["state_dim"]
        self.action_dim = AGENT_CONFIG["action_dim"]
        self.save_every = AGENT_CONFIG["save_every"]
        self.curr_step = 0
        self.since_last_save = 0  # fall back for the save_every

        # Hyperparameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.num_devices = torch.cuda.device_count()
        else:
            self.num_devices = 1


        # setting up the network
        self.net = BomberNet(input_dim=self.state_dim, output_dim=self.action_dim).float()
        self.net = self.net.to(self.device)

        self.exploration_method = AGENT_CONFIG["exploration_method"]
        self.gamma = AGENT_CONFIG["gamma"]
        self.lamb = AGENT_CONFIG["lambda"]
        self.worker_steps = AGENT_CONFIG["worker_steps"]
        self.n_mini_batch = AGENT_CONFIG["n_mini_batch"]
        self.epochs = AGENT_CONFIG["epochs"]
        self.lr = AGENT_CONFIG["learning_rate"]

        self.rewards = []
        self.save_directory = "./models"
        self.batch_size = self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch

        # TODO: implement
        self.policy = PolicyNetwork().to(device)
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr},
            {'params': self.policy.critic.parameters(), 'lr': self.lr} # do we need different lrs?
        ], eps=1e-4)
        self.policy_old = PolicyNetwork().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.all_episode_rewards = []
        self.all_mean_rewards = []
        self.episode = 0


        self.REWARD_CONFIG = REWARD_CONFIG

        if AGENT_CONFIG["load"] == True:  # load model :D
            self.load(AGENT_CONFIG["load_path"])


    def save_checkpoint(self):
        filename = os.path.join(self.save_directory, 'checkpoint_{}.pth'.format(self.episode))
        torch.save(self.policy_old.state_dict(), f=filename)
        print('Checkpoint saved to \'{}\''.format(filename))

    def load_checkpoint(self, filename):
        self.policy.load_state_dict(torch.load(os.path.join(self.save_directory, filename)))
        self.policy_old.load_state_dict(torch.load(os.path.join(self.save_directory, filename)))
        print('Resuming training from checkpoint \'{}\'.'.format(filename))


    def calculate_loss(self, samples, clip_range):
        sampled_returns = samples['returns']
        sampled_advantages = samples['advantages']
        pi, value = self.policy(samples['obs'])
        ratio = torch.exp(pi.log_prob(samples['actions']) - samples['log_pis'])
        clipped_ratio = ratio.clamp(min=1.0 - clip_range, max=1.0 + clip_range)
        policy_reward = torch.min(ratio * sampled_advantages, clipped_ratio * sampled_advantages)
        entropy_bonus = pi.entropy()
        vf_loss = self.mse_loss(value, sampled_returns)
        loss = -policy_reward + 0.5 * vf_loss - 0.01 * entropy_bonus
        return loss.mean()

    def train(self, memory,clip_range):
        samples = memory.get_samples()
        indexes = torch.randperm(self.batch_size)
        for start in range(0, self.batch_size, self.mini_batch_size):
            end = start + self.mini_batch_size
            mini_batch_indexes = indexes[start: end]
            mini_batch = {}
            for k, v in samples.items():
                mini_batch[k] = v[mini_batch_indexes]
            for _ in range(self.epochs):
                loss = self.calculate_loss(clip_range=clip_range, samples=mini_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.policy_old.load_state_dict(self.policy.state_dict())
