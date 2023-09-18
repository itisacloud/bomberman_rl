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


class PPO:
    def __init__(self):
        self.rewards = []
        self.gamma = 0.95
        self.lamda = 0.95
        self.worker_steps = 4096
        self.n_mini_batch = 4
        self.epochs = 30
        self.save_directory = "./mario_ppo"
        self.batch_size = self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch

        self.policy = PolicyNetwork.to(device)
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': 0.00025},
            {'params': self.policy.critic.parameters(), 'lr': 0.001}
        ], eps=1e-4)
        self.policy_old = PolicyNetwork().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.all_episode_rewards = []
        self.all_mean_rewards = []
        self.episode = 0

    def save_checkpoint(self):
        filename = os.path.join(self.save_directory, 'checkpoint_{}.pth'.format(self.episode))
        torch.save(self.policy_old.state_dict(), f=filename)
        print('Checkpoint saved to \'{}\''.format(filename))

    def load_checkpoint(self, filename):
        self.policy.load_state_dict(torch.load(os.path.join(self.save_directory, filename)))
        self.policy_old.load_state_dict(torch.load(os.path.join(self.save_directory, filename)))
        print('Resuming training from checkpoint \'{}\'.'.format(filename))

    def sample(self):
        rewards = np.zeros(self.worker_steps, dtype=np.float32)
        actions = np.zeros(self.worker_steps, dtype=np.int32)
        done = np.zeros(self.worker_steps, dtype=bool)
        obs = np.zeros((self.worker_steps, 4, 84, 84), dtype=np.float32)
        log_pis = np.zeros(self.worker_steps, dtype=np.float32)
        values = np.zeros(self.worker_steps, dtype=np.float32)
        for t in range(self.worker_steps):
            with torch.no_grad():
                obs[t] = self.obs
                pi, v = self.policy_old(torch.tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0))
                values[t] = v.cpu().numpy()
                a = pi.sample()
                actions[t] = a.cpu().numpy()
                log_pis[t] = pi.log_prob(a).cpu().numpy()
            self.obs, rewards[t], done[t], _ = env.step(actions[t])
            self.obs = self.obs.__array__()
            env.render()
            self.rewards.append(rewards[t])
            if done[t]:
                self.episode += 1
                self.all_episode_rewards.append(np.sum(self.rewards))
                self.rewards = []
                env.reset()
                if self.episode % 10 == 0:
                    print('Episode: {}, average reward: {}'.format(self.episode,
                                                                   np.mean(self.all_episode_rewards[-10:])))
                    self.all_mean_rewards.append(np.mean(self.all_episode_rewards[-10:]))
                    plt.plot(self.all_mean_rewards)
                    plt.savefig("{}/mean_reward_{}.png".format(self.save_directory, self.episode))
                    plt.clf()
                    self.save_checkpoint()
        returns, advantages = self.calculate_advantages(done, rewards, values)
        return {
            'obs': torch.tensor(obs.reshape(obs.shape[0], *obs.shape[1:]), dtype=torch.float32, device=device),
            'actions': torch.tensor(actions, device=device),
            'values': torch.tensor(values, device=device),
            'log_pis': torch.tensor(log_pis, device=device),
            'advantages': torch.tensor(advantages, device=device, dtype=torch.float32),
            'returns': torch.tensor(returns, device=device, dtype=torch.float32)
        }

    def calculate_advantages(self, done, rewards, values):
        _, last_value = self.policy_old(torch.tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0))
        last_value = last_value.cpu().data.numpy()
        values = np.append(values, last_value)
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            mask = 1.0 - done[i]
            delta = rewards[i] + self.gamma * values[i + 1] * mask - values[i]
            gae = delta + self.gamma * self.lamda * mask * gae
            returns.insert(0, gae + values[i])
        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

    def train(self, samples, clip_range):
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
