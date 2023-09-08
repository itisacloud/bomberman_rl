import copy
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .cache import Memory

from agents import Agent as Expert
from agents import SequentialAgentBackend

from ...coin_collector_agent import callbacks as coin_collector_agent
from ...rule_based_agent import callbacks as rule_based_agent

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT',"BOMB"]

reversed = {"UP": 0,
            "RIGHT": 1,
            "DOWN": 2,
            "LEFT": 3,
            "WAIT": 4,
            "BOMB": 5
            }

class BomberNet(nn.Module):
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

    def forward(self, input, model):
        input = input.unsqueeze(0)
        if len(input.shape) == 5:
            input = input.squeeze(0) # this feels wrong but it works
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

class Agent():
    def __init__(self,AGENT_CONFIG,REWARD_CONFIG,training):

        self.training = training

        self.agent_name = AGENT_CONFIG["agent_name"]
        self.config_name = AGENT_CONFIG["config_name"]

        self.exploration_rate_min = AGENT_CONFIG["exploration_rate_min"]
        self.exploration_rate_decay = AGENT_CONFIG["exploration_rate_decay"]
        self.exploration_rate = AGENT_CONFIG["exploration_rate"]


        self.imitation_learning = AGENT_CONFIG["imitation_learning"] # if true, the agent will learn from an expert
        self.imitation_learning_rate = AGENT_CONFIG["imitation_learning_rate"] #
        self.imitation_learning_decay = AGENT_CONFIG["imitation_learning_decay"]
        self.imitation_learning_min = AGENT_CONFIG["imitation_learning_min"]
        self.imitation_learning_expert = AGENT_CONFIG["imitation_learning_expert"]

        if self.imitation_learning:
            self.imitation_learning_expert = Expert(self.imitation_learning_expert)

        self.state_dim = AGENT_CONFIG["state_dim"]
        self.action_dim = AGENT_CONFIG["action_dim"]
        self.batch_size = AGENT_CONFIG["batch_size"]
        self.save_every = AGENT_CONFIG["save_every"]
        self.curr_step = 0

        # Hyperparameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = "./models"

        # setting up the network
        self.net = BomberNet(input_dim=self.state_dim, output_dim=self.action_dim).float()
        self.net.to(self.device)

        self.burnin = AGENT_CONFIG["burnin"]  # min. experiences before training
        self.learn_every = AGENT_CONFIG["learn_evry"] # no. of experiences between updates to Q_online
        self.sync_every = AGENT_CONFIG["sync_evry"]  # no. of experiences between Q_target & Q_online sync
        self.exploration_method = AGENT_CONFIG["exploration_method"]
        self.gamma = AGENT_CONFIG["gamma"]

        # discount factor
        if AGENT_CONFIG["loss_fn"] == "MSE":
            self.loss_fn = torch.nn.MSELoss()
        elif AGENT_CONFIG["loss_fn"] == "SmoothL1":
            self.loss_fn = torch.nn.SmoothL1Loss()
        else:
            raise ValueError("loss_fn must be either MSE or SmoothL1")

        # optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=AGENT_CONFIG["learning_rate"])

        if AGENT_CONFIG["lr_scheduler"] == True: #implement lr scheduler later
            self.lr_scheduler_step = AGENT_CONFIG["lr_scheduler_step"]
            self.lr_scheduler_gamma = AGENT_CONFIG["lr_scheduler_gamma"]
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_scheduler_step,
                                                                gamma=self.lr_scheduler_gamma)
        self.REWARD_CONFIG = REWARD_CONFIG

        if AGENT_CONFIG["load"] == True: # load model :D
            self.load(AGENT_CONFIG["load_path"])

    def learn(self,memory):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        new_state, old_state, action, reward, done = memory.sample(self.batch_size)


        # Get TD Estimate
        td_est = self.td_estimate(old_state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, new_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    # the follwoing four functions are from the tutorial and only slightly modified, is this allowed?
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def td_estimate(self, state, action):
        indices = torch.tensor(np.arange(0, self.batch_size), dtype=torch.long)
        action_tensor = torch.tensor(action, dtype=torch.long)
        current_Q = self.net(state, model="online")[indices, action_tensor]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q_online = self.net(next_state, model="online")
        best_action_online = torch.argmax(next_state_Q_online, axis=1)

        next_state_Q_target = self.net(next_state, model="target")
        next_Q_values = next_state_Q_target[np.arange(0, self.batch_size), best_action_online]

        td_targets = (reward + (1 - done.float()) * self.gamma * next_Q_values).float()
        return td_targets

    def act(self,features,state = None):
        if self.exploration_method == "epsilon-greedy" and self.training == True:
            if np.random.rand() < self.exploration_rate:
                if np.random.rand() < self.imitation_learning_rate and self.imitation_learning:
                    try:
                        action_idx = reversed[self.imitation_learning_expert.act(state)]
                    except:
                        print("fall back")
                        action_idx = np.random.randint(self.action_dim)
                else:
                    action_idx = np.random.randint(self.action_dim)
            else:
                action_values = self.net(features, model="online")
                action_idx = torch.argmax(action_values, axis=1).item()
        elif self.exploration_method == "boltzmann" and self.training == True:
            action_values = self.net(features, model="online")
            probabilities = torch.softmax(action_values / self.temperature, dim=-1)
            action_idx = torch.multinomial(probabilities, 1).item()
        else:
            raise ValueError("exploration_method must be epsilon-greedy or boltzmann")

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.imitation_learning_rate *= self.imitation_learning_decay
        self.imitation_learning_rate = max(self.imitation_learning_rate,self.imitation_learning_min)

        self.curr_step += 1
        return action_idx

    def save(self):
        if self.save_dir is None:
            print("Cannot save model. No save directory given.")
            return
        if self.curr_step % self.save_every != 0:
            return
        save_path = (
                self.save_dir + f"/{self.agent_name}_{self.config_name}_{int(self.curr_step)}.pth"
        )
        torch.save(self.net.state_dict(), save_path)

    def load(self, model_path):
        print(model_path)
        print(os.path.abspath(model_path))

        if not Path(model_path).is_file():
            print(f"Model file not found at {model_path}. Cannot load the model.")
            return

        self.net.load_state_dict(torch.load(model_path,map_location=torch.device(self.device)))
        self.sync_Q_target()  # Sync the target network with the loaded model's parameters


class Expert:
    def __init__(self, name):
        if name == "rule_based_agent":
            from ...rule_based_agent import callbacks
        else:
            raise ("Unknown Expert defined in config yaml")

        self.logger = logging.getLogger('BombeRLeWorld')
        self.callbacks = callbacks
        self.callbacks.setup(self)
    def act(self, gamestate):
        return self.callbacks.act(self,gamestate)
