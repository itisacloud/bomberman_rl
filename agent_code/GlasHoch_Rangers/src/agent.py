
import os
from pathlib import Path

import numpy as np
import torch


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
        if self.device == "cuda":
            self.num_devices = torch.cuda.device_count()
        else:
            self.num_devices = 1

        self.save_dir = "./models"

        # setting up the network
        self.net = BomberNet(input_dim=self.state_dim, output_dim=self.action_dim).float()
        self.net = self.net.to(self.device)

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
            self.lr_scheduling = True
            self.lr_scheduler_step = AGENT_CONFIG["lr_scheduler_step"]
            self.lr_scheduler_gamma = AGENT_CONFIG["lr_scheduler_gamma"]
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_scheduler_step,
                                                                gamma=self.lr_scheduler_gamma)
            self.lr_scheduler_min = AGENT_CONFIG["lr_scheduler_min"]

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
        elif self.training:
            raise ValueError("exploration_method must be epsilon-greedy or boltzmann")
        else:
            action_values = self.net(features, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()


        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.imitation_learning_rate *= self.imitation_learning_decay
        self.imitation_learning_rate = max(self.imitation_learning_rate,self.imitation_learning_min)
        if self.lr_scheduling:
            self.lr_scheduler.step()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], self.lr_scheduler_min)

        self.curr_step += 1
        return action_idx

    def save(self):
        if self.save_dir is None:
            print("Cannot save model. No save directory given.")
            return
        if self.curr_step % self.save_every != 0:
            return
        print(self.__str__())
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

    def __str__(self):
        config_str = f"Agent Configuration:\n"
        config_str += f"Agent Name: {self.agent_name}\n"
        config_str += f"Config Name: {self.config_name}\n"
        config_str += f"Exploration Rate Min: {self.exploration_rate_min}\n"
        config_str += f"Exploration Rate Decay: {self.exploration_rate_decay}\n"
        config_str += f"Exploration Rate: {self.exploration_rate}\n"
        config_str += f"Imitation Learning: {self.imitation_learning}\n"
        config_str += f"Imitation Learning Rate: {self.imitation_learning_rate}\n"
        config_str += f"Imitation Learning Decay: {self.imitation_learning_decay}\n"
        config_str += f"Imitation Learning Min: {self.imitation_learning_min}\n"
        config_str += f"Imitation Learning Expert: {self.imitation_learning_expert}\n"
        config_str += f"State Dimension: {self.state_dim}\n"
        config_str += f"Action Dimension: {self.action_dim}\n"
        config_str += f"Batch Size: {self.batch_size}\n"
        config_str += f"Save Every: {self.save_every}\n"
        config_str += f"Current Step: {self.curr_step}\n"
        config_str += f"Device: {self.device}\n"
        config_str += f"Number of Devices: {self.num_devices}\n"
        config_str += f"Save Directory: {self.save_dir}\n"
        config_str += f"Burnin: {self.burnin}\n"
        config_str += f"Learn Every: {self.learn_every}\n"
        config_str += f"Sync Every: {self.sync_every}\n"
        config_str += f"Exploration Method: {self.exploration_method}\n"
        config_str += f"Gamma: {self.gamma}\n"
        config_str += f"Loss Function: {self.loss_fn}\n"
        config_str += f"Optimizer: {self.optimizer}\n"
        config_str += f"LR Scheduler: {self.lr_scheduling}\n"
        config_str += f"LR Scheduler Step: {self.lr_scheduler_step}\n"
        config_str += f"LR Scheduler Gamma: {self.lr_scheduler_gamma}\n"
        config_str += f"LR Scheduler Min: {self.lr_scheduler_min}\n"
        config_str += f"Reward Configuration: {self.REWARD_CONFIG}\n"

        return config_str