import os

import numpy as np
import torch
import yaml
from BomberNet import Agent

from agent_code.GlasHoch_Rangers.src.State import State

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    np.random.seed(42)

    self.logger.debug('Successfully entered setup code')

    with open(os.environ.get("AGENT_CONF", "agent_code/GlasHoch_Rangers/configs/default.yml"), "r") as ymlfile:
        AGENT_CONFIG = yaml.safe_load(ymlfile)

    self.agent = Agent(**AGENT_CONFIG, training=self.train)
    self.state_processor = State(AGENT_CONFIG["state_dim"], AGENT_CONFIG["extra_dim"], AGENT_CONFIG["action_dim"], )


def act(self, game_state: dict) -> str:
    if np.random.rand() < self.exploration_rate:
        action_idx = np.random.randint(self.action_dim)
        # EXPLOIT
    else:
        features = self.state_processor.process(game_state)
        action_values = self.net(features, model="online")
        action_idx = torch.argmax(action_values, axis=1).item()
    # decrease exploration_rate
    self.exploration_rate *= self.exploration_rate_decay
    self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

    # increment step
    self.curr_step += 1

    return actions[action_idx]
