import os

import numpy as np
import torch
import yaml
from .src.BomberNet import Agent
from agent_code.GlasHoch_Rangers.src.State import State

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    np.random.seed(42)

    self.logger.debug('Successfully entered setup code')

    with open(os.environ.get("AGENT_CONF", "./configs/default.yaml"), "r") as ymlfile:
        configs = yaml.safe_load(ymlfile)

    self.AGENT_CONFIG = configs["AGENT_CONFIG"]
    self.REWARD_CONFIG = configs["REWARD_CONFIG"]


    self.agent = Agent(self.AGENT_CONFIG, self.REWARD_CONFIG, training=self.train)
    self.state_processor = State(window_size=int((self.AGENT_CONFIG["state_dim"][1] - 1) / 2))


def act(self, game_state: dict) -> str:
    features = self.state_processor.getFeatures(game_state)
    self.agent.act(features)
    return actions[self.agent.act(features)]
