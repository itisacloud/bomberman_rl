import os

import numpy as np
import torch
import yaml
from src.agent import Agent
from agent_code.GlasHoch_Rangers.src.State import State

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', "BOMB"]


def setup(self):
    """
       Setup your code. This is called once when loading each agent.
       Make sure that you prepare everything such that act(...) can be called.

       When in training mode, the separate `setup_training` in train.py is called
       after this method. This separation allows you to share your trained agent
       with other students, without revealing your training code.

       In this example, our model is a set of probabilities over actions
       that are is independent of the game state.

       :param self: This object is passed to all callbacks and you can set arbitrary values.
   """
    np.random.seed(42)

    self.logger.debug('Successfully entered setup code')

    with open(os.environ.get("AGENT_CONF", "./configs/ppo.yaml"), "r") as ymlfile:
        configs = yaml.safe_load(ymlfile)
    self.AGENT_CONFIG = configs["AGENT_CONFIG"]
    self.REWARD_CONFIG = configs["REWARD_CONFIG"]
    self.agent = Agent(self.AGENT_CONFIG, self.REWARD_CONFIG, training=self.train)
    print(self.agent)
    self.state_processor = State(window_size=int((self.AGENT_CONFIG["state_dim"][1] - 1) / 2))
    self.draw_plot = self.AGENT_CONFIG["draw_plot"]
    if self.draw_plot:
        self.mode_plot = self.AGENT_CONFIG["mode_plot"]


def act(self, game_state: dict) -> str:
    """
        Your agent should parse the input, think, and take a decision.
        When not in training mode, the maximum execution time for this method is 0.5s.

        :param self: The same object that is passed to all of your callbacks.
        :param game_state: The dictionary that describes everything on the board.
        :return: The action to take as a string.
    """
    features = self.state_processor.getFeatures(game_state)
    self.last_features = features
    action_idx = self.agent.act(features, game_state)
    return actions[action_idx]
