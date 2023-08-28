import os

from agent_code.GlasHoch_Rangers.src.BomberNet import DQNetwork
from agent_code.GlasHoch_Rangers.src.State import State
import yaml
from BomberNet import Agent

import numpy as np
def setup(self):
    np.random.seed(42)

    self.logger.debug('Successfully entered setup code')

    with open(os.environ.get("AGENT_CONF","agent_code/GlasHoch_Rangers/configs/default.yml"), "r") as ymlfile:
        AGENT_CONFIG = yaml.safe_load(ymlfile)

    self.agent = Agent(**AGENT_CONFIG, training=self.train)
    self.state_processor = State(AGENT_CONFIG["state_dim"], AGENT_CONFIG["extra_dim"], AGENT_CONFIG["action_dim"],)


def act(self):
    self.agent.act()





