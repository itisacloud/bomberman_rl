from agent_code.GlasHoch_Rangers.src.BomberNet import DQNetwork
from agent_code.GlasHoch_Rangers.src.State import State


def setup(self):
    np.random.seed()

    self.logger.debug('Successfully entered setup code')

    AGENT_CONFIG, _, N = get_parameters()

    self.agent = Agent(**AGENT_CONFIG, training=self.train)
    self.state_processor = State(N)

    if self.train:
        self.rule_based_agent = RuleBasedAgent()

def act():
    self.agent.act()



class Agent():
    def __init__(self, input_dim, extra_dim, action_dim, learnin_rate, network_arch=None, is_training=False):
        super().__init__()
        c, h, w = input_dim

        self.network_arch = network_arch
        self.extra_dim = extra_dim
        self.action_dim = action_dim

        self.q_evaluation
        self.q_target


    def act(self):
        pass

    def train(self):
        pass

    def update_target_network(self):
        pass

    def save(self):
        pass

    def load(self):
        pass




