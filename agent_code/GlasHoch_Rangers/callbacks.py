from agent_code.GlasHoch_Rangers.src.State import State


def setup(self):
    np.random.seed()

    self.logger.debug('Successfully entered setup code')

    AGENT_CONFIG, _, N = get_parameters()

    self.agent = OwnAgent(**AGENT_CONFIG, training=self.train)
    self.state_processor = State(N)

    if self.train:
        self.rule_based_agent = RuleBasedAgent()

def act():
