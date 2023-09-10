import logging

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

