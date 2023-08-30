from collections import namedtuple, defaultdict
from typing import List

from .src.cache import Memory

EVENTS = ['WAITED', 'MOVED_UP', 'MOVED_DOWN', 'MOVED_LEFT', 'MOVED_RIGHT', 'INVALID_ACTION',
          'CRATE_DESTROYED', 'COIN_COLLECTED', 'KILLED_SELF', 'KILLED_OPPONENT', 'OPPONENT_ELIMINATED',
          'BOMB_DROPPED', 'COIN_FOUND', 'SURVIVED_ROUND']

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
moves = [[1, 0], [-1, 0], [0, 1], [0, -1]]


def setup_training(self):
    self.reward_handler = RewardHandler(self.REWARD_CONFIG)
    self.memory = Memory(input_dim=self.AGENT_CONFIG["state_dim"],size=self.AGENT_CONFIG["memory_size"])
    self.past_rewards = []
    self.past_events = defaultdict(list)
    self.past_events_count = defaultdict(int)
    self.past_movements = defaultdict(int)


def game_events_occurred(self, old_game_state: dict, own_action: str, new_game_state: dict, events: List[str]):
    # perform training here
    new_features = self.state_processor.getFeatures(new_game_state)
    old_features = self.state_processor.getFeatures(old_game_state)

    own_action = int(actions.index(own_action))
    reward = self.reward_handler.reward_from_state(new_game_state, old_game_state, new_features, old_features, events )

    done = False
    self.memory.cache(old_features, new_features, own_action,reward,done)
    self.agent.learn()


    for event in events:
        self.past_events_count[event] += 1

    self.past_movements[own_action] += 1


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    features = self.state_processor.getFeatures(last_game_state)
    own_action = int(actions.index(last_action))

    reward = self.reward_handler.reward_from_state(last_game_state,last_game_state, features, features, events, )

    done = True
    self.memory.cache(features, features, own_action, reward, done)
    self.agent.learn()

    for event in events:
        self.past_events_count[event] += 1

    self.past_movements[own_action] += 1

    self.agent.save()


    # todo save model


class RewardHandler:
    def __init__(self, REWARD_CONFIG: str):
        self.REWARD_CONFIG = REWARD_CONFIG

    def reward_from_state(self, new_game_state,old_game_state, new_features, old_features, events) -> int:
        own_position = old_game_state["self"][3]
        enemy_positions = [enemy[3] for enemy in old_game_state["others"]]

        reward = 0
        for event in events:
            reward += self.REWARD_CONFIG[event]


        """
        if "BOMB_DROPPED" in events and min(
                [self.state_processor.distance(own_position, enemy) for enemy in enemy_positions]) < 3:
            reward += self.bomb_reward(new_features, old_features)"""
        return reward
