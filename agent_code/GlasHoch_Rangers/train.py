from collections import namedtuple, deque, defaultdict

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

EVENTS = ['steps', 'WAITED', 'MOVED_UP', 'MOVED_DOWN', 'MOVED_LEFT', 'MOVED_RIGHT', 'INVALID_ACTION',
            'CRATE_DESTROYED', 'COIN_COLLECTED', 'KILLED_SELF', 'KILLED_OPPONENT', 'OPPONENT_ELIMINATED',
            'BOMB_DROPPED', 'COIN_FOUND', 'SURVIVED_ROUND']


def setup_training(self):
    self.reward_handler = RewardHandler()
    self.past_rewards = []
    self.past_events = defaultdict(list)
    self.past_events_count = defaultdict(int)
    self.past_movements = defaultdict(str)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    #perform training here
    new_featues = state_to_features(new_game_state)
    old_features = state_to_features(old_game_state)

    reward = self.reward_handler.reward_from_state(new_game_state,old_game_state,events,self.past_rewards)

    self.agent.learn(new_featues,old_features,self_action,reward)

    self.passed_events(events)
    for event in events:
        self.events_count[event] += 1
    self.past_movements[self_action] += 1

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.agent.save_model()
    #todo save model

class RewardHandler():
    def __int__(self,configReward:str):
        self.configReward = configReward
    def reward_from_state(self, new_game_state,old_game_state,events,rewards) -> int:
        reward = 0
        #to






