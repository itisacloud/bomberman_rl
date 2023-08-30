from collections import namedtuple, defaultdict
from typing import List



from src.State import State
from src.State import State.distance as distance #arrrggg

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

EVENTS = ['WAITED', 'MOVED_UP', 'MOVED_DOWN', 'MOVED_LEFT', 'MOVED_RIGHT', 'INVALID_ACTION',
          'CRATE_DESTROYED', 'COIN_COLLECTED', 'KILLED_SELF', 'KILLED_OPPONENT', 'OPPONENT_ELIMINATED',
          'BOMB_DROPPED', 'COIN_FOUND', 'SURVIVED_ROUND']

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
moves = [[1, 0], [-1, 0], [0, 1], [0, -1]]


def setup_training(self):
    self.reward_handler = RewardHandler()
    self.past_rewards = []
    self.past_events = defaultdict(list)
    self.past_events_count = defaultdict(int)
    self.past_movements = defaultdict(str)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    # perform training here
    new_features = self.state_processor(new_game_state)
    old_features = self.state_processor(old_game_state)

    reward = self.reward_handler.reward_from_state(new_game_state, old_game_state, new_features, old_features, events, )

    done = False

    self.agent.learn(new_features, old_features, self_action, reward,done)

    self.passed_events(events)
    for event in events:
        self.events_count[event] += 1
    self.past_movements[self_action] += 1


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    self.agent.save_model()


    # todo save model


class RewardHandler:
    def __init__(self, REWARD_CONFIG: str):
        self.configReward = REWARD_CONFIG

    def reward_from_state(self, new_game_state, old_game_state, new_features, old_features, events) -> int:
        own_position = new_game_state["self"][3]
        enemy_positions = [enemy[3] for enemy in new_game_state["others"]]

        reward = 0
        for event in events:
            reward += self.agent.REWARD_CONFIG[event]
        if "BOMB_DROPPED" in events and min(
                [self.state_processor.distance(own_position, enemy) for enemy in enemy_positions]) < 3:
            reward += self.bomb_reward(new_features, old_features)
