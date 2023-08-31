from collections import namedtuple, defaultdict
from typing import List

import numpy as np

from .src.State import State
from .src.cache import Memory

import matplotlib.pyplot as plt

EVENTS = ['WAITED', 'MOVED_UP', 'MOVED_DOWN', 'MOVED_LEFT', 'MOVED_RIGHT', 'INVALID_ACTION',
          'CRATE_DESTROYED', 'COIN_COLLECTED', 'KILLED_SELF', 'KILLED_OPPONENT', 'OPPONENT_ELIMINATED',
          'BOMB_DROPPED', 'COIN_FOUND', 'SURVIVED_ROUND']

move_events = ['MOVED_UP', 'MOVED_DOWN', 'MOVED_LEFT', 'MOVED_RIGHT']
actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
moves = [[1, 0], [-1, 0], [0, 1], [0, -1]]


class plot:
    def __init__(self, loss_update_interval=1000, max_steps_to_plot=10000):
        self.loss_history = []
        self.steps = []
        self.loss_update_interval = loss_update_interval
        self.max_steps_to_plot = max_steps_to_plot  # Maximum number of steps to plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.loss_plot, = self.ax.plot([], [], label='Loss')
        self.ax.set_xlabel('Training Steps')
        self.ax.set_ylabel('Loss')
        self.ax.legend()

    def append(self, loss):
        self.loss_history.append(loss)
        self.steps.append(len(self.loss_history))

    def update(self):
        if self.steps[-1] % self.loss_update_interval == 0:
            if len(self.steps) > self.max_steps_to_plot:
                start_index = len(self.steps) - self.max_steps_to_plot
            else:
                start_index = 0

            self.loss_plot.set_data(self.steps[start_index:], self.loss_history[start_index:])
            self.ax.relim()  # Recalculate limits
            self.ax.autoscale_view(True, True, True)  # Autoscale the view
            plt.pause(0.1)


def setup_training(self):
    self.reward_handler = RewardHandler(self.REWARD_CONFIG)
    self.memory = Memory(input_dim=self.AGENT_CONFIG["state_dim"], size=self.AGENT_CONFIG["memory_size"])
    self.past_rewards = []
    self.past_events = defaultdict(list)
    self.past_events_count = defaultdict(int)
    self.past_movements = defaultdict(int)

    self.loss_history = []  # To keep track of loss during training
    self.loss_update_interval = 10  # Update the plot every 10 steps
    if self.draw_plot:
        self.plot = plot()


def game_events_occurred(self, old_game_state: dict, own_action: str, new_game_state: dict, events: List[str]):
    # perform training here
    new_features = self.state_processor.getFeatures(new_game_state)
    old_features = self.state_processor.getFeatures(old_game_state)

    own_action = int(actions.index(own_action))
    reward = self.reward_handler.reward_from_state(new_game_state, old_game_state, new_features, old_features, events)

    done = False

    self.memory.cache(old_features, new_features, own_action, reward, done)
    td_estimate, loss = self.agent.learn()

    for event in events:
        self.past_events_count[event] += 1

    self.past_movements[own_action] += 1

    if self.draw_plot:
        self.plot.append(loss)
        self.plot.update()


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    features = self.state_processor.getFeatures(last_game_state)
    own_action = int(actions.index(last_action))

    reward = self.reward_handler.reward_from_state(last_game_state, last_game_state, features, features, events, )

    done = True
    self.memory.cache(features, features, own_action, reward, done)
    td_estimate, loss = self.agent.learn()

    for event in events:
        self.past_events_count[event] += 1

    self.past_movements[own_action] += 1

    self.agent.save()

    self.reward_handler.new_round()

    # todo save model


class RewardHandler:
    def __init__(self, REWARD_CONFIG: str):
        self.state_processor = State(window_size=1)  # maybe move the distance function to utils or something
        self.REWARD_CONFIG = REWARD_CONFIG
        self.previous_positions = None

    def new_round(self):
        self.previous_positions = None

    def reward_from_state(self, new_game_state, old_game_state, new_features, old_features, events) -> int:
        if self.previous_positions is None:
            self.previous_positions = np.zeros_like(new_game_state["field"])

        own_position = old_game_state["self"][3]

        own_move = np.array(new_game_state["self"][3]) - np.array(old_game_state["self"][3])

        enemy_positions = [enemy[3] for enemy in old_game_state["others"]]

        reward = 0

        for event in events:
            try:
                reward += self.REWARD_CONFIG[event]
            except:
                print(f"No reward defined for event {event}")

        try:

            if "BOMB_DROPPED" in events and min(
                    [self.state_processor.distance(own_position, enemy) for enemy in enemy_positions]) < 3:
                reward += self.REWARD_CONFIG["BOMB_NEAR_ENEMY"]
                if min([self.state_processor.distance(own_position, enemy) for enemy in enemy_positions]) < 1:
                    reward += self.REWARD_CONFIG["BOMB_NEAR_ENEMY"] * 2
        except:
            pass
        center = [int(old_features.shape[1] - 1) / 2, int(old_features.shape[2] - 1) / 2]

        if sum(own_move) != 0:
            if max([old_features[5][int(center[0] + pos[0])][int(center[1] + pos[1])] for pos in moves]) == \
                    old_features[5][int(center[0] + own_move[0]), int(center[1] + own_move[1])]:
                reward += self.REWARD_CONFIG["MOVED_TOWARDS_COIN_CLUSTER"]

        self.previous_positions[own_position[0], own_position[1]] += 1

        if self.previous_positions[own_position[0], own_position[1]] > 1:
            reward += self.REWARD_CONFIG["ALREADY_VISITED"] * self.previous_positions[
                own_position[0], own_position[1]]  # scale this shit

        return reward
