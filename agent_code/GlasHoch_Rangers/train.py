from collections import namedtuple, defaultdict
from typing import List, DefaultDict

import numpy as np

from .src.State import State
from .src.cache import Memory

import matplotlib.pyplot as plt

EVENTS = ['WAITED', 'MOVED_UP', 'MOVED_DOWN', 'MOVED_LEFT', 'MOVED_RIGHT', 'INVALID_ACTION',
          'CRATE_DESTROYED', 'COIN_COLLECTED', 'KILLED_SELF', 'KILLED_OPPONENT', 'OPPONENT_ELIMINATED',
          'BOMB_DROPPED', 'COIN_FOUND', 'SURVIVED_ROUND']

move_events = ['MOVED_UP', 'MOVED_DOWN', 'MOVED_LEFT', 'MOVED_RIGHT']
actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT',  'BOMB']
moves = [[1, 0], [-1, 0], [0, 1], [0, -1]]


class plot:
    def __init__(self, plot_update_interval=1000, max_steps_to_plot=10000, running_mean_window=100, mode_plot = "static"):
        self.loss_history = []
        self.total_rewards = []
        self.event_history = []
        self.games = [0]
        self.steps = []
        self.loss_mask = []
        self.reward_running_mean = []
        self.exploration_rate_history = []
        self.rewards = []
        self.plot_update_interval = plot_update_interval
        self.max_steps_to_plot = max_steps_to_plot
        self.running_mean_window = running_mean_window
        self.save_plot_rate = plot_update_interval
        self.mode_plot = mode_plot

        # Create a figure with subplots
        self.fig, self.axs = plt.subplots(5, figsize=(10, 15))
        self.ax = self.axs[0]
        self.ax_1 = self.axs[1]
        self.ax_2 = self.axs[2]
        self.ax_3 = self.axs[3]
        self.ax_4 = self.axs[4]

        self.loss_plot, = self.ax.plot([], [], label='Loss', color='blue')
        self.ax.set_xlabel('Steps')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training Loss')
        self.ax.legend()

        self.steps_per_game_plot, = self.ax_1.plot([], [], label='Steps per game', color='green')
        self.ax_1.set_xlabel('Games')
        self.ax_1.set_ylabel('Steps per Game')
        self.ax_1.set_title('Steps per Game')

        self.total_reward_plot, = self.ax_2.plot([], [], label='Total Reward', color='yellow')
        self.ax_2.set_xlabel('Games')
        self.ax_2.set_ylabel('Rewards per Game')
        self.ax_2.set_title('Total Reward')

        self.event_plot = self.ax_3.bar([], [], label='Events', color='purple')
        self.ax_3.set_xlabel('Games')
        self.ax_3.set_ylabel('Event Counts')
        self.ax_3.set_title('Event Counts')

        self.exploration_rate_plot, = self.ax_4.plot([], [], label='Exploration Rate', color='orange')
        self.ax_4.set_xlabel('Steps')
        self.ax_4.set_ylabel('Exploration Rate')
        self.ax_4.set_title('Exploration Rate')
        self.ax_4.legend()

        self.fig.tight_layout()

        plt.ion()


    def append(self, loss, exploration_rate,reward):
        self.loss_mask.append(True) if loss is not None else self.loss_mask.append(False)
        if loss is not None:
            self.loss_history.append(loss)
        self.steps.append(len(self.loss_mask))
        self.exploration_rate_history.append(exploration_rate)
        self.rewards.append(reward)

    def append_game(self):
        self.games.append(self.steps[-1])


    def update(self):
        if len(self.loss_mask) % self.plot_update_interval != 0:
            return

        if len(self.loss_mask) != 0:
            start_index = max(0, len(self.loss_mask) - self.max_steps_to_plot)
            self.loss_plot.set_data(range(start_index, len(self.loss_history)), self.loss_history[start_index:])
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)

            if len(self.loss_history) >= self.running_mean_window:
                running_mean_loss = np.convolve(self.loss_history,
                                                np.ones(self.running_mean_window) / self.running_mean_window,
                                                mode='valid')
                self.ax.plot(range(start_index + self.running_mean_window - 1, len(self.loss_history)),
                                                     running_mean_loss, label='Running Mean Loss', color='red')

            self.steps_per_game = [game - self.games[i - 1] for i, game in enumerate(self.games) if i > 0]
            self.steps_per_game_plot.set_data(range(len(self.steps_per_game)), self.steps_per_game)
            self.ax_1.relim()
            self.ax_1.autoscale_view(True, True, True)

            if len(self.steps_per_game) >= self.running_mean_window:
                running_mean_steps = np.convolve(self.steps_per_game,
                                                 np.ones(self.running_mean_window) / self.running_mean_window,
                                                 mode='valid')
                self.ax_1.plot(
                    range(start_index + self.running_mean_window - 1, start_index + len(self.steps_per_game)),
                    running_mean_steps, label='Running Mean Steps per Game', color='red')

            self.exploration_rate_plot.set_data(self.steps, self.exploration_rate_history)
            self.ax_4.relim()
            self.ax_4.autoscale_view(True, True, True)

            rewards_per_game = [sum(self.rewards[self.games[i-1]:game]) for i, game in enumerate(self.games) if i > 0]
            self.total_reward_plot.set_data(range(len(rewards_per_game)),rewards_per_game)
            if len(rewards_per_game) >= self.running_mean_window:
                running_mean_reward = np.convolve(rewards_per_game,
                                                  np.ones(self.running_mean_window) / self.running_mean_window,
                                                  mode='valid')
                self.ax_2.plot(range(len(self.running_mean_window,running_mean_reward)),
                               running_mean_reward,
                               label='Running Mean Total Reward', color='red')
            self.ax_2.relim()
            self.ax_2.autoscale_view(True,True,True)

            self.exploration_rate_plot.set_data(self.steps, self.exploration_rate_history)
            self.ax_4.relim()  # Recalculate limits
            self.ax_4.autoscale_view(True, True, True)
            plt.pause(0.1)

    def save(self,name):
        if self.steps[-1] % self.save_plot_rate == 0 and self.mode_plot == "static":
            plt.savefig(f"./plots/{len(self.games)}_{name}.png")
def setup_training(self):
    self.reward_handler = RewardHandler(self.REWARD_CONFIG)
    self.memory = Memory(input_dim=self.AGENT_CONFIG["state_dim"], size=self.AGENT_CONFIG["memory_size"])
    self.past_rewards = []
    self.past_events = defaultdict(list)
    self.past_events_count = defaultdict(int)
    self.past_movements = defaultdict(int)

    self.loss_history = []
    self.plot_update_interval = 10
    if self.draw_plot:
        self.plot = plot(plot_update_interval=self.AGENT_CONFIG["draw_plot_every"],mode_plot=self.mode_plot)


def game_events_occurred(self, old_game_state: dict, own_action: str, new_game_state: dict, events: List[str]):
    # perform training here
    new_features = self.state_processor.getFeatures(new_game_state)
    old_features = self.state_processor.getFeatures(old_game_state)
    if self.agent.imitation_learning:
        expert_action = self.agent.imitation_learning_expert.act(old_game_state) == own_action
    else:
        expert_action = False
    own_action = int(actions.index(own_action))
    reward = self.reward_handler.reward_from_state(new_game_state, old_game_state, new_features, old_features, events,expert_action)
    done = False
    self.memory.cache(old_features, new_features, own_action, reward, done)
    td_estimate, loss = self.agent.learn(self.memory)
    exploration_rate = self.agent.exploration_rate
    for event in events:
        self.past_events_count[event] += 1

    self.past_movements[own_action] += 1

    if self.draw_plot:
        self.plot.append(loss, exploration_rate,reward)
        self.plot.update()
        self.plot.save(self.agent.agent_name)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.total_reward = 0
    features = self.state_processor.getFeatures(last_game_state)
    own_action = int(actions.index(last_action))

    reward = self.reward_handler.reward_from_state(last_game_state, last_game_state, features, features, events, )
    self.total_reward += reward

    done = True
    self.memory.cache(features, features, own_action, reward, done)
    td_estimate, loss = self.agent.learn(self.memory)

    for event in events:
        self.past_events_count[event] += 1

    self.past_movements[own_action] += 1

    self.agent.save()

    if self.draw_plot:

        self.plot.append_game()
        self.plot.save(self.agent.agent_name)

    self.reward_handler.new_round()





class RewardHandler:
    def __init__(self, REWARD_CONFIG: str):
        self.state_processor = State(window_size=1)  # maybe move the distance function to utils or something
        self.REWARD_CONFIG = REWARD_CONFIG
        self.previous_positions = defaultdict(int)
        self.moves = [np.array([0,0])]
        self.rewards = []
        self.movement_based_rewards = []

    def new_round(self):
        self.previous_positions = defaultdict(int)
        self.moves = [np.array([0,0])]

    def reward_from_state(self, new_game_state, old_game_state, new_features, old_features, events,expert_action=False) -> int:
        own_position = old_game_state["self"][3]
        own_move = np.array(new_game_state["self"][3]) - np.array(old_game_state["self"][3])

        enemy_positions = [enemy[3] for enemy in old_game_state["others"]]

        if np.all(self.moves[-1] + own_move == np.array([0, 0])) and not np.all(own_move == np.array([0, 0])):
            if self.movement_based_rewards[-1] > 0:
                reward = -self.movement_based_rewards[-1] #only undo positive rewards
            else:
                reward = 0
        else:
            reward = 0

        movement_reward = 0

        if expert_action:
            reward += self.REWARD_CONFIG["EXPERT_ACTION"]

        if not np.all(own_move == np.array([0, 0])):
            self.moves.append(own_move) #only append movements

        for event in events:
            try:
                reward += self.REWARD_CONFIG[event]
                if event in ["MOVED_UP", "MOVED_DOWN", "MOVED_LEFT", "MOVED_RIGHT"]:
                    movement_reward += self.REWARD_CONFIG[event]
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

        center = np.array([int(old_features.shape[1] - 1) / 2, int(old_features.shape[2] - 1) / 2], dtype=int)

        if sum(own_move) != 0:
            if max([old_features[5][int(center[0] + pos[0])][int(center[1] + pos[1])] for pos in moves]) == \
                    old_features[5][int(center[0] + own_move[0]), int(center[1] + own_move[1])]:
                reward += self.REWARD_CONFIG["MOVED_TOWARDS_COIN_CLUSTER"]
                movement_reward += self.REWARD_CONFIG["MOVED_TOWARDS_COIN_CLUSTER"]
            if max([old_features[6][int(center[0] + pos[0])][int(center[1] + pos[1])] for pos in moves]) == \
                    old_features[6][int(center[0] + own_move[0]), int(center[1] + own_move[1])]:
                reward += self.REWARD_CONFIG["MOVED_TOWARDS_ENEMY"]
                movement_reward += self.REWARD_CONFIG["MOVED_TOWARDS_ENEMY"]

        self.previous_positions[own_position[0], own_position[1]] += 1

        if self.previous_positions[own_position[0], own_position[1]] > 1:
            reward += self.REWARD_CONFIG["ALREADY_VISITED"] * self.previous_positions[
                own_position[0], own_position[1]]  # push to explore new areas, avoid local maximas

        if new_features[1][center[0],center[1]] == 0 and old_features[1][center[0],center[1]]> 0:
            reward += self.REWARD_CONFIG["MOVED_OUT_OF_DANGER"]
            movement_reward += self.REWARD_CONFIG["MOVED_OUT_OF_DANGER"]
        elif new_features[1][center[0],center[1]] <  old_features[1][center[0],center[1]]:
            reward += self.REWARD_CONFIG["MOVED_OUT_OF_DANGER"]*0.5
            movement_reward += self.REWARD_CONFIG["MOVED_OUT_OF_DANGER"]*0.5

        self.rewards.append(reward)

        if not np.all(own_move == np.array([0, 0])): # only append rewards from valid movements
            self.movement_based_rewards.append(movement_reward)
        return reward
