
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple, defaultdict
from typing import List, DefaultDict

class plot:
    def __init__(self, plot_update_interval=1000, max_steps_to_plot=10, running_mean_window=100,mode_plot = "static"):
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
        self.running_mean_window_games = running_mean_window
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
            self.loss_plot.set_data(range(len(self.loss_history)), self.loss_history)
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)

            if len(self.loss_history) >= self.running_mean_window_games*10:
                running_mean_loss = np.convolve(self.loss_history,
                                                np.ones(self.running_mean_window_games*10) / self.running_mean_window_games*10,
                                                mode='valid')
                self.ax.plot(range(self.running_mean_window_games*10, len(running_mean_loss) + self.running_mean_window_games*10),
                             running_mean_loss, label='Running Mean Loss', color='red')

            self.steps_per_game = [game - self.games[i - 1] for i, game in enumerate(self.games) if i > 0]
            self.steps_per_game_plot.set_data(range(len(self.steps_per_game)), self.steps_per_game)
            self.ax_1.relim()
            self.ax_1.autoscale_view(True, True, True)

            if len(self.steps_per_game) >= self.running_mean_window_games:
                running_mean_steps = np.convolve(self.steps_per_game,
                                                 np.ones(self.running_mean_window_games) / self.running_mean_window_games,
                                                 mode='valid')
                self.ax_1.plot(
                    range(self.running_mean_window_games, len(running_mean_steps) + self.running_mean_window_games),
                    running_mean_steps, label='Running Mean Steps per Game', color='red')

            self.exploration_rate_plot.set_data(self.steps, self.exploration_rate_history)
            self.ax_4.relim()
            self.ax_4.autoscale_view(True, True, True)

            rewards_per_game = [sum(self.rewards[self.games[i-1]:game]) for i, game in enumerate(self.games) if i > 0]
            self.total_reward_plot.set_data(range(len(rewards_per_game)),rewards_per_game)
            if len(rewards_per_game) >= self.running_mean_window_games:
                running_mean_reward = np.convolve(rewards_per_game,
                                                  np.ones(self.running_mean_window_games) / self.running_mean_window_games,
                                                  mode='valid')
                print(len(range(self.running_mean_window_games, len(running_mean_reward) + self.running_mean_window_games)))
                print(len(running_mean_reward))
                self.ax_2.plot(range(self.running_mean_window_games, len(running_mean_reward) + self.running_mean_window_games),
                               running_mean_reward,
                               label='Running Mean Total Reward', color='red')
            self.ax_2.relim()
            self.ax_2.autoscale_view(True,True,True)

            self.exploration_rate_plot.set_data(self.steps, self.exploration_rate_history)
            self.ax_4.relim()  # Recalculate limits
            self.ax_4.autoscale_view(True, True, True)
            plt.pause(0.1)

    def save(self,name):
        plt.savefig(f"./plots/{len(self.games)}_{name}.png")