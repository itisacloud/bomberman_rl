from collections import namedtuple, defaultdict
from typing import List, DefaultDict

import numpy as np
import torch

from src.State import State
from src.Memory import Memory
from src.plots import plot

import matplotlib.pyplot as plt

EVENTS = ['WAITED', 'MOVED_UP', 'MOVED_DOWN', 'MOVED_LEFT', 'MOVED_RIGHT', 'INVALID_ACTION',
          'CRATE_DESTROYED', 'COIN_COLLECTED', 'KILLED_SELF', 'KILLED_OPPONENT', 'OPPONENT_ELIMINATED',
          'BOMB_DROPPED', 'COIN_FOUND', 'SURVIVED_ROUND']

move_events = ['MOVED_UP', 'MOVED_DOWN', 'MOVED_LEFT', 'MOVED_RIGHT']
actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
moves = [[1, 0], [-1, 0], [0, 1], [0, -1]]


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.reward_handler = RewardHandler(self.REWARD_CONFIG)
    self.past_rewards = []
    self.batch_size = 0
    self.memory = Memory(input_dim=self.AGENT_CONFIG["state_dim"], size=self.batch_size)
    self.curr_step = 0

    """
    rewards = np.zeros(self.worker_steps, dtype=np.float32)
    actions = np.zeros(self.worker_steps, dtype=np.int32)
    done = np.zeros(self.worker_steps, dtype=bool)
    obs = np.zeros((self.worker_steps, 4, 84, 84), dtype=np.float32)
    log_pis = np.zeros(self.worker_steps, dtype=np.float32)
    values = np.zeros(self.worker_steps, dtype=np.float32)
    
    """


def game_events_occurred(self, old_game_state: dict, own_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    # perform training here
    features = self.state_processor.getFeatures(old_game_state)
    new_features = self.state_processor.getFeatures(new_game_state)
    pi, v = self.policy_old(torch.tensor(self.features, dtype=torch.float32, device=self.device).unsqueeze(0))
    value = v.cpu().numpy()
    a = pi.sample()
    action = a.cpu().numpy()
    log_pis = pi.log_prob(a).cpu().numpy()

    reward = self.reward_handler(new_game_state, old_game_state, new_features, self.features, events,
                                 expert_action=False, expert_ratio=0.0)

    self.memory.cache(value, action, log_pis, reward, new_features, False)

    if self.curr_step % self.batch_size == 0:
        self.memory.returns, self.memory.advantges = self.calculate_advantages(False, self.memory.rewards,
                                                                               self.memory.values)
        self.agent.train(self.memory)  # learning happens (backprob through the network)
        self.memory = self.memory(input_dim=self.AGENT_CONFIG["state_dim"], size=self.batch_size)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """

    self.total_reward = 0
    features = self.state_processor.getFeatures(last_game_state)
    own_action = int(actions.index(last_action))
    reward = self.reward_handler.reward_from_state(last_game_state, last_game_state, features, features, events,
                                                   expert_action=False)
    self.total_reward += reward

    done = True
    self.memory.cache(features, new_features, own_action, reward, done)
    td_estimate, loss = self.agent.learn(self.memory)

    for event in events:
        self.past_events_count[event] += 1

    self.past_movements[own_action] += 1

    self.agent.save()

    if self.draw_plot:
        self.plot.append_game()
        self.plot.save(self.agent.agent_name)

    self.reward_handler.new_round()


def calculate_advantages(self, done, rewards, values):
    _, last_value = self.policy_old(torch.tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0))
    last_value = last_value.cpu().data.numpy()
    values = np.append(values, last_value)
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        mask = 1.0 - done[i]
        delta = rewards[i] + self.gamma * values[i + 1] * mask - values[i]
        gae = delta + self.gamma * self.lamda * mask * gae
        returns.insert(0, gae + values[i])
    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-8)


class RewardHandler:
    """
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    def __init__(self, REWARD_CONFIG: str):
        self.state_processor = State(window_size=1)  # maybe move the distance function to utils or something
        self.REWARD_CONFIG = REWARD_CONFIG
        self.previous_positions = defaultdict(int)
        self.moves = [np.array([0, 0])]
        self.rewards = []
        self.movement_based_rewards = []

    def new_round(self):
        self.previous_positions = defaultdict(int)
        self.moves = [np.array([0, 0])]

    def reward_from_state(self, new_game_state, old_game_state, new_features, old_features, events, expert_action=False,
                          expert_ratio=0.0) -> int:
        own_position = old_game_state["self"][3]
        own_move = np.array(new_game_state["self"][3]) - np.array(old_game_state["self"][3])

        enemy_positions = [enemy[3] for enemy in old_game_state["others"]]

        if np.all(self.moves[-1] + own_move == np.array([0, 0])) and not np.all(own_move == np.array([0, 0])):
            if self.movement_based_rewards[-1] > 0:
                reward = -self.movement_based_rewards[-1]  # only undo positive rewards
            else:
                reward = 0
        else:
            reward = 0

        movement_reward = 0

        if expert_action:
            reward += self.REWARD_CONFIG["EXPERT_ACTION"] * expert_ratio

        if not np.all(own_move == np.array([0, 0])):
            self.moves.append(own_move)  # only append movements

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

        if new_features[1][center[0], center[1]] == 0 and old_features[1][center[0], center[1]] > 0:
            reward += self.REWARD_CONFIG["MOVED_OUT_OF_DANGER"]
            movement_reward += self.REWARD_CONFIG["MOVED_OUT_OF_DANGER"]
        elif new_features[1][center[0], center[1]] < old_features[1][center[0], center[1]]:
            reward += self.REWARD_CONFIG["MOVED_OUT_OF_DANGER"] * 0.5
            movement_reward += self.REWARD_CONFIG["MOVED_OUT_OF_DANGER"] * 0.5

        self.rewards.append(reward)

        if not np.all(own_move == np.array([0, 0])):  # only append rewards from valid movements
            self.movement_based_rewards.append(movement_reward)
        return reward
