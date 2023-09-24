from collections import namedtuple, defaultdict
from typing import List, DefaultDict

import numpy as np

from .src.State import State
from .src.cache import cache
from .src.plots import plot

import matplotlib.pyplot as plt

EVENTS = ['WAITED', 'MOVED_UP', 'MOVED_DOWN', 'MOVED_LEFT', 'MOVED_RIGHT', 'INVALID_ACTION',
          'CRATE_DESTROYED', 'COIN_COLLECTED', 'KILLED_SELF', 'KILLED_OPPONENT', 'OPPONENT_ELIMINATED',
          'BOMB_DROPPED', 'COIN_FOUND', 'SURVIVED_ROUND']

move_events = ['MOVED_UP', 'MOVED_DOWN', 'MOVED_LEFT', 'MOVED_RIGHT']
actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT',  'BOMB']
moves = [[1, 0], [-1, 0], [0, 1], [0, -1]]

def setup_training(self):
    self.reward_handler = RewardHandler(self.REWARD_CONFIG)
    self.memory = cache(input_dim=self.AGENT_CONFIG["features_dim"], size=self.AGENT_CONFIG["memory_size"], rotation_augment = self.AGENT_CONFIG["rotation_augment"], rotation_augment_prob = self.AGENT_CONFIG["rotation_augment_prob"])
    self.past_rewards = []
    self.past_events = []
    self.past_Qs = []
    self.past_actions = []
    self.past_events_count = defaultdict(int)
    self.past_movements = defaultdict(int)

    self.loss_history = []
    self.plot_update_interval = 10
    if self.draw_plot:
        self.plot = plot(plot_update_interval=self.AGENT_CONFIG["draw_plot_every"],mode_plot=self.mode_plot,imitation_learning=True)

def game_events_occurred(self, old_game_state: dict, own_action: str, new_game_state: dict, events: List[str]):
    # perform training here
    old_features = self.last_features
    new_features = self.state_processor.get_features(new_game_state)

    if self.agent.imitation_learning and self.REWARD_CONFIG["EXPERT_ACTION"] > 0:
        expert_action = self.agent.imitation_learning_expert.act(old_game_state) == own_action
    else:
        expert_action = False

    own_action = int(actions.index(own_action))
    reward ,events = self.reward_handler.reward_from_state(new_game_state, old_game_state, new_features, old_features, events,expert_action,self.agent.imitation_learning_rate)
    done = False
    self.past_Qs.append(self.agent.net(old_features,model="online"))
    self.memory.cache(old_features, new_features, own_action, reward, done)
    td_estimate, loss = self.agent.learn(self.memory)
    exploration_rate = self.agent.exploration_rate

    if self.draw_plot:
        self.plot.append(loss, exploration_rate,reward,self.agent.imitation_learning_rate)
        self.plot.update()
        if self.agent.curr_step % self.agent.save_every == 0:
            self.plot.save(self.agent.agent_name)
    for event in events:
        self.past_events_count[event] += 1
    self.past_events.append(events)
    self.past_rewards.append(reward)
    self.past_actions.append(own_action)
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    self.total_reward = 0
    features = self.last_features
    own_action = int(actions.index(last_action))
    reward,events = self.reward_handler.reward_from_state(last_game_state, last_game_state, features, features, events,expert_action=False)
    self.total_reward += reward
    self.past_Qs.append(self.agent.net(features,model="online"))
    self.past_actions.append(own_action)

    done = True
    self.memory.cache(features, features, own_action, reward, done)
    td_estimate, loss = self.agent.learn(self.memory)

    self.agent.save()

    if self.draw_plot:
        self.plot.append_game()
        if self.agent.curr_step % self.agent.save_every == 0:
            self.plot.save(self.agent.agent_name)

    for event in events:
        self.past_events_count[event] += 1
    self.past_events.append(events)
    self.past_rewards.append(reward)


    self.past_events_count = defaultdict(int)
    self.past_events = []
    self.past_rewards = []
    self.past_Qs = []
    self.past_actions = []
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

    def reward_from_state(self, new_game_state, old_game_state, new_features, old_features, events,expert_action=False,expert_ratio = 0.0) -> int:

        own_position = old_game_state["self"][3]
        own_move = np.array(new_game_state["self"][3]) - np.array(old_game_state["self"][3])
        field = old_game_state["field"]
        enemy_positions = [enemy[3] for enemy in old_game_state["others"]]


        if np.all(self.moves[-1] + own_move == np.array([0, 0])) and not np.all(own_move == np.array([0, 0])):
            if self.movement_based_rewards[-1] > 0:
                reward = -self.movement_based_rewards[-1] #only undo positive rewards
            else:
                reward = 0
        else:
            reward = 0

        movement_reward = 0

        if not np.all(own_move == np.array([0, 0])):
            self.moves.append(own_move) #only append movements

        for event in events:
            try:
                reward += self.REWARD_CONFIG[event]
                if event in ["MOVED_UP", "MOVED_DOWN", "MOVED_LEFT", "MOVED_RIGHT"]:
                    movement_reward += self.REWARD_CONFIG[event]
            except:
                print(f"No reward defined for event {event}")

        if expert_action:
            reward += self.REWARD_CONFIG["EXPERT_ACTION"] * expert_ratio
            events.append("EXPERT_ACTION")

        try:
            if "BOMB_DROPPED" in events and min(
                    [self.state_processor.distance(own_position, enemy) for enemy in enemy_positions]) < 3:
                reward += self.REWARD_CONFIG["BOMB_NEAR_ENEMY"]
                events.append("BOMB_NEAR_ENEMY")
                if min([self.state_processor.distance(own_position, enemy) for enemy in enemy_positions]) < 1:
                    reward += self.REWARD_CONFIG["BOMB_NEAR_ENEMY"] * 2
                    events.append("BOMB_NEAR_ENEMY_VERY_CLOSE")
        except:
            pass

        center = np.array([int(old_features.shape[1] - 1) / 2, int(old_features.shape[2] - 1) / 2], dtype=int)

        if sum(own_move) != 0:
            if max([old_features[5][int(center[0] + pos[0])][int(center[1] + pos[1])] for pos in moves]) == \
                    old_features[5][int(center[0] + own_move[0]), int(center[1] + own_move[1])]:
                reward += self.REWARD_CONFIG["MOVED_TOWARDS_COIN_CLUSTER"]
                movement_reward += self.REWARD_CONFIG["MOVED_TOWARDS_COIN_CLUSTER"]
                events.append("MOVED_TOWARDS_COIN_CLUSTER")
            if max([old_features[6][int(center[0] + pos[0])][int(center[1] + pos[1])] for pos in moves]) == \
                    old_features[6][int(center[0] + own_move[0]), int(center[1] + own_move[1])]:
                reward += self.REWARD_CONFIG["MOVED_TOWARDS_ENEMY"]
                movement_reward += self.REWARD_CONFIG["MOVED_TOWARDS_ENEMY"]
                events.append("MOVED_TOWARDS_ENEMY")

        self.previous_positions[own_position[0], own_position[1]] += 1

        if self.previous_positions[own_position[0], own_position[1]] > 1 or "WAITED" in events or "INVALID ACTION" in events:
            reward += self.REWARD_CONFIG["ALREADY_VISITED"] * self.previous_positions[
                own_position[0], own_position[1]]  # push to explore new areas, avoid local maximas
            events.append("ALREADY_VISITED")
        if old_features[1][center[0]+own_move[0],center[1]+own_move[1]] == 0 and old_features[1][center[0],center[1]]> 0:
            reward += self.REWARD_CONFIG["MOVED_OUT_OF_DANGER"]
            events.append("MOVED_OUT_OF_DANGER")
        if old_features[1][center[0]+own_move[0],center[1]+own_move[1]] != 0 and old_features[1][center[0],center[1]] == 0:
            reward -= self.REWARD_CONFIG["MOVED_OUT_OF_DANGER"] #handle this diffrently since its the explosion can change
            events.append("MOVED_INTO_DANGER")
        self.rewards.append(reward)

        #move inwards
        sum_moves = np.sum(np.array(self.moves),axis=0)
        if abs(sum_moves[0]) > 1 and abs(sum_moves[1]) >1:
            reward += self.REWARD_CONFIG["LEFT_START_AXIS"]
            events.append("LEFT_STRART_AXIS")
            if abs(sum_moves[0]) != field.shape[0] or abs(sum_moves[0]) != field.shape[0]:
                reward += self.REWARD_CONFIG["MOVED_INWARDS"]
                events.append("MOVED_INWARDS")
        if not np.all(own_move == np.array([0, 0])): # only append rewards from valid movements
            self.movement_based_rewards.append(movement_reward)
        return reward, events
