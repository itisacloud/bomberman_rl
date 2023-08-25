import numpy as np
import copy
import torch
from collections import deque

class State():
    def __init__(self, window_size):
        self.window_size = window_size
        self.new_round()

    def new_round(self):
        self.bomb_timeout = {}

        self.previous_position = None

        self.previous_state = None

        self.previous_action = None

        self.current_step = 0


    def window(self,map,position,window_size,constant = -1):

        padded = np.pad = np.pad(map, window_size, mode='constant', constant_values=constant)

        return padded[position[0]:position[0]+2*window_size+1,position[1]:position[1]+2*window_size+1]
    def get_blast_coords(self, field, bombs_pos, blast_strength):
        x,y = bombs_pos
        blast_coords = [(x,y)]
        for i in range(1,blast_strength+1):
            if field[x+i,y] == -1:
                break
            blast_coords.append((x+i,y))
        for i in range(1,blast_strength+1):
            if field[x-i,y] == -1:
                break
            blast_coords.append((x-i,y))
        for i in range(1,blast_strength+1):
            if field[x,y+i] == -1:
                break
            blast_coords.append((x,y+i))
        for i in range(1,blast_strength+1):
            if field[x,y-i] == -1:
                break
            blast_coords.append((x,y-i))
        return blast_coords

    def get_explosion_map(self,field,bombs):

        future_explosion_map = np.zeros_like(field)

        for bomb in bombs:
            pos = bomb[0]
            timer = bomb[1]
            blast_coords = self.get_blast_coords(field,pos,bomb[1])


            for x,y in blast_coords:
                future_explosion_map[x,y] = max(4-timer,future_explosion_map[x,y])

        self.last_explosion_map = future_explosion_map

        return future_explosion_map

    def get_movable_fields(self,field,explosion_map,bombs,enemies_pos):
        movable_fields = np.zeros_like(field)
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                if field[i,j] == 0:
                    movable_fields[i,j] = 1
                else:
                    movable_fields[i,j] = -1
                if explosion_map[i,j] > 3:
                    movable_fields[i,j] = -1
        for bombs  in bombs:
            pos = bombs[0]
            movable_fields[pos[0],pos[1]] = -1

        for enemy_pos in enemies_pos:
            movable_fields[enemy_pos[0],enemy_pos[1]] = -1

        return movable_fields



    def distance(self, pos1, pos2):
        return np.sum(np.abs(np.array(pos1) - np.array(pos2)))


    def getFeatures(self, game_state):
        # get features

        self.current_step += 1

        field = game_state['field']
        bomb = game_state['bombs']
        coins = game_state['coins']

        agent_pos = game_state['self'][3]
        has_bomb = game_state['self'][2]

        explosion_map = game_state['explosion_map']

        others = game_state['others']

        enemies_pos = [i[3] for i in others]
        enemies_bomb = [i[2] for i in others]

        bomb_pos = [i[0] for i in bomb]
        bomb_timer = [i[1] for i in bomb]

        explosion_map = self.get_explosion_map(field,bomb)

        enemies_pos_map = np.zeros_like(field)


        #position of enemys on map
        for i, pos in enumerate(enemies_pos):
            enemies_pos_map[pos[1], pos[0]] = 1
        enemies_pos_map = self.window(enemies_pos_map,agent_pos, self.window_size)

        coins_pos_map = np.zeros_like(field)
        for pos in coins:
            coins_pos_map[pos[1], pos[0]] = 1
        coins_pos_map = self.window(coins_pos_map,agent_pos, self.window_size)


        distances_to_coins = [self.distance(agent_pos,coin) for coin in coins]

        #direction to closest coin
        if len(distances_to_coins) > 0:
            closest_coin = coins[np.argmin(distances_to_coins)]
            direction_to_closest_coin = np.array(closest_coin) - np.array(agent_pos)

        if len(distances_to_coins) > 1:
            weighted = np.array(2,1)
            directions = np.array(coins) - np.array(agent_pos)
            for i, direction in enumerate(directions):
                weighted += direction/distances_to_coins[i]
            weighted = weighted/len(coins)


        moveable_fields = self.get_movable_fields(field,explosion_map,bomb,enemies_pos)

        # apply windows
        field = self.window(field,agent_pos,self.window_size,constant = -2)
        explosion_map = self.window(explosion_map,agent_pos,self.window_size,constant = 0)
        coins_pos_map = self.window(coins_pos_map,agent_pos,self.window_size,constant = 0)
        enemies_pos_map = self.window(enemies_pos_map,agent_pos,self.window_size,constant = 0)
        moveable_fields = self.window(moveable_fields,agent_pos,self.window_size,constant = -1)

        # get features





