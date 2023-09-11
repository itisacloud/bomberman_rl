import heapq

import numpy as np
import torch


rotations = [0, 1, 2, 3]


class State:
    def __init__(self, window_size,enemies=0):
        self.window_size = window_size
        self.new_round()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enemies = enemies
        self.rotation = 0

    def new_round(self):
        self.bomb_timeout = {}

        self.previous_position = None

        self.previous_state = None

        self.previous_action = None

        self.current_step = 0

    def window(self, map, position, window_size, constant=-1):
        padded = np.pad(map, pad_width= window_size, mode='constant', constant_values=constant)
        return padded[position[0]:position[0] + 2 * window_size + 1, position[1]:position[1] + 2 * window_size + 1]

    def extra_to_map(self,extra_features,field):
        maps = []
        for feature in extra_features:
            if feature == 0 or feature == False:
                maps.append(np.zeros_like(field))
            else:
                if feature == True:
                    feature = 1
                maps.append(np.ones_like(field)*feature)
        return np.array(maps)

    def get_blast_coords(self, field, bombs_pos, blast_strength):
        x, y = bombs_pos
        blast_coords = [(x, y)]
        for i in range(1, blast_strength + 1):
            if field[x + i, y] == -1:
                break
            blast_coords.append((x + i, y))
        for i in range(1, blast_strength + 1):
            if field[x - i, y] == -1:
                break
            blast_coords.append((x - i, y))
        for i in range(1, blast_strength + 1):
            if field[x, y + i] == -1:
                break
            blast_coords.append((x, y + i))
        for i in range(1, blast_strength + 1):
            if field[x, y - i] == -1:
                break
            blast_coords.append((x, y - i))
        return blast_coords

    def get_explosion_map(self, field, bombs):
        future_explosion_map = np.zeros_like(field)

        for bomb in bombs:
            pos = bomb[0]
            timer = bomb[1]
            blast_coords = self.get_blast_coords(field, pos, bomb[1])

            for x, y in blast_coords:
                future_explosion_map[x, y] = max(4 - timer, future_explosion_map[x, y])
        self.last_explosion_map = future_explosion_map

        return future_explosion_map

    def get_movable_fields(self, field, explosion_map, bombs, enemies_pos):
        movable_fields = np.zeros_like(field)
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                if field[i, j] == 0:
                    movable_fields[i, j] = 1
                else:
                    movable_fields[i, j] = -1
                if explosion_map[i, j] > 3:
                    movable_fields[i, j] = -1
        for bombs in bombs:
            pos = bombs[0]
            movable_fields[pos[0], pos[1]] = -1

        for enemy_pos in enemies_pos:
            movable_fields[enemy_pos[0], enemy_pos[1]] = -1

        return movable_fields

    def a_star(self, matrix, start, goal):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        open_set = [(0, start)]  # Priority queue (cost, position)
        closed_set = set()
        path_matrix = [[None for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
        path_matrix[start[0]][start[1]] = []

        while open_set:
            cost, current = heapq.heappop(open_set)

            if current == goal:
                return path_matrix[current[0]][current[1]]

            closed_set.add(current)

            for dx, dy in directions:
                new_x, new_y = current[0] + dx, current[1] + dy
                new_pos = (new_x, new_y)

                if (
                        0 <= new_x < len(matrix) and
                        0 <= new_y < len(matrix[0]) and
                        matrix[new_x][new_y] == 0 and
                        new_pos not in closed_set
                ):
                    new_cost = cost + 1
                    heapq.heappush(open_set, (new_cost + self.distance(new_pos, goal), new_pos))
                    if path_matrix[new_x][new_y] is None or len(path_matrix[new_x][new_y]) > len(
                            path_matrix[current[0]][current[1]]) + 1:
                        path_matrix[new_x][new_y] = path_matrix[current[0]][current[1]] + [current]

        return None
    def paths_to_idw_matrix(self,field,paths):
        matrix = np.zeros_like(field,dtype=np.float32)
        for path in paths:
            if path is None:
                continue
            for i,pos_path in enumerate(path):
                matrix[pos_path[0]][pos_path[1]] += 1 / len(path)
        return matrix



    def distance(self, pos1, pos2):
        return np.sum(np.abs(np.array(pos1) - np.array(pos2)))

    def getReachabelFields(self, field, movable_fields, pos, steps=20):
        reachable_fields = np.zeros_like(field, dtype=np.int32)
        reachable_fields[pos[0], pos[1]] = 1

        for i in range(steps):
            prev_reachable = np.copy(reachable_fields)

            for x, y in np.argwhere(reachable_fields == 1):
                neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

                for nx, ny in neighbors:
                    if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and movable_fields[nx, ny] == 1:
                        reachable_fields[nx, ny] = 1

            if np.array_equal(reachable_fields, prev_reachable):
                break

        return reachable_fields

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

        self.enemies = max(self.enemies,len(others))

        enemies_pos = [i[3] for i in others]
        enemies_bomb = [i[2] for i in others]

        if len(enemies_bomb) < self.enemies:
            enemies_bomb += [False for _ in range(self.enemies-len(enemies_bomb))]

        explosion_map = self.get_explosion_map(field, bomb)

        enemies_pos_map = np.zeros_like(field)

        # position of enemys on map
        for i, pos in enumerate(enemies_pos):
            enemies_pos_map[pos[0], pos[1]] = 1

        coins_pos_map = np.zeros_like(field)
        for pos in coins:
            coins_pos_map[pos[1], pos[0]] = 1

        paths_to_coins = [self.a_star(field, agent_pos, coin) for coin in coins]
        coins_idw_map = self.paths_to_idw_matrix(field, paths_to_coins)

        paths_to_enemies = [self.a_star(field, agent_pos, enemy) for enemy in enemies_pos]
        enemies_idw_map = self.paths_to_idw_matrix(field, paths_to_enemies)

        moveable_fields = self.get_movable_fields(field, explosion_map, bomb, enemies_pos)
        reachabel_fields = self.getReachabelFields(field, moveable_fields, agent_pos, )
        # apply windows

        field = self.window(field, agent_pos, self.window_size, constant=-2)
        explosion_map = self.window(explosion_map, agent_pos, self.window_size, constant=0)
        coins_pos_map = self.window(coins_pos_map, agent_pos, self.window_size, constant=0)
        enemies_pos_map = self.window(enemies_pos_map, agent_pos, self.window_size, constant=0)
        reachabel_fields = self.window(reachabel_fields, agent_pos, self.window_size, constant=-1)
        coins_idw_map = self.window(coins_idw_map, agent_pos, self.window_size, constant=0)
        enemies_idw_map = self.window(enemies_idw_map, agent_pos, self.window_size, constant=0)

        features = np.array([field,explosion_map,coins_pos_map,enemies_pos_map,reachabel_fields,coins_idw_map,enemies_idw_map]) # get features

        features = torch.tensor(features).to(torch.float32).to(self.device)

        return features
