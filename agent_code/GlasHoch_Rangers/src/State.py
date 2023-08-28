import numpy as np


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

    def window(self, map, position, window_size, constant=-1):

        padded = np.pad = np.pad(map, window_size, mode='constant', constant_values=constant)

        return padded[position[0]:position[0] + 2 * window_size + 1, position[1]:position[1] + 2 * window_size + 1]

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
                        matrix[new_x][new_y] == 1 and
                        new_pos not in closed_set
                ):
                    new_cost = cost + 1
                    heapq.heappush(open_set, (new_cost + self.distance(new_pos, goal), new_pos))
                    if path_matrix[new_x][new_y] is None or len(path_matrix[new_x][new_y]) > len(
                            path_matrix[current[0]][current[1]]) + 1:
                        path_matrix[new_x][new_y] = path_matrix[current[0]][current[1]] + [current]

        return None

    def distance(self, pos1, pos2):
        return np.sum(np.abs(np.array(pos1) - np.array(pos2)))

    def getReachabelFields(self, field, movable_fields, pos, steps):
        reachabel_fields = np.zeros_like(field)
        reachabel_fields[pos[0], pos[1]] = 1
        for i in range(steps):
            prev_len = len(np.where(reachabel_fields == 1)[0])
            for x, y in np.where(reachabel_fields == 1):
                if movable_fields[x + 1, y] == 1:
                    reachabel_fields[x + 1, y] = 1
                if movable_fields[x - 1, y] == 1:
                    reachabel_fields[x - 1, y] = 1
                if movable_fields[x, y + 1] == 1:
                    reachabel_fields[x, y + 1] = 1
                if movable_fields[x, y - 1] == 1:
                    reachabel_fields[x, y - 1] = 1
            if len(np.where(reachabel_fields == 1)[0]) == prev_len:
                break
        return reachabel_fields

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

        explosion_map = self.get_explosion_map(field, bomb)

        enemies_pos_map = np.zeros_like(field)

        # position of enemys on map
        for i, pos in enumerate(enemies_pos):
            enemies_pos_map[pos[1], pos[0]] = 1
        enemies_pos_map = self.window(enemies_pos_map, agent_pos, self.window_size)

        coins_pos_map = np.zeros_like(field)
        for pos in coins:
            coins_pos_map[pos[1], pos[0]] = 1
        coins_pos_map = self.window(coins_pos_map, agent_pos, self.window_size)

        distances_to_coins = [self.distance(agent_pos, coin) for coin in coins]

        # direction to closest coin
        if len(distances_to_coins) > 0:
            closest_coin = coins[np.argmin(distances_to_coins)]
            direction_to_closest_coin = np.array(closest_coin) - np.array(agent_pos)

        if len(distances_to_coins) > 1:
            weighted = np.array(2, 1)
            directions = np.array(coins) - np.array(agent_pos)
            for i, direction in enumerate(directions):
                weighted += direction / distances_to_coins[i]
            weighted = weighted / len(coins)

        moveable_fields = self.get_movable_fields(field, explosion_map, bomb, enemies_pos)
        reachabel_fields = self.getReachabelFields(field, moveable_fields, agent_pos, )

        # apply windows
        field = self.window(field, agent_pos, self.window_size, constant=-2)
        explosion_map = self.window(explosion_map, agent_pos, self.window_size, constant=0)
        coins_pos_map = self.window(coins_pos_map, agent_pos, self.window_size, constant=0)
        enemies_pos_map = self.window(enemies_pos_map, agent_pos, self.window_size, constant=0)
        moveable_fields = self.window(moveable_fields, agent_pos, self.window_size, constant=-1)
        reachabel_fields = self.window(reachabel_fields, agent_pos, self.window_size, constant=0)

        # get features
