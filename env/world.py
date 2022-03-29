from math import comb
from operator import ne
from types import new_class
import numpy as np
from sklearn import neighbors

class MultiAgentWorld():
    
    def __init__(self, map_dim, density, transparent_walls, agents, partial_observation_prob, agent_view_shape, np_random=None):
        self.shape = map_dim
        self._density = density
        self._transparent_walls = transparent_walls
        self._agents = agents
        self._partial_observation_prob = partial_observation_prob
        self._agent_view_shape = agent_view_shape
        self._np_random = np_random

        self.map_size = self.shape[0] * self.shape[1]
        self._agent_name_mapping = dict(zip(self._agents, list(range(1, len(self._agents)+1))))
        self.agent_locations = {}

    # MARK: Reset
    def reset(self):
        self._generate_maps()
        self.agent_locations = {agent:self._get_free_location() for agent in self._agents}

        self._explorable_cells = self._find_agent_exploration_area()

        self._unique_cells_seen = {agent: 0 for agent in self._agents}
        self._non_unique_cells_seen = {agent: 0 for agent in self._agents}
        self._teammate_in_sight_count = {agent: 0 for agent in self._agents}

    def _find_agent_exploration_area(self):
        areas = {agent: None for agent in self._agents}

        for agent in self._agents:
            if areas[agent] is None:
                flood_map, applicable_agents = self._flood_fill_from_point(self.agent_locations[agent])
                for flood_agent in applicable_agents:
                    areas[flood_agent] = (flood_map, np.count_nonzero(flood_map))

        return areas

    def _flood_fill_from_point(self, start):
        agent_map = self.maps["robot_positions"]

        flood_map = np.zeros(self.shape, dtype=int)
        applicable_agents = []

        next_points = [start]
        while len(next_points) > 0:
            point = next_points.pop()
            flood_map[point] = 1

            # If there is an agent here, make a note of it
            if agent_map[point] != 0:
                applicable_agents.append("agent_" + str(agent_map[point] - 1))

            neighbours = self._unnocupied_neighbours(point)
            for n in neighbours:
                if flood_map[n] == 0:
                    next_points.append(n)

        return flood_map, applicable_agents

    def _unnocupied_neighbours(self, point):
        neighbours = []
        for offset in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            potential_loc = (point[0] + offset[0], point[1] + offset[1])
            if self._location_in_bounds(potential_loc) and self._maps["obstacles"][potential_loc] == 0:
                neighbours.append(potential_loc)

        return neighbours

    def _generate_maps(self):
        clutter_density = self._density if self._density is not None\
            else self._np_random.uniform(0.3, 0.7)

        self._maps = {
            "obstacles":np.array([[1 if self._np_random.uniform() < clutter_density else 0 for _ in range(self.shape[0])] for _ in range(self.shape[1])], dtype=int),
            "explored_space":np.zeros(self.shape, dtype=int),
            "local_exploration_maps":{agent:np.zeros(self.shape, dtype=int) for agent in self._agents}
        }

    # MARK: Action
    def move(self, agent, offset):
        """
        moves the given agent by the value offset. offset can only move 1 square up, down,
        left or right, and not diagonally.

        returns the distance the agent travelled
        """
        y_off, x_off = offset
        if y_off == 0 and x_off == 0:
            return 0

        assert y_off != x_off
        assert y_off <= 1 and y_off >= -1
        assert x_off <= 1 and x_off >= -1

        current_location = self.agent_locations[agent]
        new_location = self._offset_point(current_location, offset)

        # If new location is in bounds and not occupied, move the agent
        if self._location_in_bounds(new_location) and not self._location_occupied(new_location):
            self.agent_locations[agent] = new_location

            return 1

        return 0

    # MARK: Observation
    @property
    def maps(self):
        output_maps = self._maps.copy()
        del output_maps["local_exploration_maps"]

        # Robot Positions
        output_maps["robot_positions"] = np.zeros(self.shape, dtype=int)

        for agent, location in self.agent_locations.items():
            num_id = self._agent_name_mapping[agent]
            output_maps["robot_positions"][location] = num_id

        return output_maps

    def agent_location(self, agent):
        return self.agent_locations[agent]

    @property
    def nb_cells_explored(self):
        return np.count_nonzero(self._maps["explored_space"])

    def search_fully_complete(self):
        return self.nb_cells_explored == self.map_size

    def agent_complete(self, agent):
        agent_local_map = self._maps["local_exploration_maps"][agent]
        possible_exploration_map, length = self._explorable_cells[agent]

        non_zero_count = np.count_nonzero(possible_exploration_map & agent_local_map)
        return non_zero_count == length

    def observation_for_agent(self, agent):
        pos = self.agent_locations[agent]

        # observation bounds
        view_bounds = self._get_view_exts(pos)
        corrected_bounds, offsets = self._correct_bounds(*view_bounds, self.shape)

        # get the observation(s)
        observation = self._map_slice(*corrected_bounds)
        pos_in_obs = self._pos_in_view_bounds(offsets)

        # Apply occlusion mask to remove anything that the agent cant see
        if self._transparent_walls:
            visible_mask = np.ones_like(observation["obstacles"], dtype=bool)
        else:
            visible_mask = self._generate_occlusion_mask(~observation["obstacles"].astype(bool), pos_in_obs)

        # Apply mask
        for key, obs_map in observation.items():
            for i, row in enumerate(obs_map):
                for j, _ in enumerate(row):
                    if not visible_mask[i, j]:
                        observation[key][i,j] = 0

        ty, by, tx, bx = corrected_bounds
        self._maps["explored_space"][ty:by, tx:bx] |= visible_mask
        self._maps["local_exploration_maps"][agent][ty:by, tx:bx] |= visible_mask

        return observation, corrected_bounds, offsets, visible_mask

    def _get_view_exts(self, pos):
        """
        Get the extents of the set of tiles visible to the agent
        Note: the indices COULD be out of bounds and need to be checked for
        """
        h, w = self._agent_view_shape
        offset_h = h//2
        offset_w = w//2

        # topY, botY, topX, botX
        return (
            pos[0] - offset_h,
            pos[0] + offset_h + 1,
            pos[1] - offset_w,
            pos[1] + offset_w + 1
        )

    def _correct_bounds(self, topY, botY, topX, botX, map_dim):
        """
        Returns the corrected indices and the offsets applied
        """
        # Fix any out of bounds issues
        top_x_offset = 0
        bot_x_offset = 0
        top_y_offset = 0
        bot_y_offset = 0

        if topY < 0:
            top_y_offset = -topY
        elif botY >= map_dim[0]:
            bot_y_offset = botY - map_dim[0]

        if topX < 0:
            top_x_offset = -topX
        elif botX >= map_dim[1]:
            bot_x_offset = botX - map_dim[1]

        # create slices
        (t,b,l,r) = (
            topY + top_y_offset,
            botY - bot_y_offset, 
            topX + top_x_offset,
            botX - bot_x_offset
        )

        return (t,b,l,r), (top_y_offset, bot_y_offset, top_x_offset, bot_x_offset)

    def _map_slice(self, topY, botY, topX, botX):
        h, w = self.shape

        assert botY >= 0 and topY < h
        assert botX >= 0 and topX < w

        slices = {}
        for key, map_value in self.maps.items():
            slices[key] = map_value.copy()[topY:botY, topX:botX]

        return slices

    def _pos_in_view_bounds(self, offsets):
        w, h = self._agent_view_shape
        ty_off, _, tx_off, _ = offsets
        
        y = h//2 - ty_off
        x = w//2 - tx_off
            
        return (y, x)
        
    def _generate_occlusion_mask(self, grid, agent_pos):
        mask = np.zeros(grid.shape[:2], dtype=bool)
        mask[agent_pos[0], agent_pos[1]] = True
        width, height = grid.shape[:2]

        for j in range(agent_pos[1],0,-1):
            for i in range(agent_pos[0], width):
                if mask[i,j] and grid[i,j]:
                    if i < width - 1:
                        mask[i + 1, j] = True
                    if j > 0:
                        mask[i, j - 1] = True
                        if i < width - 1:
                            mask[i + 1, j - 1] = True

            for i in range(agent_pos[0],0,-1):
                if mask[i,j] and grid[i,j]:    
                    if i > 0:
                        mask[i - 1, j] = True
                    if j > 0:
                        mask[i, j - 1] = True
                        if i > 0:
                            mask[i - 1, j - 1] = True


        for j in range(agent_pos[1], height):
            for i in range(agent_pos[0], width):
                if mask[i,j] and grid[i,j]:
                    if i < width - 1:
                        mask[i + 1, j] = True
                    if j < height-1:
                        mask[i, j + 1] = True
                        if i < width - 1:
                            mask[i + 1, j + 1] = True

            for i in range(agent_pos[0],0,-1):
                if mask[i,j] and grid[i,j]:
                    if i > 0:
                        mask[i - 1, j] = True
                    if j < height-1:
                        mask[i, j + 1] = True
                        if i > 0:
                            mask[i - 1, j + 1] = True

         # Slight chance that the agent doesn't observe a cell
        for i, row in enumerate(mask):
            for j, _ in enumerate(row):
                if self._np_random.uniform() < self._partial_observation_prob:
                    mask[i, j] = False   
        return mask

    # MARK: Util
    def combine_maps(self, agent_1, agent_2):
        self._maps["local_exploration_maps"][agent_1] |= self._maps["local_exploration_maps"][agent_2]
        self._maps["local_exploration_maps"][agent_2] = self._maps["local_exploration_maps"][agent_1].copy()

    def _location_occupied(self, location):
        assert self._location_in_bounds(location)

        return self._maps["obstacles"][location] == 1 or\
            location in self.agent_locations.values()

    def _location_in_bounds(self, location):
        row, col = location
        max_h, max_w = self.shape

        return row >= 0 and row < max_h\
            and col >= 0 and col < max_w

    def _get_free_location(self):
         # Keep trying to select a location for the agent to start
        h, w = self.shape
        while True:
            pos = (
                self._np_random.randint(h - 1), 
                self._np_random.randint(w - 1)
            )

            # If unoccupied, we can continue
            if not self._location_occupied(pos):
                break

        return pos

    def _offset_point(self, point, offset):
        return (
            point[0] + offset[0],
            point[1] + offset[1]
        )
