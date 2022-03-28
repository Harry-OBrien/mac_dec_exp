from math import comb
from types import new_class
import numpy as np

class MultiAgentWorld():
    
    def __init__(self, map_dim, density, transparent_walls, agents, partial_observation_prob, agent_view_shape, np_random=None):
        self._map_dim = map_dim
        self._density = density
        self._transparent_walls = transparent_walls
        self._agents = agents
        self._partial_observation_prob = partial_observation_prob
        self._agent_view_shape = agent_view_shape
        self._np_random = np_random

        self._map_size = self._map_dim[0] * self._map_dim[1]
        self._agent_name_mapping = dict(zip(self._agents, list(range(1, len(self._agents)+1))))
        self._agent_locations = {}
    # MARK: Properties
    @property
    def shape(self):
        return self._map_dim

    @property
    def map_size(self):
        return self._map_size

    @property
    def maps(self):
        output_maps = self._maps.copy()
        del output_maps["local_exploration_maps"]

        # Robot Positions
        output_maps["robot_positions"] = np.zeros(self._map_dim, dtype=int)

        for agent, location in self._agent_locations.items():
            num_id = self._agent_name_mapping[agent]
            output_maps["robot_positions"][location] = num_id

        return output_maps

    @property
    def agent_locations(self):
        return self._agent_locations

    def agent_location(self, agent):
        return self._agent_locations[agent]

    @property
    def nb_cells_explored(self):
        return np.count_nonzero(self._maps["explored_space"])

    @property
    def search_complete(self):
        return self.nb_cells_explored == self._map_size

    # -------------------- Public functions -------------------- #
    def on_step_complete(self):
        # Reset values
        self._unique_cells_seen = {agent: 0 for agent in self._agents}
        self._non_unique_cells_seen = {agent: 0 for agent in self._agents}

    def calculate_reward(self, agent):
        reward = 0
        if agent == "global":
            reward -= 1
            for teammate in self._agents:
                reward += self._unique_cells_seen[teammate]
                reward -= self._non_unique_cells_seen[teammate]
            
            if self.search_complete:
                reward += 100
        else:
            reward += self._unique_cells_seen[agent]
            reward -= self._non_unique_cells_seen[agent]

            sight_count = self._teammate_in_sight_count[agent]
            if sight_count >= 2 and sight_count <= 7:
                reward -= np.sqrt(np.exp(sight_count)/5)
            elif sight_count > 7:
                reward -= 15

        return reward

    def reset(self):
        self._generate_maps()
        self._agent_locations = {agent:self._get_free_location() for agent in self._agents}

        self._unique_cells_seen = {agent: 0 for agent in self._agents}
        self._non_unique_cells_seen = {agent: 0 for agent in self._agents}
        self._teammate_in_sight_count = {agent: 0 for agent in self._agents}

    # MARK: Actions
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

        current_location = self._agent_locations[agent]
        new_location = self._offset_point(current_location, offset)

        # If new location is in bounds and not occupied, move the agent
        if self._location_in_bounds(new_location) and not self._location_occupied(new_location):
            self._agent_locations[agent] = new_location

            return 1

        return 0

    # MARK: Observations
    def observation_for_agent(self, agent):
        pos = self._agent_locations[agent]

        view_bounds = self._get_view_exts(pos)
        corrected_bounds, offsets = self._correct_bounds(*view_bounds, self._map_dim)

        # get the observations
        observation = self._map_slice(*corrected_bounds)
        pos_in_obs = self._pos_in_bounds(offsets)

        # Apply occlusion mask to remove anything that the agent cant see
        mask = np.ones_like(observation["obstacles"], dtype=bool)
        if not self._transparent_walls:
            mask = self._occlude_mask(~observation["obstacles"].astype(bool), pos_in_obs)

        # Slight chance that the agent doesn't observe a cell
        for i, row in enumerate(mask):
            for j, _ in enumerate(row):
                if self._np_random.uniform() < self._partial_observation_prob:
                    mask[i, j] = False

        for key, obs_map in observation.items():
            for i, row in enumerate(obs_map):
                for j, _ in enumerate(row):
                    if not mask[i, j]:
                        observation[key][i,j] = 0

        # Counting values for later reward functions
        ty, by, tx, bx = corrected_bounds
        observation_size = mask.shape[0] * mask.shape[1]
        combined_maps = np.zeros_like(mask, dtype=int)
        for teammate, local_map in self._maps["local_exploration_maps"].items():
            # Update global map
            if teammate == agent:
                self._maps["local_exploration_maps"][teammate][ty:by, tx:bx] |= mask
                self._maps["explored_space"][ty:by, tx:bx] |= mask

            # Build combination of other teammates maps
            combined_maps |= local_map[ty:by, tx:bx]
        
        non_unique_cell_count = np.count_nonzero(combined_maps)
        self._unique_cells_seen[agent] = observation_size - non_unique_cell_count
        self._non_unique_cells_seen[agent] = non_unique_cell_count

        if self._teammate_in_observation(agent, observation["robot_positions"]):
            self._teammate_in_sight_count[agent] += 1
        else:
            self._teammate_in_sight_count[agent] = 0

        return observation, corrected_bounds, offsets, mask

    # ------------------- Private functions ------------------- #
    # MARK: Util
    def _generate_maps(self):
        clutter_density = self._density if self._density is not None\
            else self._np_random.uniform(0.3, 0.7)

        self._maps = {
            "obstacles":np.array([[1 if self._np_random.uniform() < clutter_density else 0 for _ in range(self._map_dim[0])] for _ in range(self._map_dim[1])], dtype=int),
            "explored_space":np.zeros(self._map_dim, dtype=int),
            "local_exploration_maps":{agent:np.zeros(self._map_dim, dtype=int) for agent in self._agents}
        }

    def _location_occupied(self, location):
        return self._maps["obstacles"][location] == 1 or location in self._agent_locations.values()

    def _location_in_bounds(self, location):
        row, col = location
        max_h, max_w = self._map_dim

        return row >= 0 and row < max_h\
            and col >= 0 and col < max_w

    def _get_free_location(self):
         # Keep trying to select a location for the agent to start
        h, w = self._map_dim
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

    # MARK: Observation Private Fxs
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
        h, w = self._map_dim

        assert botY >= 0 and topY < h
        assert botX >= 0 and topX < w

        slices = {}
        for key, map_value in self.maps.items():
            slices[key] = map_value.copy()[topY:botY, topX:botX]

        return slices

    def _pos_in_bounds(self, offsets):
        w, h = self._agent_view_shape
        ty_off, _, tx_off, _ = offsets
        
        y = h//2 - ty_off
        x = w//2 - tx_off
            
        return (y, x)
        
    def _occlude_mask(self, grid, agent_pos):
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
                        
        return mask

    def _teammate_in_observation(self, agent, robot_positions):
        for row in robot_positions:
            for id in row:
                if id != 0 and id != self._agent_name_mapping[agent]:
                    return True
        
        return False