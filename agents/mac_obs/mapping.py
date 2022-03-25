import numpy as np

"""
The Mapping module performs both mapping and map merging. 
Each robot updates its local map, 𝓂𝑖 ∈ M, and the area explored, |𝐸𝑖 |, using its corresponding sensors.
A map is defined by a four-channel image, where each channel is a binary feature map of the explored space, obstacles, 
observed robot positions, and goal candidates. 
Local maps are merged when robots are within 𝑑𝑠 to provide updates on explored regions.
A global map, M, is generated for centralized training, by combining all robots’ local maps at each timestep.
"""
class Local_Map:
    def __init__(self, map_dim, view_dim, our_numerical_id):
        self.map_dim = map_dim
        self.n_elements = self.map_dim[0] * self.map_dim[1]
        self.view_dim = view_dim
        self.our_numerical_id = our_numerical_id

        # idgaf if you want to see 6 blocks ahead. We do 1, 3, 5, 7, etc only.
        # We could implement custom shapes in the future, but again, idgaf right now :).
        assert self.view_dim[0] % 2 == 1 and self.view_dim[1] % 2 == 1

        self.reset_maps()

    def reset_maps(self):
        """
        Resets all maps to blank values
        """
        self._maps = {"explored_space":  np.zeros(self.map_dim, dtype=int),
                      "obstacles":       np.zeros(self.map_dim, dtype=int),
                      "goal_candidates": np.zeros(self.map_dim, dtype=int)}

        self._teammate_locations = {}
        self._teammate_last_goals = {}
        self._teammate_current_goals = {}

    def get_shape(self):
        return self.map_dim

    def agent_locations(self):
        return self._teammate_locations.copy()

    def get_maps(self):
        """
        Gets the agent's local maps

        # Returns
        5 binary feature maps of the explored space, obstacles, observed robot positions, and goal candidates
        """
        output_maps = self._maps.copy()

        output_maps["robot_positions"] = np.zeros(self.map_dim, dtype=int)
        for loc in self._teammate_locations.items():
            output_maps["robot_positions"][loc] = 1

        output_maps["teammate_last_goals"] = np.zeros(self.map_dim, dtype=int)
        for last_goal in self._teammate_last_goals.items():
            if last_goal is not None:
                output_maps["teammate_last_goals"][last_goal] = 1

        output_maps["teammate_current_goals"] = np.zeros(self.map_dim, dtype=int)
        for goal in self._teammate_current_goals.items():
            if goal is not None:
                output_maps["teammate_current_goals"][goal] = 1

        # Now contains:
        #   explored_space
        #   obstacles
        #   robot_positions
        #   (teammate) last_goals
        #   (teammate) current_goals
        #   (our) goal_candidates
        return output_maps

    @property
    def explored_ratio(self):
        return np.count_nonzero(self._maps["explored_space"]) / self.n_elements

    def append_observation(self, obstacle_observation, explored_observation, robot_positions, teammates_in_range, observation_bounds):
        # Split the bounds tuple into components
        topY, botY, topX, botX = observation_bounds

        # apply onto map
        self._maps["explored_space"][topY:botY, topX:botX] |= explored_observation
        self._maps["obstacles"][topY:botY, topX:botX] |= obstacle_observation

        # find teammates in range and update their locations
        if not teammates_in_range:
            return
        
        for i, row in enumerate(range(topY, botY)):
            for j, col in enumerate(range(topX, botX)):
                robot_id = robot_positions[i, j]
                # Don't bother if it's either our ID or not an agent
                if robot_id == 0 or robot_id == self.our_numerical_id + 1:
                    continue

                self._teammate_locations[robot_id - 1] = (row, col)

    def set_goal_candidates(self, goal_candidates):
        self._maps["goal_candidates"] = np.zeros(self.map_dim, dtype=int)
        for candidate in goal_candidates:
            self._maps["goal_candidates"][candidate] = 1

    def import_shared_map(self, shared_maps, teammate_id, last_goal, current_goal):
        """
        Merges the agent's local map and the map of another agent.
        This is really easy as the agent's have perfect observation and localisation skills, so it's a case of putting a 1
        in the maps where there wasn't one

        # Argument
            shared_maps: The other agent's maps
            agent id as an int
            last goal
            current goal
        """
        # maps
        self._maps["explored_space"] |= shared_maps["explored_space"]
        self._maps["obstacles"] |= shared_maps["obstacles"]

        # last goal
        self._teammate_last_goals[teammate_id] = last_goal

        # current goal
        self._teammate_last_goals[teammate_id] = current_goal