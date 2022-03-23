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
    def __init__(self, map_dim, view_dim):
        self.map_dim = map_dim
        self.n_elements = self.map_dim[0] * self.map_dim[1]
        self.view_dim = view_dim

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
                      "robot_positions": np.zeros(self.map_dim, dtype=int),
                      "goal_candidates": np.zeros(self.map_dim, dtype=int)}

    def get_shape(self):
        return self.map_dim

    def get_maps(self):
        """
        Gets the agent's local maps

        # Returns
        4 binary feature maps of the explored space, obstacles, observed robot positions, and goal candidates
        """
        return self._maps

    @property
    def explored_ratio(self):
        return np.count_nonzero(self._maps["explored_space"]) / self.n_elements

    def append_observation(self, obstacle_observation, explored_observation, teammate_observation, observation_bounds):
        topY, botY, topX, botX = observation_bounds

        self._maps["explored_space"][topY:botY, topX:botX] |= explored_observation
        self._maps["obstacles"][topY:botY, topX:botX] |= obstacle_observation

        # TODO: Only remove position of teammates if we can see them in this observation
        self._maps["robot_positions"] = np.zeros(self.map_dim, dtype=int)
        self._maps["robot_positions"][topY:botY, topX:botX] |= teammate_observation

    def set_goal_candidates(self, goal_candidates):
        self._maps["goal_candidates"] = np.zeros(self.map_dim, dtype=int)
        for candidate in goal_candidates:
            self._maps["goal_candidates"][candidate] = 1

    def import_shared_map(self, shared_maps):
        """
        Merges the agent's local map and the map of another agent.
        This is really easy as the agent's have perfect observation and localisation skills, so it's a case of putting a 1
        in the maps where there wasn't one

        # Argument
            shared_maps: The other agent's maps as 4 binary feature maps
        """
        for key, shared_map in shared_maps.items():
            if key == "goal_candidates":
                continue

            self._maps[key] |= shared_map