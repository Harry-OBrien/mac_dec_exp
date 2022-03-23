import numpy as np

# The teammate detection module outputs a Boolean value, delta-sub-i, to indicate if a teammate is 
# observed within sensing range d-sub-s
class Teammate_Detector:

    def __init__(self, env, our_id):
        self._env = env
        self._our_id = our_id
        self._numerical_id = int(our_id[-1])

        self._agents_in_range = []

    def update_observation(self, observation):
        self._observation = observation

    def teammate_in_range(self):
        """
        Takes the sensing range of the agent and returns all visible agents in this range (including ourself)

        # Argument
            the idices for where we want to slice the map/our sensing range

        # Returns
            Boolean if agent in range
        """
        if self._observation is None:
            print("WARN: Tried to detect teammates without any sensor data... stoopid")
            return False

        self._agents_in_range = []

        for row in self._observation:
            for val in row:
                # We need a +1 here because the IDs are offset by 1 in the agent map
                # There's defo a better way of doing this, but it is defo a 'TO-DO' item, yk?
                if val != 0 and val != self._numerical_id + 1:
                    self._agents_in_range.append(val)

        if len(self._agents_in_range) > 0:
                return True

        return False

    def get_shared_obs(self):
        for target_id in self._agents_in_range:
            success = self._env.communicate_with_agent(self._numerical_id, target_id)

