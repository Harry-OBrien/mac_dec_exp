import numpy as np

# The teammate detection module outputs a Boolean value, delta-sub-i, to indicate if a teammate is 
# observed within sensing range d-sub-s
class Teammate_Detector:

    def __init__(self, env, our_id):
        self._env = env
        self._our_id = our_id
        self._numerical_id = int(our_id[-1])    # We're assuming that our string id is in the form '*[number]', e.g. 'agent_0'

        self._agents_in_range = []
        self._found_teammate = False

    def update_observation(self, observation):
        self._observation = observation
        self._agents_in_range = []
        self._searched_for_agents = False
        self._found_teammate = False

    def teammate_in_range(self):
        """
        states whether we have any of our teammates in our sensing range

        # Returns
            Boolean if agent in range
        """
        # Short circuit if we have already found the agents in the latest observation
        if self._searched_for_agents:
            return self._found_teammate

        if self._observation is None:
            print("WARN: Tried to detect teammates without any sensor data.")
            return False

        self._agents_in_range = self.get_agents_in_range()

        if len(self._agents_in_range) > 0:
            self._found_teammate = True
        else:
            self._found_teammate = False

        return self._found_teammate

    def get_agents_in_range(self):
        # Short circuit if we have already found the agents in the latest observation
        if self._searched_for_agents:
            return self._agents_in_range

        if self._observation is None:
            print("WARN: Tried to detect teammates without any sensor data... stoopid")
            return False

        agents_ids = []

        for row in self._observation:
            for val in row:
                # We need a +1 here because the IDs are offset by 1 in the agent map
                # There's defo a better way of doing this, but it is defo a 'TO-DO' item, yk?
                if val != 0 and val != self._numerical_id + 1:
                    agents_ids.append(val)

        self._searched_for_agents = True

        return agents_ids

    def communicate_with_team(self):
        for target_id in self._agents_in_range:
            success = self._env.communicate_with_agent(self._numerical_id, target_id)

