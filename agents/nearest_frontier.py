from .models.action_state_model import ActionStateModel
from .base import Base_Agent

from .mac_action import low_level_controller
from .mac_obs import goal_extraction, localisation, mapping, teammate_detection

# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, LeakyReLU, Input, concatenate

# from rl.policy import MaxBoltzmannQPolicy

import numpy as np

import sys

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
NO_ACTION = 4

# TODO: Remove AI stuff and anything not related to JUST nearest frontier
class NearestFrontierAgent(Base_Agent):
    def __init__(self, n_actions, observation_dim, map_dim, env, id, update_target_every=1_000, 
                 lr=1e-3, gamma=0.99, mem_size=10_000, batch_size=16, min_mem_size=128,
                 frontier_width=6):

        self._id = id
        self._numerical_id = int(self._id[-1])

        self.n_actions = n_actions
        self.observation_shape = observation_dim
        self.state_space = map_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.min_mem_size = min_mem_size

        self.update_target_every = update_target_every
        self.target_update_counter = 0

        # self.layers.summary()

        # self.model = ActionStateModel(
        #     state_dim=self.state_space, 
        #     action_dim=self.n_actions,
        #     model = self.layers,
        #     policy=MaxBoltzmannQPolicy(),
        #     optimizer=Adam(learning_rate=lr))

        # self.target_model = ActionStateModel(
            # state_dim=self.state_space, 
            # action_dim=self.n_actions,
            # model = self.layers,
            # policy=MaxBoltzmannQPolicy(),
            # optimizer=Adam(learning_rate=lr))

        # Macro action/observation stuff
        self.localiser = localisation.Localisation()

        self.teammate_detector = teammate_detection.Teammate_Detector(env, self._id)
        self.prev_agent_goals = {}
        self.last_known_agent_pos = {}
        self.agent_goals = {}

        self.mapping = mapping.Local_Map(map_dim=map_dim, view_dim=observation_dim, our_numerical_id=self._numerical_id)
        self.goal_extraction = goal_extraction.Goal_Extractor(local_mapper=self.mapping, frontier_width=frontier_width)
        self.current_goal = None
        self.last_goal = None

        self.navigator = low_level_controller.Navigation_Controller(self.mapping, self.localiser, self.teammate_detector)

    def reset_observations(self):
        print("resetting agent!")
        self.mapping.reset_maps()
        self.localiser.update_location(None)
        self.navigator.episode_reset()

    def target_update(self):
        pass

    # Backwards pass through network to update weights
    # Not applicable here
    def replay(self):
        pass

    # forward pass through network
    def get_action(self, observation):
        """
        Gets the agents desired action from a given observation

        param: observation

            self.observations[agent] = {
                "obstacles":pad_array(raw_observation["obstacles"], 1, *offsets),
                "robot_positions":pad_array(raw_observation["robot_positions"], 0, *offsets),
                "explored_space":pad_array(occlusion_mask, 0, *offsets),
                "bounds":bounds,
                "pos":self._agent_locations[agent],
            }

        returns
            action in range 0 -> n_actions - 1
        """
        # If we have no observation, we can't really do much...
        if observation is None:
            return NO_ACTION

        # Update location
        self.localiser.update_location(observation["pos"])

        # find teammates and share data
        self.teammate_detector.update_observation(observation["robot_positions"])
        if self.teammate_detector.teammate_in_range():
            self.teammate_detector.communicate_with_team()

        # Append observation to map
        self.mapping.append_observation(
            obstacle_observation=observation["obstacles"], 
            explored_observation=observation["explored_space"], 
            robot_positions=observation["robot_positions"],
            teammates_in_range=self.teammate_detector.teammate_in_range(),
            observation_bounds=observation["bounds"])

        # Get the next move
        next_move = self._select_next_move()
        assert next_move is not None

        # moves=["up", "right", "down", "left", "no move"]
        # print("this move:", moves[next_move])
        return next_move

    def _select_next_move(self):
        # we need to check this every time because there is a change that another agent
        # was blocking our path and this has now changed
        mac_dec_selection_success = True
        # If we need to select a new goal location
        if self.navigator.reached_goal() or self.teammate_detector.teammate_in_range():
            mac_dec_selection_success = self._select_new_macro_action()

        if mac_dec_selection_success:
            try:
                return self.navigator.next_move()
            except low_level_controller.PathNotFoundError:
                return NO_ACTION
        else:
            # Explored everything we can.
            # TODO: Stay in this state
            return NO_ACTION # No move

    def _select_new_macro_action(self):

        observation, goal_candidates = self._compile_macro_observation()

        # There is a case where we may have explored all that we can in a local area, but NOT the entire map
        # so the env doesn't think we're finished
        if len(goal_candidates) == 0:
            return False

        # This is our nearest frontier stuff
        
        # Sort the goals as distances between us and the points
        nearest_idxs = self._goal_arg_sort(goal_candidates)

        # Attempt to select the closest goal
        goal_idx = None
        for idx in nearest_idxs:
            try:
                # If we can't reach this goal, try the next closest goal
                self.navigator.set_goal(goal_candidates[idx])
                goal_idx = idx
                break
            except low_level_controller.PathNotFoundError:
                continue

        # No goal was found
        if goal_idx is None:
            # We could do some optimisation and check if a we're blocked by a teammate or an obstacle,
            # but that's a future task...
            return False

        self.last_goal = self.current_goal
        self.current_goal = goal_candidates[goal_idx]
            
        # We've now got a goal that we can reach!
        return True

    def _goal_arg_sort(self, goal_candidates):
        assert len(goal_candidates) == 4

        distances = [0] * 4
        current_pos = self.localiser.get_pos()
        for i, candidate in enumerate(goal_candidates):
            distances[i] = np.sqrt((current_pos[0] - candidate[0])**2 + (current_pos[1] - candidate[1])**2)

        return np.argsort(distances)

    def _compile_macro_observation(self):
        """
        𝑧𝑖 = < 𝑥𝑖,𝜚𝑖,𝛽𝑖,𝓂𝑖,|𝐸𝑖|,𝒢𝑖 >
        zi = < pos, teammate_in_sight, {prev_goals, positions, goals}, maps, %explored, [goals]]>
        """
        # 
        goals = self.goal_extraction.generate_goals()
        self.mapping.set_goal_candidates(goals)

        pos_map = self._points_to_map_space([self.localiser.get_pos()]).flatten()

        maps = self.mapping.get_maps()
        conv_maps = {
            "explored_space":maps["explored_space"],
            "obstacles":maps["obstacles"],
            "robot_positions":maps["robot_positions"],
            "goal_candidates":maps["goal_candidates"]
        }
        teammate_last_goals = maps["teammate_last_goals"].flatten()
        teammate_current_goals = maps["teammate_current_goals"].flatten()

        percent_explored = self.mapping.explored_ratio
        
        macro_observation = {
            "x":pos_map,
            "delta":[self.teammate_detector.teammate_in_range()],
            "beta_last_g":teammate_last_goals,
            "beta_this_g":teammate_current_goals,
            "m":np.stack(conv_maps),
            "E":[percent_explored],
            "g":self._points_to_map_space(goals).flatten()
        }

        return macro_observation, goals
        
    def _points_to_map_space(self, points=[]):
        env_map = np.zeros((self.state_space), dtype=int)
        for point in points:
            env_map[point] = 1

        return env_map

    # MARK: Comms
    def get_callbacks(self):
        return (self.data_tx_callback, self.data_rx_callback)

    def data_tx_callback(self):
        output_data = {
            "agent_id":self._numerical_id,
            "last_goal":self.last_goal if self.last_goal is None else self.current_goal,
            "current_goal":self.current_goal,
            # "position":self.localiser.get_state()[0],
            "maps":self.mapping.get_maps()
        }

        return output_data

    def data_rx_callback(self, data):
        self.mapping.import_shared_map(
            data["maps"], 
            data["agent_id"], 
            data["last_goal"], 
            data["current_goal"])

    def append_to_mem(self, observation, action, reward, terminal):
        pass