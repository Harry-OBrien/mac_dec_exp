from .models.action_state_model import ActionStateModel
from .base import Base_Agent
from .mac_action import low_level_controller
from .mac_obs import goal_extraction, localisation, mapping, teammate_detection
from .memory.macro_memory import SequentialMacroMemory

import sys
sys.path.append("../env")
from env.actions import Action

import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, LeakyReLU, Input, concatenate

from rl.policy import MaxBoltzmannQPolicy

class MacDecDDQNAgent(Base_Agent):
    def __init__(self, n_actions, observation_dim, map_dim, env, ids, update_target_every=1_000, 
                 lr=1e-3, gamma=0.99, mem_size=10_000, batch_size=16, min_mem_size=128,
                 frontier_width=6):

        # Environment stuff
        self.ids = ids
        self._numerical_id = int(self._id[-1])
        self.n_actions = n_actions
        self.observation_shape = observation_dim
        self.state_space = map_dim

        # training stuff
        self.batch_size = batch_size
        self.gamma = gamma
        self.min_mem_size = min_mem_size
        self.update_target_every = update_target_every
        self.target_update_counter = 0

        # Create model
        self.create_network(self.state_space, self.n_actions).summary()
        num_goals_combos = 64

        self.model = ActionStateModel(
            state_dim=self.state_space, 
            action_dim=self.n_actions,
            model = self.create_network(self.state_space, num_goals_combos),
            policy=MaxBoltzmannQPolicy(),
            optimizer=Adam(learning_rate=lr))

        self.target_model = ActionStateModel(
            state_dim=self.state_space, 
            action_dim=self.n_actions,
            model = self.create_network(self.state_space, num_goals_combos),
            policy=MaxBoltzmannQPolicy(),
            optimizer=Adam(learning_rate=lr))

        self.memory = SequentialMacroMemory(limit=mem_size, window_length=1)

        # TODO: Abstract all the individual agent shite into their own class
        # Macro action/observation stuff
        # self.localiser = localisation.Localisation()
        # self.teammate_detector = teammate_detection.Teammate_Detector(env, self._id)
        # self.mapping = mapping.Local_Map(map_dim=map_dim, view_dim=observation_dim, our_numerical_id=self._numerical_id)
        # self.goal_extraction = goal_extraction.Goal_Extractor(local_mapper=self.mapping, frontier_width=frontier_width)
        # self.navigator = low_level_controller.Navigation_Controller(self.mapping, self.localiser, self.teammate_detector)

        self.reset_observations()

    def create_network(self, state_space, action_space):
        # map_input: 4, 32x32 binary feature maps
        map_input = Input(shape=(*state_space, 4), name="map_input")

        # macro_obs_input: 8 vals: 
        #   robot location as 1 hot map (20x20)
        #   teammate(s) in range (1)
        #   [teammate_info] = {last goals: (20x20), current_goals(20x20)}
        #   percent complete [1]
        #   [our goals](20x20) * 64

        map_size = self.state_space[0] * self.state_space[1]
        input_length = map_size + 1 + map_size + map_size + 1 + map_size # equal to 1602 values for a 20x20 map
        macro_obs_input = Input(shape=(input_length, ), name="macro_observations_input") 

        # First branch is convolutional model to analyse map input
        # CONV2D => LReLu
        x = Conv2D(filters=8, kernel_size=(4,4), strides=(2, 2), name="C1")(map_input)
        x = LeakyReLU()(x)

        # CONV2D => LReLu
        x = Conv2D(filters=16, kernel_size=(3,3), strides=(2, 2),  name="C2")(x)
        x = LeakyReLU()(x)

        # CONV2D => LReLu
        x = Conv2D(filters=16, kernel_size=(2,2), strides=(2, 2),  name="C3")(x)
        x = LeakyReLU()(x)

        # Flattened => FC => LReLu
        x = Flatten()(x)
        x = Dense(32,  name="F1")(x)
        x = LeakyReLU()(x)

        # FC => LReLu
        x = Dense(10,  name="F2")(x)
        x = LeakyReLU()(x)

        x = Model(inputs=map_input, outputs=x)

        # Second branch is a FCL to analyse macro observations
        y = Dense(128, name="F3")(macro_obs_input)
        y = LeakyReLU()(y)
        y = Dropout(0.2)(y)
        y = Model(inputs=macro_obs_input, outputs=y)

        combined = concatenate([x.output, y.output])

        z = Dense(128,  name="F4")(combined)
        z = LeakyReLU()(z)

        z = Dense(128,  name="F5")(z)
        z = LeakyReLU()(z)

        z = Dense(128,  name="F6")(z)
        z = LeakyReLU()(z)

        model_output = Dense(action_space, activation='linear')(z)
        
        inputs = [map_input, macro_obs_input]
        return Model(inputs, model_output, name="CEP")

    def reset_observations(self):
        # print("resetting agent!")
        self.mapping.reset_maps()
        self.localiser.update_location(None)
        self.navigator.episode_reset()

        self.prev_agent_goals = {}
        self.last_known_agent_pos = {}
        self.agent_goals = {}

        self.current_goal = None
        self.last_goal = None
        
        self.mac_dec_selection_success = True

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    # Backwards pass through network to update weights
    def replay(self):
        # Only start training if we have enough samples saved
        if self.memory.nb_entries < self.min_mem_size:
            return

        minibatch = self.memory.sample(self.batch_size)

        current_states, _, _, next_states, _ = minibatch
        current_qs_list = self.model.predict(current_states, batch_size=self.batch_size)

        future_qs_list = self.target_model.predict(next_states, batch_size=self.batch_size)

        # Training data
        X, y = [], []

        for i, (state, action, reward, _, terminal) in enumerate(minibatch):
            if not terminal:
                max_future_q = np.max(future_qs_list[i])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            # update Q val for given state
            current_qs = current_qs_list[i][0]
            current_qs[action] = new_q

            # and append to training data
            X.append(state)
            y.append(current_qs)

        # Fit on all samples as one bath, log only at terminal states
        self.model.train(np.array(X), np.array(y), batch_size=self.batch_size)

        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_every:
            self.target_update_counter = 0
            self.target_update()

    # forward pass through network
    def get_action(self, observation, done):
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
        if observation is None:
            return Action.NO_MOVEMENT

        self._update_local_observations(observation)

        # Get the next macro action
        # if (self.mac_dec_selection_success and self.navigator.reached_goal()) or\
        if self.navigator.reached_goal() or self.teammate_detector.teammate_in_range():
            if self.current_goal is not None:
                self.memory.append_macro_from_real_time()

            if not done:
                self.mac_dec_selection_success = self._select_new_macro_action()

        next_move = self._select_next_move()
        assert next_move is not None

        return next_move

    def _update_local_observations(self, observation):
        # Update location
        self.localiser.update_location(observation["pos"])

        # update the observation for the teammate detector
        self.teammate_detector.update_observation(observation["robot_positions"])

        # Append observation to map
        self.mapping.append_observation(
            obstacle_observation=observation["obstacles"], 
            explored_observation=observation["explored_space"], 
            robot_positions=observation["robot_positions"],
            teammates_in_range=self.teammate_detector.teammate_in_range(),
            observation_bounds=observation["bounds"])

        # find teammates and share data
        if self.teammate_detector.teammate_in_range():
            self.teammate_detector.communicate_with_team()

    def _select_next_move(self):
        # if self.mac_dec_selection_success:
        try:
            return Action(self.navigator.next_move())
        except low_level_controller.PathNotFoundError:
            return Action.NO_MOVEMENT
        # else:
        #     # Explored everything we can for now.
        #     return Action.NO_MOVEMENT

    def _select_new_macro_action(self):
        observation, goal_candidates = self._compile_macro_observation()

        # There is a case where we may have explored all that we can in an area, but it is NOT the entire map
        # so the env doesn't think we're finished
        if len(goal_candidates) == 0:
            return False

        goal_idx = self.model.select_action(observation)
        goal_pos = goal_candidates[goal_idx]

        self.last_goal = self.current_goal
        self.current_goal = goal_pos
        try:
            self.navigator.set_goal(self.current_goal)
        except low_level_controller.PathNotFoundError:
            return False

        self.memory.macro_action_begin(observation, goal_idx)
        return True

    def _compile_macro_observation(self):
        """
        𝑧𝑖 = < 𝑥𝑖,𝜚𝑖,𝛽𝑖,𝓂𝑖,|𝐸𝑖|,𝒢𝑖 >
        zi = < pos, teammate_in_sight, {prev_goals, positions, goals}, maps, %explored, [goals]]>
        """
        goals = self.goal_extraction.generate_goals()
        self.mapping.set_goal_candidates(goals)

        pos_map = self._points_to_map_space([self.localiser.get_pos()]).flatten()

        maps = self.mapping.get_maps()
        conv_maps = [
            maps["explored_space"],
            maps["obstacles"],
            maps["robot_positions"],
            maps["goal_candidates"]
        ]
        teammate_last_goals = maps["teammate_last_goals"].flatten()
        teammate_current_goals = maps["teammate_current_goals"].flatten()

        percent_explored = self.mapping.explored_ratio
        
        macro_observation = {
            "x":pos_map,
            "delta":[self.teammate_detector.teammate_in_range()],
            "beta_last_g":teammate_last_goals,
            "beta_this_g":teammate_current_goals,
            "m":np.stack(conv_maps, axis=2),
            "E":[percent_explored],
            "g":self._points_to_map_space(goals).flatten()
        }

        return macro_observation, goals
        
    def _points_to_map_space(self, points=[]):
        env_map = np.zeros((self.state_space), dtype=int)
        for point in points:
            env_map[point] = 1

        return env_map

    def append_to_mem(self, observation, action, reward, terminal):
        if self.current_goal is not None:
            self.memory.append(observation, action, reward, terminal)
        
    # MARK: Comms
    def get_callbacks(self):
        return (self.data_tx_callback, self.data_rx_callback)

    def data_tx_callback(self):
        output_data = {
            "agent_id":self._numerical_id,
            "last_goal":self.last_goal if self.last_goal is not None else self.current_goal,
            "current_goal":self.current_goal,
            "maps":self.mapping.get_maps()
        }

        return output_data

    def data_rx_callback(self, data):
        # import the data into our maps
        self.mapping.import_shared_map(
            data["maps"], 
            data["agent_id"], 
            data["last_goal"], 
            data["current_goal"]
        )