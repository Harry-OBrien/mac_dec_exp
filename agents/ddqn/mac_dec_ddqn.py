from .core_agent.core_agent import CoreAgent
from .models.action_state_model import ActionStateModel
from .base import Base_Agent
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
    def __init__(self, n_actions, observation_dim, map_dim, env, id, centalised_model, update_target_every=1_000, 
                 lr=1e-3, gamma=0.99, mem_size=10_000, batch_size=16, min_mem_size=128,
                 frontier_width=6):

        # Environment stuff
        self._id = id
        self._numerical_id = int(self._id[-1])
        self.n_actions = n_actions
        self.observation_shape = observation_dim
        self._map_dim = map_dim
        self.centalised_model = centalised_model

        # training stuff
        self.batch_size = batch_size
        self.gamma = gamma
        self.min_mem_size = min_mem_size
        self.update_target_every = update_target_every
        self.target_update_counter = 0

        # Create model
        self.create_network(self._map_dim, 4).summary()
        num_goals = 4

        self.model = ActionStateModel(
            model = self.create_network(self._map_dim, num_goals),
            policy=MaxBoltzmannQPolicy(),
            optimizer=Adam(learning_rate=lr))

        self.target_model = ActionStateModel(
            model = self.create_network(self._map_dim, num_goals),
            policy=MaxBoltzmannQPolicy(),
            optimizer=Adam(learning_rate=lr))

        self.memory = SequentialMacroMemory(limit=mem_size, window_length=1)

        # Macro action/observation stuff
        self.core = CoreAgent(id, env, map_dim, observation_dim, frontier_width)
        self.reset_observations()

    def reset_observations(self):
        self.core.reset_observations()

    def create_network(self, state_space, action_space):
        # map_input: four 20x20 binary feature maps
        map_input = Input(shape=(*state_space, 4), name="map_input")

        # macro_obs_input: 8 vals: 
        #   robot location as 1 hot map (20x20)
        #   teammate(s) in range (1)
        #   [teammate_info] = {last goals: (20x20), current_goals(20x20)}
        #   percent complete [1]
        #   [our goals](20x20)

        assert len(state_space) == 2

        map_size = state_space[0] * state_space[1]
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
        y = Dense(64, name="F3")(macro_obs_input)
        y = LeakyReLU()(y)
        y = Dropout(0.2)(y)
        y = Model(inputs=macro_obs_input, outputs=y)

        combined = concatenate([x.output, y.output])

        z = Dense(64,  name="F4")(combined)
        z = LeakyReLU()(z)

        z = Dense(64,  name="F5")(z)
        z = LeakyReLU()(z)

        z = Dense(64,  name="F6")(z)
        z = LeakyReLU()(z)

        model_output = Dense(action_space, activation='linear')(z)
        
        inputs = [map_input, macro_obs_input]
        return Model(inputs, model_output, name="DEP")

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    # Backwards pass through network to update weights
    def replay(self):
        # Only start training if we have enough samples saved
        if self.memory.nb_entries < self.min_mem_size:
            return

        minibatch = self.memory.sample(self.batch_size)

        current_states = [experience[0] for experience in minibatch]
        current_qs_list = self.model.predict(current_states, batch_size=self.batch_size)

        next_states = [experience[3] for experience in minibatch]
        # USING THE CENTRALISED MODEL HERE! :D (CTDE)
        future_qs_list = self.centalised_model.predict(next_states, batch_size=self.batch_size)

        # Training data
        X = [] # the input data (top of network)
        y = [] # the output rewards (bottom of network)

        for i, (state, action, reward, next_state, terminal) in enumerate(minibatch):
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
    def get_action(self, observation):
       
        self.core.update_local_observation(observation)

        recalculate_action = self.core.can_see_teammate or self.core.reached_goal or not self.core.can_move_to_goal

        if recalculate_action:
            macro_observation = self.macro_observation
            if len(self.core.goal_candidates) > 0:
                macro_observation = [macro_observation.pop("m"), np.concatenate(list(macro_observation.values()))]
                reshaped = [macro_observation[0].reshape(1, *macro_observation[0].shape),
                            macro_observation[1].reshape(1, *macro_observation[1].shape)]

                goal_idx = self.model.select_action(reshaped)
                # goal_idx = 0
                self.core.current_goal = self.core.goal_candidates[goal_idx]
                self.memory.macro_action_begin(reshaped, goal_idx)
            else:
                self.core.current_goal = None

            # print(self.core.current_goal)

        return self.core.next_action

    @property
    def macro_observation(self):
        local_maps = self.core.maps

        conv_maps = [
            local_maps["explored_space"],
            local_maps["obstacles"],
            local_maps["robot_positions"],
            local_maps["goal_candidates"]
        ]

        location_map = np.zeros(self._map_dim)
        location_map[self.core.pos] = 1

        return {
            "x":location_map.flatten(),
            "delta":[self.core.can_see_teammate],
            "beta_last_g":local_maps["teammate_last_goals"].flatten(),
            "beta_this_g":local_maps["teammate_current_goals"].flatten(),
            "m":np.stack(conv_maps, axis=2),
            "E":[self.core.explored_ratio],
            "g":local_maps["goal_candidates"].flatten()
        }

    def append_to_mem(self, observation, action, reward, terminal):
        self.memory.append(observation, action, reward, terminal) 

    # MARK: Comms
    def get_callbacks(self):
        return self.core.callbacks
