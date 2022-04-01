from agents.core_agent.core_agent import CoreAgent
from .models.action_state_model import ActionStateModel
from .base import Base_Agent
from .mac_action import low_level_controller
from .mac_obs import goal_extraction, localisation, mapping, teammate_detection
from .memory.macro_memory import SequentialMacroMemory

import sys
sys.path.append("../env")
from env.actions import Action

import numpy as np
import itertools

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, LeakyReLU, Input, concatenate

from rl.policy import MaxBoltzmannQPolicy

# TODO: Abstract a load of shit away that we've inserted into core_agent.py
class MacDecDDQNAgent(Base_Agent):
    def __init__(self, n_actions, observation_dim, map_dim, env, ids, update_target_every=1_000, 
                 lr=1e-3, gamma=0.99, mem_size=10_000, batch_size=16, min_mem_size=128,
                 frontier_width=6):

        # local stuff
        self.n_actions = n_actions
        self.goal_combinations = None
        self.latest_potential_candidates = None
        self.teammates_in_range = False
        self._map_dim = map_dim

        # training stuff
        self.batch_size = batch_size
        self.gamma = gamma
        self.min_mem_size = min_mem_size
        self.update_target_every = update_target_every
        self.target_update_counter = 0

        # Create model
        self.create_network(self.state_space, 64).summary()
        num_goals_combos = 64

        self.model = ActionStateModel(
            model = self.create_network(self.state_space, num_goals_combos),
            policy=MaxBoltzmannQPolicy(),
            optimizer=Adam(learning_rate=lr))

        self.target_model = ActionStateModel(
            model = self.create_network(self.state_space, num_goals_combos),
            policy=MaxBoltzmannQPolicy(),
            optimizer=Adam(learning_rate=lr))

        # create memory
        self.memory = SequentialMacroMemory(limit=mem_size, window_length=1)

        # Macro action/observation stuff
        self._agents = {}
        for id in ids:
            self._agents[id] = CoreAgent(id, env, map_dim, observation_dim, frontier_width)
            self._agents[id].reset_observations()

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
        for agent in self.agents.values():
            agent.reset_observations()

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    # Backwards pass through network to update weights
    # TODO: Extract values from training to use in decentralised execution
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
    def get_action(self, observations):
        assert len(observations) == len(self._agents)
        
        recalculate_action = False
        goals_are_none = True
        self.teammates_in_range = False
        for agent_id, observation in observations.items():
            self._agents[agent_id].update_local_observation(observation)
            self.teammates_in_range |= self._agents[agent_id].can_see_teammate
            goals_are_none &= (self._agents[agent_id].current_goal is None)
            recalculate_action |= (self.teammates_in_range | self._agents[agent_id].reached_goal | goals_are_none)

        if recalculate_action:
            # get goals
            self.latest_potential_candidates = [agent.goal_candidates for agent in self._agents.values()]
            self.goal_combinations = list(itertools.product(*self.latest_potential_candidates))
            assert len(self.goal_combinations) == 4**len(self.agents)

            team_goal_idx = self.model.select_action(self.macro_observation)
            goals = self.goal_combinations[team_goal_idx]

            for i, agent in enumerate(self._agents.values()):
                agent.current_goal = goals[i]

        return {agent_id: agent.next_action for agent_id, agent in self._agents.items()}

    @property
    def macro_observation(self):
        """
        𝑧𝑖 = < 𝑥𝑖,𝜚𝑖,𝛽𝑖,𝓂𝑖,|𝐸𝑖|,𝒢𝑖 >
        zi = < pos, teammate_in_sight, {prev_goals, positions, goals}, maps, %explored, [goals]]>
        """
        combined_maps = {
            "explored_space":np.zeros(self._map_dim),
            "obstacles":np.zeros(self._map_dim),
            "robot_positions":np.zeros(self._map_dim),
            "goal_candidates":np.zeros(self._map_dim),
            "last_goals":np.zeros(self._map_dim),
            "current_goals":np.zeros(self._map_dim)
        }

        goal_candidates_maps = []
        for agent in self._agents:
            agent_maps = agent.maps
            combined_maps["explored_space"] |= agent_maps["explored_space"]
            combined_maps["obstacles"] |= agent_maps["obstacles"]
            combined_maps["robot_positions"][agent.pos] = 1
            combined_maps["goal_candidates"] |= combined_maps["goal_candidates"]
            goal_candidates_maps.append(combined_maps["goal_candidates"])
            combined_maps["last_goals"] |= agent_maps["last_goals"]
            combined_maps["current_goals"] |= agent_maps["current_goals"]

        percent_explored = np.count_nonzero(combined_maps["explored_space"]) / combined_maps["explored_space"].size

        conv_maps = [
            combined_maps["explored_space"],
            combined_maps["obstacles"],
            combined_maps["robot_positions"],
            combined_maps["goal_candidates"]
        ]

        return {
            "x":combined_maps["robot_positions"].flatten(),
            "delta":[self.teammates_in_range],
            "beta_last_g":combined_maps["last_goals"].flatten(),
            "beta_this_g":combined_maps["current_goals"].flatten(),
            "m":np.stack(conv_maps, axis=2),
            "E":[percent_explored],
            "g":goal_candidates_maps.flatten()
        }

    # TODO: Check macro actions work
    def append_to_mem(self, observation, action, reward, terminal):
        self.memory.append(observation, action, reward, terminal)