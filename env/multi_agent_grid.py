import functools
from gym import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
from gym.utils import seeding
import numpy as np

from .world import MultiAgentWorld
from .rendering import Renderer
from .episode_logger import EpisodeLogger
from .actions import Action
from .pad_array import pad_array

def raw_env(map_shape, n_agents, **kwargs):
    env = parallel_env(map_shape, n_agents, **kwargs)
    # env = parallel_to_aec(env)

    return env

class parallel_env(ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "multi_agent_grid_world_v1.5"}

    def __init__(self, map_shape, n_agents, clutter_density=None, global_view=False, movement_failure_prob=0.1, partial_observation_prob=0.1, communication_dropout_prob=0.0, 
                 max_steps=75, agent_view_shape=(5,5), seed=None, transparent_walls=False, pad_output=True, screen_size=500, logfile_dir="./history_log.json"):
        '''
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        self._global_view = global_view
        self._movement_failure_prob = movement_failure_prob
        self._communication_dropout_prob = communication_dropout_prob
        self._max_steps = max_steps
        self._agent_view_shape = agent_view_shape
        self._pad_output = pad_output

        self.possible_agents = ["agent_" + str(r) for r in range(n_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(1, len(self.possible_agents) + 1))))

        self.seed(seed)

        self._world = MultiAgentWorld(
            map_dim=map_shape,
            density=clutter_density,
            transparent_walls=transparent_walls,
            agents=self.possible_agents,
            partial_observation_prob=partial_observation_prob,
            agent_view_shape=agent_view_shape,
            np_random=self._np_random)

        self._renderer = Renderer(self._world, screen_size)
        self._environment_log = EpisodeLogger(self._world, logfile_dir)
        
        self.comms_callbacks = {agent: None for agent in self.possible_agents}
        self.communications_this_step = {agent:[] for agent in self.possible_agents}

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent=None):
         # 3 binary feature maps:
        #   explored space (by this agent)
        #   obstacles
        #   observed robot positions
        return spaces.Box(low=0, high=1, shape=(3, self._agent_view_shape[0], self._agent_view_shape[1]))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent=None):
        # (up, right, down, left, nothing) -> 5
        return spaces.Discrete(5)

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        '''
        self._renderer.render()

    def capture_episode(self, *args):
        self._environment_log.capture_episode(*args)

    # MARK: Reset
    def reset(self):
        '''
        Resets the environment to a starting state.

        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection

        and must set up the environment so that render(), and step() can be 
        called without issues.

        Returns the observations for each agent
        '''
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.num_steps = 0

        self._teammate_in_sight_count = {agent:0 for agent in self.agents}
        self.communications_this_step = {agent:[] for agent in self.agents}
        self._local_interaction_count = 0

        self._world.reset()
        self._environment_log.episode_reset()

        observations = {}
        self._last_masks = {}

        if self._global_view:
            observations = self._world.maps
        else:
            for agent in self.agents:
                initial_observation, _, _, _, initial_mask = self._observation_for_agent(agent)
                observations[agent] = initial_observation
                self._last_masks[agent] = initial_mask

        return observations

    # MARK: Step
    def step(self, actions):
        '''
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        '''
        # If a user passes in actions with no agents, then just return empty observations, etc.
        assert actions is not None

        # Perform actions for agents
        distance_travelled = 0
        actions_taken = {}
        for agent, action in actions.items():
            actions_taken[agent] = self._perform_action(agent, action)
            if actions_taken[agent] != Action.NO_MOVEMENT:
                distance_travelled += 1

        # current observation is just the other player's most recent action
        # if self._global_view:
        #     observations = self._world.maps
        # else:
        #     observations = {agent: self._observation_for_agent(agent) for agent in self.agents}
        observations = {}
        rewards = {}
        for agent in self.agents:
            agent_observation, raw_observation, _, _, this_mask = self._observation_for_agent(agent)
            observations[agent] = agent_observation

            reward = self._calculate_reward(agent,
                                            this_mask,
                                            self._last_masks[agent],
                                            raw_observation,
                                            actions_taken[agent])

            rewards[agent] = reward
            self._last_masks[agent] = this_mask

        if self._global_view:
            observations = self._world.maps

            rewards = sum(rewards.values())
            rewards -= 1
            if self._world.search_fully_complete():
                rewards += 100

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        dones = {}
        env_done = self.num_steps >= self._max_steps or self._world.search_fully_complete()
        for agent in self.agents:
            try:
                if dones[agent]:
                    continue
            except KeyError:
                pass
            
            dones[agent] =  env_done or self._world.agent_complete(agent)

        if env_done or self._all_complete(dones):
            self.agents = []

        # logging etc
        self._environment_log.capture_step(
            reward=sum(list(rewards.values())), 
            explored_count=self._world.nb_cells_explored,
            map_size=self._world.map_size,
            distance_travelled=distance_travelled,
            local_interactions=self._local_interaction_count)
        self.num_steps += 1

        return observations, rewards, dones, infos

    def _all_complete(self, dones):
        for done in dones.values():
            if not done:
                return False

        return True

    # MARK: Action
    def _perform_action(self, agent, action):
        if action == Action.NO_MOVEMENT:
            return action

        # Slight chance that we'll move to a random cell instead of the desired
        if self._np_random.uniform() < self._movement_failure_prob:
            action = self._random_action()

        movement = [(-1, 0), (0, 1), (1, 0), (0, -1)][action.value]
        _ = self._world.move(agent, movement)

        return action

    def _random_action(self):
        # We do -1 here because we don't want to include NO_MOVEMENT
        action = self._np_random.choice(self.action_space().n - 1)
        return Action(action)

    # MARK: Observation
    def _observation_for_agent(self, agent):
        raw_observation, bounds, offsets, occlusion_mask = self._world.observation_for_agent(agent)

        if self._pad_output:
            agent_observation = {
                "obstacles":pad_array(raw_observation["obstacles"], 1, *offsets),
                "robot_positions":pad_array(raw_observation["robot_positions"], 0, *offsets),
                "explored_space":pad_array(occlusion_mask, 0, *offsets),
                "bounds":bounds,
                "pos":self._world.agent_location(agent),
            }
        else:
            agent_observation = {
                "obstacles":raw_observation["obstacles"],
                "robot_positions":raw_observation["robot_positions"],
                "explored_space":occlusion_mask,
                "bounds":bounds,
                "pos":self._world.agent_location(agent),
            }

        return agent_observation, raw_observation, bounds, offsets, occlusion_mask

    # MARK: Reward
    def _calculate_reward(self, agent, this_mask, last_mask, raw_observation, action):
        reward = 0
        pos_expl_reward, neg_expl_reward, new_cell_mask = self._count_new_cells(this_mask, 
                                                                    last_mask, 
                                                                    raw_observation["explored_space"],
                                                                    action)

        # print(pos_expl_reward, neg_expl_reward, new_cell_mask.astype(int), sep="\n")

        reward += pos_expl_reward
        reward -= neg_expl_reward

        if self._teammate_in_observation(self.agent_name_mapping[agent], raw_observation["robot_positions"]):
            self._teammate_in_sight_count[agent] += 1
        else:
            self._teammate_in_sight_count[agent] = 0            

        if not self._global_view:
            sight_count =  self._teammate_in_sight_count[agent]
            if sight_count >= 2 and sight_count <= 7:
                reward -= np.sqrt(np.exp(sight_count)/5)
            elif sight_count > 7:
                reward -= 15

        return reward

    def _count_new_cells(self, this_mask, last_mask, observation, action):
        axis = 0 if (action == Action.UP or action == Action.DOWN) else 1

        mask_shape_diff = this_mask.shape[axis] - last_mask.shape[axis]
        offset = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)][action.value][axis]

        shifted_mask = self._generate_shifted_mask(last_mask, mask_shape_diff, offset, axis)
        new_cell_mask = ~shifted_mask & this_mask

        # Calculate reward values
        new_cells = np.count_nonzero(new_cell_mask)
        prev_observed_cells = np.count_nonzero(observation & new_cell_mask)
    
        positive_reward, negative_reward = new_cells - prev_observed_cells, prev_observed_cells
        return positive_reward, negative_reward, new_cell_mask

    def _generate_shifted_mask(self, last_mask, size_diff, offset, axis):
        assert offset >= -1 and offset <= 1
        
        shifted_slice = last_mask
        
        if offset == -1:  
            if size_diff <= 0:
                bot_y = -1 if axis == 0 else None
                bot_x = -1 if axis == 1 else None
                shifted_slice = shifted_slice[:bot_y, :bot_x]
                
            if size_diff >= 0:
                shifted_slice = np.insert(shifted_slice, 0, 0, axis=axis)

        elif offset == 1:
            if size_diff <= 0:
                top_y = 1 if axis == 0 else None
                top_x = 1 if axis == 1 else None
                shifted_slice = shifted_slice[top_y:, top_x:]
                
            if size_diff >= 0:
                shifted_slice = np.insert(shifted_slice, shifted_slice.shape[axis], 0, axis=axis)
                
        return shifted_slice

    def _teammate_in_observation(self, this_agent_id, robot_positions_observation):
        for row in robot_positions_observation:
            for teammate in row:
                if teammate and teammate != this_agent_id:
                    return True

        return False

    # MARK: Communication
    def register_communication_callback(self, agent_id, callback):
        self.comms_callbacks[agent_id] = callback

    def _agent_in_sight(self, num_tx_id, num_rx_id):
        tx_agent = self.agents[num_tx_id]
        agent_observation = self.observations[tx_agent]["robot_positions"]

        mapped_id = num_rx_id + 1
        for row in agent_observation:
            for id in row:
                if id == mapped_id:
                    return True
        
        return False

    def communicate_with_agent(self, num_tx_id, num_rx_id):
        """
        Attempts to handle the sharding of two agents that are within sight

        # Returns
            true on success, false otherwise
        """
        if num_tx_id == num_rx_id:
            print("WARN: Agent is trying to communicate with itself")
            return False

        try:
            tx_agent = self.agents[num_tx_id]
            rx_agent = self.agents[num_rx_id]
        except IndexError:
            print("WARN: either tx or rx id is invalid during communication")
            return False
            
        # Check if tx agent can even see rx agent
        #TODO: Implement
        # if not self._agent_in_sight(num_tx_id, num_rx_id):
        #     print("WARN: Agent thought it could see an agent which it cannot.")
        #     return False

        # check that we haven't tried to communicate before
        if rx_agent not in self.communications_this_step[tx_agent]:
            self.communications_this_step[tx_agent].append(rx_agent)
            self.communications_this_step[rx_agent].append(tx_agent)
            self._local_interaction_count += 1

        # Slight chance of communication failing for a duration of 7 time steps
        if self._teammate_in_sight_count[tx_agent] <= 7:
            if self._np_random.random() < self._communication_dropout_prob:
                return False

        self._world.combine_maps(tx_agent, rx_agent)
            
        # callbacks transmit (tx) and reciece (rx)
        if self.comms_callbacks[rx_agent] is None or self.comms_callbacks[tx_agent] is None:
            print("WARN: Either", tx_agent, "or", rx_agent, "hasn't registered a callback.")
            return False

        target_cb_tx, target_cb_rx = self.comms_callbacks[rx_agent]
        sender_cb_tx, sender_cb_rx = self.comms_callbacks[tx_agent]

        target_cb_rx(sender_cb_tx())
        sender_cb_rx(target_cb_tx())

        return True

    # MARK: Close
    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        self._environment_log.close()
        self._renderer.close()
