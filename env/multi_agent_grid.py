import numpy as np
from gym import spaces
from gym.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector
import functools

def rotate_grid(grid, rot_k):
    '''
    This function basically replicates np.rot90 (with the correct args for rotating images).
    But it's faster.
    '''
    rot_k = rot_k % 4
    if rot_k==3:
        return np.moveaxis(grid[:,::-1], 0, 1)
    elif rot_k==1:
        return np.moveaxis(grid[::-1,:], 0, 1)
    elif rot_k==2:
        return grid[::-1,::-1]
    else:
        return grid

def pad_array(array, pad_value, top=0, right=0, bottom=0, left=0):
    padded = array.copy()
    h, w = padded.shape

    if right > 0:
        pad_r = [[pad_value] * h for _ in range(right)]
        padded = np.insert(padded, w, pad_r, axis=1)

    if left > 0:
        pad_l = [[pad_value] * h for _ in range(left)]
        padded = np.insert(padded, 0, pad_l, axis=1)

    h, w = padded.shape
    if bottom > 0:
        pad_b = [[pad_value] * w for _ in range(bottom)]
        padded = np.insert(padded, h, pad_b, axis=0)

    if top > 0:
        pad_t = [[pad_value] * w for _ in range(top)]
        padded = np.insert(padded, 0, pad_t, axis=0)

    return padded

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def make_env(map_shape, n_agents, **kwargs):
    '''
    The make_env function often wraps the environment in wrappers by default.
    '''
    env = raw_env(map_shape, n_agents, **kwargs)

    unwrapped = env
    
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)

    return env, unwrapped

class raw_env(AECEnv):
    metadata = {'render_modes': ['human'], "name": "multi_agent_grid_world_v1"}

    def __init__(self, map_shape, n_agents, clutter_density=None, movement_failure_prob=0.0, communication_dropout_prob=0.0, 
                 step_penalty=-1, decay_reward=False, max_steps=250, agent_view_shape=(5,5), view_offset=0, seed=None, transparent_walls=False,
                 overlay_maps=True, pad_output=True):
        """
        The init method takes in environment arguments and
        should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        """
        self._map_shape = map_shape
        self._n_agents = n_agents
        self._clutter_density = clutter_density
        self._movement_failure_prob = movement_failure_prob
        self._communication_dropout_prob = communication_dropout_prob
        self._step_penalty = step_penalty
        self._decay_reward = decay_reward
        self._max_steps = max_steps
        self._agent_view_shape = agent_view_shape
        self._view_offset = view_offset
        self._transparent_walls = transparent_walls
        self._overlay_maps = overlay_maps
        self._pad_output = pad_output

        # rendering shite
        self._rendering_grid = [[None] * self._map_shape[1] for _ in range(self._map_shape[0])]
        self._viewer = None

        # seed env (if none, a random seed is used)
        self.seed(seed)

        assert self._n_agents >= 0
        assert self._agent_view_shape[0] >= 0 and self._agent_view_shape[1] >= 0

        w, h = self._map_shape

        self._map_size = w * h
        assert w > 0 and h > 0 and self._map_size >= n_agents
        assert self._agent_view_shape[0] * self._agent_view_shape[1] <= self._map_size

        # agent naming and colouring
        self.possible_agents = ["agent_" + str(r) for r in range(self._n_agents)]

        self._agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        colours = [
            (255, 0, 0),        # Red
            (0, 255, 0),        # Green
            (0, 0, 255),        # Blue
            (255, 255, 0),      # Yellow
            (255, 0, 255),      # Purple
            (0, 255, 255)]      # Turqoise
        
        assert self._n_agents <= len(colours)
        self._agent_colours = {agent: colours[i] for i, agent in enumerate(self.possible_agents)}

        self._agent_locations = {}

        # environment spaces
        # (up, right, down, left, nothing) -> 5
        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.possible_agents}

        # 3 binary feature maps:
        #   explored space (by this agent)
        #   obstacles
        #   observed robot positions
        self.observation_spaces = {agent: spaces.Box(low=0, high=1, shape=(3, self._agent_view_shape[0], self._agent_view_shape[1]))
                                        for agent in self.possible_agents}

        self.observation_spaces["global"] = spaces.Box(low=0, high=1, shape=(3, *self._map_shape))

    @property
    def agent_name_mapping(self, agent_id):
        return self._agent_name_mapping[agent_id]

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        '''
        Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        '''
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        '''
        Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the action_spaces dict
        '''
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def observe(self, agent):
        '''
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        '''
        return self.observations[agent]

    def state(self):
        '''
        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX
        '''
        return self._maps

    def step(self, action):
        '''
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        '''
        agent = self.agent_selection
        # print(agent, action)

        if self.dones[agent]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            return self._was_done_step(action)

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # perform action on environment
        self._take_action(action)

        # Update observation
        raw_observation, bounds, offsets, occlusion_mask = self._get_observation(agent)

        if self._pad_output:
            self.observations[agent] = {
                "obstacles":pad_array(raw_observation["obstacles"], 1, *offsets),
                "robot_positions":pad_array(raw_observation["robot_positions"], 0, *offsets),
                "explored_space":pad_array(occlusion_mask, 0, *offsets),
                "bounds":bounds,
                "pos":self._agent_locations[agent],
            }
        else:
            # Unpadded version
            self.observations[agent] = {
                "obstacles":raw_observation["obstacles"],
                "robot_positions":raw_observation["robot_positions"],
                "explored_space":occlusion_mask,
                "bounds":bounds,
                "pos":self._agent_locations[agent],
            }

        # update explored area (not applied until all agents have moved)
        ty, by, tx, bx = bounds
        self._explored_this_round[ty:by, tx:bx] |= occlusion_mask

        # Calculate rewards
        self._calculate_rewards(raw_observation, occlusion_mask)

        # once everyone has moved, update our global map of locations
        if self._agent_selector.is_last():
            # step complete
            self.num_moves += 1
            if self.num_moves >= self._max_steps or self._complete():
                self.dones = dict.fromkeys(self.dones, True)

            # update maps
            self._regenerate_agent_map()
            self._maps["explored_space"] |= self._explored_this_round
            self._explored_this_round = np.zeros(self._map_shape, dtype=int)
        # else:
        #     self._clear_rewards()

        # select the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

    # MARK: Action
    def _take_action(self, action):
        agent = self.agent_selection
        # print(agent, action)

        # Slight chance of robot not doing as expected
        if self._np_random.uniform() < self._movement_failure_prob: 
            action = self._random_move(agent)

        current_location = self._agent_locations[agent]

        next_pos = None
        try:
        # up, right, down, left, nothing
            offsets = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
            next_pos = (
                current_location[0] + offsets[action][0], # row ( y )
                current_location[1] + offsets[action][1]  # col ( x )
            )
        except IndexError:
            return

        if self._point_out_of_bounds(next_pos):
            return

        # if not blocked, move agent
        if not self._maps["obstacles"][next_pos] and\
            not self._maps["robot_positions"][next_pos]:
            self._agent_locations[agent] = next_pos

    def _random_move(self, agent):
        # We do -1 here because the 
        return self._np_random.choice(self.action_space(agent).n - 1)

    def _point_out_of_bounds(self, pos):
        if pos[0] < 0 or pos[0] >= self._map_shape[0]\
            or pos[1] < 0 or pos[1] >= self._map_shape[1]:
            return True

        return False

    # MARK: Observation
    def _get_observation(self, agent):
        """
        Gets the observation of the current agent

        returns 
            the observation (not always the dim shape as agent_view_shape), 
            the occlusion mask (what the agent can actually see) 
            and the bounds defining where in the map the observation is looking at
        """
        # Find bounds of agent in map
        pos = self._agent_locations[agent]
        view_bounds = self._get_view_exts(pos)
        corrected_bounds, offsets = self._correct_bounds(*view_bounds, self._map_shape)

        # get the observations
        observation = self._map_slice(*corrected_bounds)
        pos_in_obs = self._pos_in_bounds(corrected_bounds, offsets)

        # Apply occlusion mask to remove anything that the agent cant see
        mask = np.ones_like(observation, dtype=bool)
        if not self._transparent_walls:
            mask = self._occlude_mask(~observation["obstacles"].astype(bool), pos_in_obs)

        for key in observation.keys():
            observation[key] &= mask

        return observation, corrected_bounds, offsets, mask
                
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
        h, w = self._map_shape

        assert botY >= 0 and topY < h
        assert botX >= 0 and topX < w

        slices = {}
        for key, full_map in self._maps.items():
            slices[key] = full_map.copy()[topY:botY, topX:botX]

        return slices

    def _pos_in_bounds(self, corrected_bounds, offsets):
        h, w = (
            corrected_bounds[1] - corrected_bounds[0],
            corrected_bounds[3] - corrected_bounds[2])
        ty_off, by_off, tx_off, bx_off = offsets
        
        y = clamp(h//2 - ty_off + by_off, 0, h-1)
        x = clamp(w//2 - tx_off + bx_off, 0, w-1)
            
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

    # MARK: Reward 
    def _calculate_rewards(self, raw_observation, occlusion_mask):
        agent = self.agent_selection
        this_agent_id = self._agent_name_mapping[agent]

        self.rewards[agent] = 0
        self.rewards["global"] += self._step_penalty

        for i, row in enumerate(raw_observation["explored_space"]):
            for j, explored in enumerate(row):
                if occlusion_mask[i, j] and explored:
                    self.rewards[agent] -= 1
                    self.rewards["global"] -= 1
                elif occlusion_mask[i, j] and not explored:
                    self.rewards[agent] += 1
                    self.rewards["global"] += 1

        teammate_found = False
        for row in raw_observation["robot_positions"] & occlusion_mask:
            for teammate in row:
                if teammate and teammate != this_agent_id:
                    self._teammate_in_sight_count[agent] += 1
                    teammate_found = True
                    break
            if teammate_found:
                break
        
        if not teammate_found:
            self._teammate_in_sight_count[agent] = 0

        sight_count = self._teammate_in_sight_count[agent]
        if sight_count >= 2 and sight_count <= 7:
            self.rewards[agent] -= np.sqrt(np.exp(sight_count)/5)
        elif sight_count > 7:
            self.rewards[agent] -= 15

        if self._complete():
            self.rewards["global"] += 100

    def _complete(self):
        return np.count_nonzero(self._maps["explored_space"]) == self._map_size

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
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the current states dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        '''
        print("Resetting env!")
        self.agents = self.possible_agents[:]

        # Maps
        # If we haven't set an explicit value for clutter density, randomly choose one between 30% and 70%
        clutter_density = self._clutter_density if self._clutter_density is not None else self._np_random.uniform(0.3, 0.7)
        self._maps = {
            "obstacles":np.array([[1 if self._np_random.uniform() < clutter_density else 0 for _ in range(self._map_shape[0])] for _ in range(self._map_shape[1])], dtype=int),
            "explored_space":np.zeros(self._map_shape, dtype=int),
            "robot_positions":None
        }

        self._explored_this_round = np.zeros(self._map_shape, dtype=int)

        # agent locations        
        for agent in self.agents:
            init_pos = self._choose_starting_pos_for(agent)
            self._agent_locations[agent] = init_pos

        self._regenerate_agent_map()

        self._teammate_in_sight_count = {agent: 0 for agent in self.agents}

        self.rewards = {agent: 0 for agent in self.agents}
        self.rewards["global"] = 0
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards["global"] = 0
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.current_states = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        self.comms_callbacks = {agent: None for agent in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        # generate an initial observation
        for agent in self.agents:
            raw_observation, bounds, offsets, occlusion_mask = self._get_observation(agent)
            if self._pad_output:
                self.observations[agent] = {
                    "obstacles":pad_array(raw_observation["obstacles"], 1, *offsets),
                    "robot_positions":pad_array(raw_observation["robot_positions"], 0, *offsets),
                    "explored_space":pad_array(occlusion_mask, 0, *offsets),
                    "bounds":bounds,
                    "pos":self._agent_locations[agent],
                }
            else:
                # Unpadded version
                self.observations[agent] = {
                    "obstacles":raw_observation["obstacles"],
                    "robot_positions":raw_observation["robot_positions"],
                    "explored_space":occlusion_mask,
                    "bounds":bounds,
                    "pos":self._agent_locations[agent],
                }

            # update explored area (not applied until all agents have moved)
            ty, by, tx, bx = bounds
            self._explored_this_round[ty:by, tx:bx] |= occlusion_mask

    def _choose_starting_pos_for(self, agent):
        # Keep trying to select a location for the agent to start
        while True:
            pos = (
                self._np_random.randint(self._map_shape[0] - 1), 
                self._np_random.randint(self._map_shape[1] - 1)
            )
            occupied = self._maps["obstacles"][pos]
            # If unoccupied, we can continue
            if not occupied:
                break
            else:
                print("failed to place", agent, "in initial pos", pos, ". retrying...")

        print("agent placed at", pos)
        return pos

    def _regenerate_agent_map(self):
        agent_map = np.zeros(shape=self._map_shape, dtype=int)
        for i, agent in enumerate(self.agents):
            pos = self._agent_locations[agent]
            agent_map[pos] = i + 1

        self._maps["robot_positions"] = agent_map

    # MARK: Render
    def render(self, mode="human"):
        print(self._agent_locations["agent_0"])
        SCREEN_SIZE = 500
        square_dimension = 0.0

        if self._viewer is None:
            from gym.envs.classic_control import rendering

            square_dimension = SCREEN_SIZE / self._map_shape[0]
            self._viewer = rendering.Viewer(SCREEN_SIZE, SCREEN_SIZE)

            for i in range(self._map_shape[0]):
                for j in range(self._map_shape[1]):
                    l, r, t, b = (
                        j * square_dimension,
                        (j + 1) * square_dimension,
                        i * square_dimension,
                        (i + 1) * square_dimension,
                    )
                    square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    border = rendering.PolyLine([(l, b), (l, t), (r, t), (r, b)], True)

                    self._rendering_grid[i][j] = square
                    self._viewer.add_geom(square)
                    self._viewer.add_geom(border)

        for i in range(self._map_shape[0]):
            for j in range(self._map_shape[1]):
                square = self._rendering_grid[self._map_shape[0] - 1 - i][j]

                # If a robot exists in this square
                if (self._maps["robot_positions"][i, j] != 0):
                    #robot's pos
                    agent = "agent_" + str(self._maps["robot_positions"][i, j] - 1)
                    try:
                        square.set_color(*self._agent_colours[agent])
                    except KeyError:
                        print(agent, "don't exists broski")

                # if unexplored
                elif not self._maps["explored_space"][i, j]:
                    if (not self._maps["obstacles"][i, j]):
                        square.set_color(0.8, 0.8, 0.8)
                    else:
                        square.set_color(0.3, 0.3, 0.3)

                # Square is explored and blocked
                elif (self._maps["obstacles"][i, j]):
                    square.set_color(0, 0, 0)

                # square is explored and empty
                else:
                    square.set_color(1, 1, 1)

        return self._viewer.render(return_rgb_array=(mode == "rgb_array"))

    # MARK: Communication
    def register_communication_callback(self, agent_id, callback):
        self.comms_callbacks[agent_id] = callback

    def _agent_in_sight(self, num_tx_id, num_rx_id):
        tx_agent = self.agents[num_tx_id]
        agent_observation = self.observations[tx_agent]["robot_positions"]

        for row in agent_observation:
            for id in row:
                if id == num_rx_id:
                    return True
        
        return False

    def communicate_with_agent(self, num_tx_id, num_rx_id):
        return False
        """
        Attempts to handle the sharding of two agents that are within sight

        # Returns
            true on success, false otherwise
        """
        try:
            tx_agent = self.agents[num_tx_id]
            rx_agent = self.agents[num_rx_id]
        except IndexError:
            print("WARN: either tx or rx id is invalid during communication")
            return False
            
        # Check if tx agent can even see rx agent
        if not self._agent_in_sight(num_tx_id, num_rx_id):
            return False

        # Slight chance of communication failing
        if self._np_random.random() < self._communication_dropout_prob:
            return False
            
        # callbacks transmit (tx) and reciece (rx)
        target_cb_tx, target_cb_rx = self.comms_callbacks[rx_agent]
        sender_cb_tx, sender_cb_rx = self.comms_callbacks[tx_agent]

        if target_cb_tx is not None and sender_cb_tx is not None:
            target_cb_tx(sender_cb_tx())
        
        if sender_cb_rx is not None and target_cb_rx is not None:
            sender_cb_rx(target_cb_rx())

        return True

    # MARK: Close
    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        pass