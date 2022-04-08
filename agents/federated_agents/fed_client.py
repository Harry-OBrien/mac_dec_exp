from ..core_agent.core_agent import CoreAgent
import numpy as np
from ..memory.macro_memory import SequentialMacroMemory

class FederatedClient():
    def __init__(self, model_fn, mem_size, map_dim, observation_dim, frontier_width, target_update_every):
        self._model = model_fn()
        self._target_model = model_fn()

        self._map_dim = map_dim
        self._observation_dim = observation_dim
        self._frontier_width = frontier_width

        self.memory = SequentialMacroMemory(limit=mem_size, window_length=1)
        self._target_update_count = 0
        self._target_update_every = target_update_every

    def recieve_updated_weights(self, server_weights):
        self._model.set_weights(server_weights)

        self._target_update_count += 1
        if self._target_update_count >= self._target_update_every:
            self._target_update_count = 0
            self._target_model.set_weights(server_weights)

    def client_update(self, n_episodes, env):
        self.core_agents = {agent:CoreAgent(id=agent, env=env, map_dim=self._map_dim, observation_dim=self._observation_dim, frontier_width=self._frontier_width)\
            for agent in env.possible_agents}

        for _ in range(n_episodes):
            self._train(env)

        return self._model.get_weights()

    def _all_done(self, dones):
        for done in dones:
            if not done:
                return False

        return True

    def _train(self, env):
        """performs training (user the server's model) on the clients dataset"""
        states = env.reset()
        dones = {agent: False for agent in env.possible_agents}

        episode_memory = {agent:[] for agent in env.possible_agents}
        while not self._all_done(dones):
            actions = {}
            for agent, state in states.items():
                actions[agent] = self._get_action(self.core_agents[agent], states[agent])

            states, rewards, dones, _ = env.step(actions)

            for agent in env.possible_agents:
                episode_memory[agent].append((states[agent], actions[agent], rewards[agent], dones[agent]))

        # Episode complete, append the memories (ensuring a terminus at the end of each mem
        # try to run the training (if we have enough data)
        for agent_episode_mem in episode_memory.values():
            for step in agent_episode_mem:
                self.memory.append(*step)

            # The last 'done' value should always be true
            assert agent_episode_mem[-1][3] == True

        for agent in env.possible_agents():
            self._replay()
            self.core_agents[agent].reset_observations()


    def _get_action(self, core_agent, observation):
        core_agent.update_local_observation(observation)

        recalculate_action = core_agent.can_see_teammate or core_agent.reached_goal or not core_agent.can_move_to_goal

        if recalculate_action:
            macro_observation = self._macro_observation_for_agent(core_agent)
            if len(core_agent.goal_candidates) > 0:
                macro_observation = [macro_observation.pop("m"), np.concatenate(list(macro_observation.values()))]
                reshaped = [macro_observation[0].reshape(1, *macro_observation[0].shape),
                            macro_observation[1].reshape(1, *macro_observation[1].shape)]

                goal_idx = self.model.select_action(reshaped)
                # goal_idx = 0
                core_agent.current_goal = core_agent.goal_candidates[goal_idx]
                self.memory.macro_action_begin(reshaped, goal_idx)
            else:
                core_agent.current_goal = None

            # print(core_agent.current_goal)

        return core_agent.next_action

    def _macro_observation_for_agent(self, core_agent):
        local_maps = core_agent.maps

        conv_maps = [
            local_maps["explored_space"],
            local_maps["obstacles"],
            local_maps["robot_positions"],
            local_maps["goal_candidates"]
        ]

        location_map = np.zeros(self._map_dim)
        location_map[core_agent.pos] = 1

        return {
            "x":location_map.flatten(),
            "delta":[core_agent.can_see_teammate],
            "beta_last_g":local_maps["teammate_last_goals"].flatten(),
            "beta_this_g":local_maps["teammate_current_goals"].flatten(),
            "m":np.stack(conv_maps, axis=2),
            "E":[core_agent.explored_ratio],
            "g":local_maps["goal_candidates"].flatten()
        }

    def _replay(self, agent):
        # Only start training if we have enough samples saved
        if self.memory.nb_entries < self.min_mem_size:
            return

        minibatch = self.memory.sample(self.batch_size)

        current_states = [experience[0] for experience in minibatch]
        current_qs_list = self.model.predict(current_states, batch_size=self.batch_size)

        next_states = [experience[3] for experience in minibatch]
        future_qs_list = self._target_model.predict(next_states, batch_size=self.batch_size)

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

            