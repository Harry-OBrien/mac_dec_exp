# from agents.mac_dec_ddqn import Mac_Dec_DDQN_Agent
from agents.nearest_frontier import NearestFrontierAgent
# import env as multi_agent_grid
from env.actions import Action
from env.multi_agent_grid import parallel_env

def all_complete(done_dict):
    for complete in done_dict.values():
        if not complete:
            return False

    return True

def fit(env, agents, nb_episodes, visualise=False):

    from pettingzoo.utils import average_total_reward
    # average_total_reward(env, max_episodes=10, max_steps=75)


    # Start training
    for ep_idx in range(nb_episodes):            
            
        states = env.reset()
        env.render()

        print("Starting episode", ep_idx + 1)
        dones = {agent: False for agent in agents.keys()}
        
        while not all_complete(dones):
            actions = {agent_id: agent.get_action(states[agent_id]) for agent_id, agent in agents.items()}
            states, rewards, dones, _ = env.step(actions)

            for agent_id, agent in agents.items():
                agent.append_to_mem(states[agent_id], actions[agent_id], rewards[agent_id], dones[agent_id])

            env.render()

        # allow the agents to train
        for agent in agents.values():
            agent.replay()
            agent.target_update()
            agent.reset_observations()

        # Finally, log all the data from the episode
        env.unwrapped.capture_episode(ep_idx, ep_idx == nb_episodes - 1)

def main():
    training_config = {
        "n_episodes":10
    }

    env_config = {
        "map_shape":(20, 20),
        "n_agents":3,
        "seed":0,
        # "clutter_density":0.8,
        "max_steps":75,
        "pad_output":False,
        "agent_view_shape":(9, 9),
        "screen_size":500,
        "logfile_dir":"./agent_training_history.json"
    }

    # Create env
    env = parallel_env(**env_config)
    
    # Initialise agents
    agents = {}
    for agent in env.possible_agents:
        observation_space = env.observation_space(agent).shape
        n_actions = env.action_space(agent).n

        agents[agent] = NearestFrontierAgent(
            n_actions=n_actions, 
            observation_dim=observation_space, 
            map_dim=env_config["map_shape"],
            env=env,
            id=agent)

        env.unwrapped.register_communication_callback(agent, agents[agent].get_callbacks())

    fit(env, agents, nb_episodes=training_config["n_episodes"], visualise=True)
    env.close()

if __name__ == "__main__":
    main()