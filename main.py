# from agents.mac_dec_ddqn import Mac_Dec_DDQN_Agent
from agents.nearest_frontier import NearestFrontierAgent
from env.multi_agent_grid import make_env
# from keras.callbacks import History
# from rl.callbacks import CallbackList
# import numpy as np

def fit(env, agents, nb_episodes, callbacks=[], visualise=False):

    rewards={agent: [] for agent in env.agents}

    # Start training
    for _ in range(nb_episodes):
        env.reset()
       
        episode_rewards={agent: 0 for agent in env.agents}
        episode_rewards["global"] = 0

        for i, agent_id in enumerate(env.agent_iter()):
            agent = agents[agent_id]
            state, reward, done, _ = env.last()

            # TODO: IF we are done, we should probs let the agent know about this so it can save the macro action to replay mem
            action = None if done else agent.get_action(state)
            env.step(action)

            agent.append_to_mem(state, action, reward, done)

            episode_rewards[agent_id] += reward
            episode_rewards["global"] += reward

            if visualise and i % len(agents) == 0:
                env.render()

        for agent in agents.values():
            agent.replay()
            agent.target_update()
            rewards[agent_id].append(episode_rewards[agent_id])
            agent.reset_observations()

    return rewards

def main():
    training_config = {
        "n_episodes":100
    }

    env_config = {
        "map_shape":(20, 20),
        "n_agents":4,
        "seed":0,
        # "clutter_density":0.3,
        "max_steps":50,
        "pad_output":False,
        "agent_view_shape":(9, 9),
        "view_offset":4,
        "screen_size":500
    }

    # Create env
    env = make_env(**env_config)
    env.reset()
    
    # Initialise agents
    agents = {}
    for agent in env.agents:
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