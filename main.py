from agents.mac_dec_ddqn import Mac_Dec_DDQN_Agent
from env.multi_agent_grid import make_env
# from keras.callbacks import History
# from rl.callbacks import CallbackList
import numpy as np

# TO CONTINUE:
#   About to implement communication between teammates to share info
#   both client and env side not yet started
#   look at get_action method of agent for naming

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

            action = None if done else agent.get_action(state)
            env.step(action)

            agent.append_to_mem(state, action, reward, done)

            episode_rewards[agent_id] += reward
            episode_rewards["global"] += reward

            if visualise and i % len(agents) == 0:
                env.render()

        for agent_id in env.agents:
            agent = agents[agent_id]
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
        "n_agents":1,
        "seed":0,
        "clutter_density":0.0,
        "max_steps":500,
        "pad_output":False,
        "agent_view_shape":(9, 9),
        "view_offset":4
    }

    # Create env
    env, unwrapped = make_env(**env_config)
    env.reset()
    
    # Initialise agents
    agents = {}
    for agent in env.agents:
        observation_space = env.observation_space(agent).shape
        n_actions = env.action_space(agent).n

        agents[agent] = Mac_Dec_DDQN_Agent(
            n_actions=n_actions, 
            observation_dim=observation_space, 
            map_dim=env_config["map_shape"],
            env=unwrapped,
            id=agent)

        unwrapped.register_communication_callback(agent, agents[agent].get_callbacks())

    fit(env, agents, nb_episodes=training_config["n_episodes"], visualise=True)
    env.close()

if __name__ == "__main__":
    main()