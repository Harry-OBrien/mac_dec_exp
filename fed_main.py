import numpy as np
from agents.federated_agents.fed_server import FederatedServer
from agents.federated_agents.fed_client import FederatedClient
from agents.federated_agents.model import model_fn
from env.multi_agent_grid import parallel_env

MAP_DIM = (20,20,3)
ACTION_SPACE = 4

def broadcast_weights_to(new_weights, clients):
    for client in clients:
        client.recieve_updated_weights(new_weights)

def next(server_weights, clients, env):
    # Broadcast the new server weights to the clients
    broadcast_weights_to(server_weights, clients)

    # Each client computes their updated weights.
    client_weights = []
    for client in clients:
        client_weights.append(client.client_update(env, n_episodes=10))

    # The server averages these updates.
    mean_client_weights = list()
    for weights_list_tuple in zip(*client_weights): 
        mean_client_weights.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        )

    # The server updates its model.
    return server_weights

# MARK: Evaluation
def evaluate(server_state):
    keras_model = model_fn()
    keras_model.set_weights(server_state)

def main():
    env_config = {
        "map_shape":(20, 20),
        "n_agents":3,
        "seed":0,
        # "clutter_density":0.2,
        "max_steps":75,
        "pad_output":False,
        "agent_view_shape":(9, 9),
        "screen_size":500
    }

    env = parallel_env(**env_config)
    NUM_EPOCHS = 25_000
    NUM_CLIENTS = 10
    np.random.seed(0)

    server = FederatedServer(model_fn)
    clients = [FederatedClient(model_fn, 1000, (20, 20), (9, 9), 6, 200)\
        for _ in range(NUM_CLIENTS)]

    server_state = server.server_weights

    # Training
    for _ in range(NUM_EPOCHS):
        server_state = next(server_state, clients, env)
        server.server_update(server_state)

    evaluate(server.server_weights)

if __name__ == "__main__":
    main()

# NOTES:
"""
In the HFRL problem, the environment, state space, and action space can replace the data set, feature space, and label space of basic FL.

Note that the environment Ei is independent of the other environments,


• Step 1: The initialization/join process can be divided into two cases, one is when the agent has no model locally, and the other is when 
    the agent has a model locally. For the first case, the agent can directly download the shared global model from coordinator. For the 
    second case, the agent needs to confirm the model type and parameters with the central coordinator.
• Step 2: Each agent independently observes the state of the environment and determines the private strategy based on the local model. The 
    selected action is evaluated by the next state and received reward. All agents train respective models in state-action-reward-state (SARS) 
    cycles.
• Step 3: Local model parameters are encrypted and transmitted to the coordinator. Agents may submit local models at any time as long as the 
    trigger conditions are met.
• Step 4: The coordinator conducts the specific aggregation algorithm to evolve the global federated model. Actually, there is no need to 
    wait for submissions from all agents, and appropriate aggregation conditions can be formulated depending on communication resources.
• Step 5: The coordinator sends back the aggregated model to the agents.
• Step 6: The agents improve their respective models by fusing the federated model.

"""