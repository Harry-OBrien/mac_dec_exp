import numpy as np

# class SimpleDDQN(DQNAgent):
#     def __init__(self, **kwargs):
#         super.__init__(**kwargs)

class ActionStateModel:
    def __init__(self, state_dim, action_dim, model, policy, optimizer):
        # , gamma, policy, enable_double_dqn, memory, nb_steps_warmup,
        # batch_size, target_model_update, nb_actions
        
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.policy = policy
        
        self.model = model
        self.compile(optimizer=optimizer, metrics=["accuracy"])
    
    def compile(self, optimizer, metrics=[]):
        self.model.compile(loss='mse', optimizer=optimizer, metrics=metrics)
    
    def predict(self, state):
        return self.model.predict(state)
    
    def get_action(self, state):

        # state = {
        #     "x":pos_arr,
        #     "delta":self.teammate_detector.teammate_in_range(),
        #     "beta_last_g":teammate_last_goals,
        #     "beta_x":teammate_positions,
        #     "beta_this_g":teammate_current_goals,
        #     "m":agent_maps,
        #     "E":percent_explored,
        #     "g":goals
        # }

        map_input = state.pop("m")
        fc_input = np.concatenate(list(state.values()))
        q_value = self.predict(map_input, fc_input)

        # Q-Boltzmann policy
        action = self.policy.select_action(q_value)

        return action

    def train(self, states, targets, batch_size=1):
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size, shuffle=False)
    
