from .base import Base_Agent
# import numpy as np

class Random_Agent(Base_Agent):
    def __init__(self, n_actions, state_space):
        super().__init__(n_actions, state_space)
        
    def get_action(self, _):
        return 0
        # return np.random.choice(self._n_actions)