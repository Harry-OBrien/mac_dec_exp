class Base_Agent():
    def reset_observations(self):
        pass
    
    def get_action(self, observation):
        raise NotImplementedError()

    def train(self, observation, reward, done):
        pass

    def target_update(self):
        pass

    def replay(self):
        pass

    def append_to_mem(self):
        pass