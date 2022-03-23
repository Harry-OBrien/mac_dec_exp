class Base_Agent():
    def __init__(self):
        pass
    
    def get_action(self, observation):
        raise NotImplementedError()

    def train(self, observation, reward, done):
        pass