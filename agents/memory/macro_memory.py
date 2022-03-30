from rl.memory import SequentialMemory, deque

class ResetableMemory(SequentialMemory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self):
        self.actions = deque(maxlen=self.limit)
        self.rewards = deque(maxlen=self.limit)
        self.terminals = deque(maxlen=self.limit)
        self.observations = deque(maxlen=self.limit)

    def calculate_reward_sum(self):
        sum = 0
        for value in self.rewards:
            sum += value

        return sum

    def terminal(self):
        return self.terminals[-1]

class SequentialMacroMemory(SequentialMemory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.real_time_mem = ResetableMemory(**kwargs)

    def append(self, *args, **kwargs):
        if self.last_observation is None or self.last_action is None:
            # print("WARN: Attempted to append temporal action without any associated macro view and/or observation")
            return

        self.real_time_mem.append(*args, **kwargs)

        # Future To Do: This is janky af, maybe don't do this???
        _, _, _, done = args
        if done:
            self.append_macro_from_real_time()

    def macro_action_begin(self, observation, action):
        self.last_observation = observation
        self.last_action = action

        # just in case, flush the RT memory
        self.real_time_mem.reset()

    def append_macro_from_real_time(self):
        if self.last_observation is None or self.last_action is None:
            # print("WARN: Attempted to append macro action without first calling 'macro_action_start'")
            return

        macro_reward = self.real_time_mem.calculate_reward_sum()
        terminal = self.real_time_mem.terminal()
        super().append(observation=self.last_observation, action=self.last_action, reward=macro_reward, terminal=terminal)

        self.real_time_mem.reset()
        self.last_observation = None
        self.last_action = None

    def _clear_real_time_memory(self):
        self.real_time_mem.reset()
