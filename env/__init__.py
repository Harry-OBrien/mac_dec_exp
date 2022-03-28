from .multi_agent_grid import raw_env
from pettingzoo.utils import wrappers

def make_env(map_shape, n_agents, **kwargs):
    '''
    The make_env function often wraps the environment in wrappers by default.
    '''
    env = raw_env(map_shape, n_agents, **kwargs)
    
    # this wrapper helps error handling for discrete action spaces
    # env = wrappers.AssertOutOfBoundsWrapper(env)

    return env