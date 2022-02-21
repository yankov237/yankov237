import gym
import numpy as np


class SCMLComponent:
    @property
    def observation_names(self):
      return ()

    @property
    def limits(self):
      return ()

    @property
    def nominal_state(self):
      return ()
    
    @property
    def observation_space(self):
      return gym.spaces.Box(0,0, shape=(0,), dtype=np.float64)

    def get_observation(state):
        return ()