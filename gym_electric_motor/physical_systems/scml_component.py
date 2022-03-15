import gym
import numpy as np


class SCMLComponent:

    _empty_array = np.array([])

    @property
    def observation_names(self):
      return ()
    
    @property
    def observation_units(self):
      return ()

    @property
    def observation_tex_names(self):
      return self.observation_names
    
    @property
    def observation_tex_units(self):
      return self.observation_units

    @property
    def limits(self):
      return self._empty_array

    @property
    def nominal_state(self):
      return self._empty_array
    
    @property
    def observation_space(self):
      return gym.spaces.Box(0,0, shape=(0,), dtype=np.float64)

    def get_observation(self, state=()):
        return self._empty_array
