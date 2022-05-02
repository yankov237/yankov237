import gym
import numpy as np

from .mechanical_load import MechanicalLoad


class ConstantSpeedLoad(MechanicalLoad):
    """Constant speed mechanical load system which will always set the speed to a fixed value."""

    HAS_JACOBIAN = False

    @property
    def limits(self):
        return self._omega

    @property
    def nominal_state(self):
        return self._omega

    @property
    def observation_names(self):
        return ['omega']
    
    @property
    def observation_tex_names(self):
        return [r'\omega']
    
    @property
    def observation_units(self):
        return ['1/s']
    
    @property
    def observation_units(self):
        return [r'\frac{1}{s}']
    
    @property
    def observation_space(self):
        return gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        if isinstance(value, (int, float)):
            omega = np.full(self._shape, float(value))
        else:
            omega = np.copy(value)
        self._shape = omega.shape
        self._omega = np.asarray(omega)

    @property
    def speed_shape(self):
        return self._shape
        
    def get_omega(self, load_state):
        """
        Returns:
            float: Constant value for omega in rad/s.
        """
        return self._omega

    def __init__(self, omega=0, j_load=0.0, shape=(1,)):
        """
        Args:
            omega_fixed(float)): Fix value for the speed in rad/s.
        """
        super().__init__(j_load=j_load)
        self._shape = shape
        self.omega = omega

    def get_observation(self, load_state):
        return self._omega
