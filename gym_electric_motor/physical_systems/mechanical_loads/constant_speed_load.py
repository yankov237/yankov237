import numpy as np

from .mechanical_load import MechanicalLoad


class ConstantSpeedLoad(MechanicalLoad):
    """Constant speed mechanical load system which will always set the speed to a fixed value."""

    HAS_JACOBIAN = True

    @property
    def get_omega(self, load_state):
        """
        Returns:
            float: Constant value for omega in rad/s.
        """
        return self._omega
    
    def set_omega(self, omega):
        self._omega = float(omega)

    def __init__(self, omega_fixed=0, j_load=0.0):
        """
        Args:
            omega_fixed(float)): Fix value for the speed in rad/s.
        """
        super().__init__(j_load=j_load)
        self._omega = omega_fixed 

    def get_observation(self, load_state):
        return self._omega
