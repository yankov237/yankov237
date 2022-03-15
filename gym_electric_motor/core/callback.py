import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gym_electric_motor.core import ElectricMotorEnvironment    

class Callback:
    """The abstract base class for Callbacks in GEM.
    Each of its functions gets called at one point in the :mod:`~gym_electric_motor.core.ElectricMotorEnvironment`.
    
    Attributes:
        _env: A reference to the GEM environment to have full control over the environment durint runtime.
    """

    def __init__(self):
        self._env = None

    def set_env(self, env: 'ElectricMotorEnvironment'):
        """Sets the environment."""
        self._env = env

    def on_reset_begin(self):
        """Gets called at the beginning of each reset"""
        pass

    def on_reset_end(self, state: np.ndarray, reference: np.ndarray):
        """Gets called at the end of each reset"""
        pass

    def on_step_begin(self, k: int, action: np.ndarray or int):
        """Gets called at the beginning of each step"""
        pass

    def on_step_end(self, k: int, state: np.ndarray, reference: np.ndarray, reward: float, done: bool):
        """Gets called at the end of each step"""
        pass

    def on_close(self):
        """Gets called at the beginning of a close"""
        pass
