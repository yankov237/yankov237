from typing import Callable, Tuple
import numpy as np

from gym_electric_motor.physical_systems import VoltageSupply


class ACSupply(VoltageSupply):
    """An AC voltage supply"""

    @property
    def u_sup(self):
        return self._u_sup

    @property
    def reset_phi(self) -> Callable:
        return self._reset_phi

    @reset_phi.setter
    def reset_phi(self, value : Callable):
        self._reset_phi = value
    
    @property
    def u_nominal(self) -> np.ndarray:
        return self._u_nominal

    @u_nominal.setter
    def u_nominal(self, value : np.ndarray):
        self._u_nominal = value

    @property
    def supply_shape(self) -> Tuple[int]:
        return self._u_nominal.shape

    def __init__(self, u_nominal=230, frequency=50, shape=(1,),):
        """
        Args:
            u_nominal(float): Single phasic effective value of the voltage supply
            shape(tuple[int]):
            frequency(float):
        """

        super().__init__()
        if isinstance(u_nominal, (float, int)):
            self._u_nominal = np.full(shape, u_nominal)
        else:
            self._u_nominal = np.array(u_nominal)

        if isinstance(frequency, (float, int)):
            self._frequency = np.full(shape, frequency)
        else:
            self._frequency = np.array(frequency)
        
        assert self._u_nominal.shape == self._frequency.shape, \
             'Shape mismatch between nominal voltage (u_nominal) and frequency of the voltage supply.'

        self._amplitude = np.sqrt(2) * self._u_nominal

        self._reset_phi = lambda shape: 2 * np.pi * np.random.rand(*shape) 
        
        
    def reset(self):
        self._phi = self.reset_phi(self.supply_shape)
        
    
    def get_u_sup(self, t: float, i_sup: np.ndarray) -> np.ndarray:
        # Docstring of superclass
        self._u_sup = self._amplitude * np.sin( 2 * np.pi * self._frequency * t + self._phi)
        return self._u_sup