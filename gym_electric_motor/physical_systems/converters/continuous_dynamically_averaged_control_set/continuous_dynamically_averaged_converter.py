import numpy as np

from gym_electric_motor.physical_systems.converters import PowerElectronicConverter


class ContDynamicallyAveragedConverter(PowerElectronicConverter):
    """Base class for all continuously controlled converters that calculate the input voltages to the motor with a
    dynamically averaged model over one time step.

    This class also implements the interlocking time of the transistors as a discount on the output voltage.
    """

    _reset_action = [0]

    def __init__(self, tau=1e-4):
        # Docstring in base class
        super().__init__()
        self._tau = tau

    def set_action(self, action, t):
        # Docstring in base class
        return super().set_action(t, np.clip(action, self.action_space.low, self.action_space.high))      
