import numpy as np

from gym_electric_motor.physical_systems.voltage_supplies import VoltageSupply


class IdealVoltageSupply(VoltageSupply):
    """Ideal Voltage Supply that supplies with u_nominal independent of the time and the supply current."""

    def __init__(self, u_nominal=600.0):
        # Docstring of superclass
        super().__init__(u_nominal)
        self._observation = np.array([u_nominal])
        self._voltage = [self.u_nominal]

    def get_u_sup(self, t, i_sup):
        # Docstring of superclass
        return [self._u_nominal]

    def get_observation(self, t, i_sup):
        # Docstring of superclass
        return self._observation