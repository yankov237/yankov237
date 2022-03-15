import numpy as np

from gym_electric_motor.physical_systems.voltage_supplies import VoltageSupply


class IdealVoltageSupply(VoltageSupply):
    """Ideal Voltage Supply that supplies with u_nominal independent of the time and the supply current."""

    @property
    def u_nominal(self) -> np.ndarray:
        return self._u_nominal

    @property
    def u_sup(self) -> np.ndarray:
        return self._u_nominal

    @u_nominal.setter
    def u_nominal(self, value):
        if isinstance(value, (int, float)):
            u_nom = np.full(self._supply_shape, float(value))
        else:
            u_nom = np.copy(value)
        self._supply_shape = u_nom.shape
        self._u_nominal = u_nom

    @property
    def supply_shape(self):
        return self._supply_shape

    def __init__(self, u_nominal=600.0, shape=(1,)):
        super().__init__()
        self._supply_shape = shape
        self.u_nominal = u_nominal

    def get_u_sup(self, t, i_sup):
        # Docstring of superclass
        return self._u_nominal