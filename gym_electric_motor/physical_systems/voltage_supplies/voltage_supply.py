import gym
import numpy as np

from gym_electric_motor.physical_systems import SCMLComponent


class VoltageSupply(SCMLComponent):
    """Base class for all VoltageSupplies to be used in an SCMLSystem."""

    @property
    def u_sup(self) -> np.ndarray:
        """np.ndarray[float]: Currently supplied voltage."""
        raise NotImplementedError

    @property
    def supply_shape(self):
        """Tuple[int]: Space of the supplied voltage."""
        raise NotImplementedError

    def reset(self):
        """Resets the voltage supply to an initial state.

        This method is called at every reset of the physical system.
        """
        pass

    def get_u_sup(self, t: float, i_sup: np.ndarray) -> np.ndarray:
        """Gets the supply voltage based on the floating supply current i_sup, the time t and optional further arguments.

        Args:
            t(float): Current time of the system.
            i_sup(float): Supply current floating into the system.
            
        Returns:
             np.ndarray[float]: Supply Voltage at time t.
        """
        raise NotImplementedError
