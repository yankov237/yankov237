from gym.spaces import Box
from gym_electric_motor.physical_systems.solvers.solvers import EulerSolver
import warnings
import numpy as np


class VoltageSupply:
    """Base class for all VoltageSupplies to be used in a SCMLSystem.

    Parameter:
        supply_range(Tuple(float,float)): Minimal and maximal possible value for the voltage supply.
        u_nominal(float): Nominal supply voltage
    """

    @property
    def state_names(self):
        """Tuple(string): Names of the returned states."""
        return ()
    
    @property
    def state_space(self):
        """gym.spaces.Box: Space of the observed state."""
        return Box(np.array([]),np.array([]), dtype=np.float64)

    @property
    def u_nominal(self):
        """
        Returns:
             float: Nominal Voltage of the Voltage Supply
        """
        return self._u_nominal

    def __init__(self, u_nominal):
        """
        Args:
            u_nominal(float): Nominal voltage of the Voltage Supply.
        """
        self._u_nominal = u_nominal

    def reset(self):
        """Resets the voltage supply to an initial state.

        This method is called at every reset of the physical system.
        """
        pass

    def get_voltage(self, t, i_sup):
        """
        Get the supply voltage based on the floating supply current i_sup, the time t and optional further arguments.

        Args:
            t(float): Current time of the system.
            i_sup(float): Supply current floating into the system.
            
        Returns:
             list(float): Supply Voltage(s) at time t.
        """
        raise NotImplementedError

    def get_observation(self, t, i_sup):
        """Returns the observation of the voltage supplies state.
        
        Args:
            t(float): Current time of the system.
            i_sup(float): Supply current floating into the system.
        
        Returns:
            Element of state_space: The observation for the physical system.
        """
        return np.array([])









