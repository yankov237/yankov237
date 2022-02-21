from gym.spaces import Box
from gym_electric_motor.physical_systems.ode_solvers.solvers import EulerSolver
import warnings
import numpy as np


class VoltageSupply:
    """Base class for all VoltageSupplies to be used in a SCMLSystem.

    Parameter:
        supply_range(Tuple(float,float)): Minimal and maximal possible value for the voltage supply.
        u_nominal(float): Nominal supply voltage
    """

    @property
    def u_sup(self):
        """np.ndarray[float]: Currently supplied voltage."""
        return self._u_sup

    @property
    def u_nominal(self):
        """float: Nominal Voltage of the Voltage Supply"""
        return self._u_nominal

    @property
    def state_names(self):
        """Tuple(string): Names of the returned states."""
        return ()
    
    @property
    def state_space(self):
        """gym.spaces.Box: Space of the observed state."""
        return Box(np.array([]),np.array([]), dtype=np.float64)

    @property
    def supply_space(self):
        """gym.spaces.Box: Space of the supplied voltage."""
        return Box(-1, 1, shape=(1,), dtype=np.float64)
    
    def __init__(self, u_nominal, shape=(1,)):
        """
        Args:
            u_nominal(float): Nominal voltage of the Voltage Supply.
        """
        self._u_nominal = u_nominal
        self._u_sup = None

    def reset(self):
        """Resets the voltage supply to an initial state.

        This method is called at every reset of the physical system.
        """
        pass

    def get_u_sup(self, t, i_sup):
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
            numpy.ndarray[float]: The observation for the physical system.
        """
        return np.array([])









