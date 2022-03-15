import gym
import numpy as np

from gym_electric_motor.physical_systems import SCMLComponent


class PowerElectronicConverter(SCMLComponent):
    """Base class for all converters in a SCMLSystem."""

    @property
    def action_space(self):
        raise NotImplementedError

    @property
    def action(self):
        return self._action

    @property
    def output_shape(self):
        raise NotImplementedError

    @property
    def supply_shape(self):
        raise NotImplementedError

    @property
    def tau(self):
        return self._tau
    
    @tau.setter
    def tau(self, value):
        self._tau = float(value)
    
    #: Default action that is taken after a reset.
    _reset_action = None

    def __init__(self):
        """
        Args:
            tau(float): Discrete time step of the system in seconds
        """
        self.tau = 1e-4
        self._action_start_time = 0.0
        self._action = self._reset_action

    def reset(self):
        """Resets all converter states to a default."""
        self._action = self._reset_action
        self._action_start_time = 0.0

    def set_action(self, t, action):
        """Sets the next action of the converter at the beginning of a simulation step in the system.

        Args:
            t(float): Time at the beginning of the simulation step in seconds.
            action(element of action_space): The control action on the converter.

        Returns:
            list(float): Times when a switching action occurs and the conversion function must be called by the system.
        """
        self._action_start_time = t
        self._current_action = action
        return self._set_switching_pattern(t, action)

    def i_sup(self, i_out):
        """Calculates the current, the converter takes from the supply for the given output currents and the current switching state.

        Args:
            i_out(list(float)): All currents flowing out of the converter and into the motor.

        Returns:
            float: The current drawn from the supply.
        """
        raise NotImplementedError

    def convert(self, t, i_out, u_sup):
        """The conversion function that converts the previously set action to an input voltage for the motor.

        This function has to be called at least at every previously defined switching time, because the input voltage
        for the motor might change at these times.

        Args:
            t(float): Current time of the system.
            i_out(np.ndarray[float]): All currents that flow out of the converter into the motor.
            u_in(np.ndarray[float]): All supply voltages.

        Returns:
             np.ndarray(float): All input voltages to the motor.
        """
        raise NotImplementedError

    def _set_switching_pattern(self, t, action):
        """Method to calculate the switching pattern and corresponding switching times for the next time step.
        
        At least, the next time step [t + tau] is returned.

        Returns:
             list(float): Switching times.
        """
        self._switching_pattern = (action,)
        return []
