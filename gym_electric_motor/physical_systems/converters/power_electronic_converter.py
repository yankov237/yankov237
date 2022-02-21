import gym
import numpy as np

from gym_electric_motor.physical_systems import SCMLComponent


class PowerElectronicConverter(SCMLComponent):
    """Base class for all converters in a SCMLSystem.
 
    Properties:
        | *voltages(tuple(float, float))*: Determines which output voltage polarities the converter can generate.
        | E.g. (0, 1) - Only positive voltages / (-1, 1) Positive and negative voltages

        | *currents(tuple(float, float))*: Determines which output current polarities the converter can generate.
        | E.g. (0, 1) - Only positive currents / (-1, 1) Positive and negative currents
    """

    
    #: gym.Space that defines the set of all possible actions for the converter
    action_space = None

    #: gym.Space that defines the space of output voltages
    voltage_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

    #: gym.Space that defines the space of output currents
    current_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float64)
    
    #: gym.Space that defines the space of supply voltages
    supply_voltage_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float64)
    
    #: Default action that is taken after a reset.
    _reset_action = None

    def __init__(self, tau, interlocking_time=0.0):
        """
        Args:
            tau(float): Discrete time step of the system in seconds
            interlocking_time(float): Interlocking time of the transistors in seconds
        """
        self._tau = tau
        self._interlocking_time = interlocking_time
        self._action_start_time = 0.0
        self._current_action = self._reset_action

    def reset(self):
        """Reset all converter states to a default."""
        self._current_action = self._reset_action
        self._action_start_time = 0.0
        return [0.0]

    def set_action(self, action, t):
        """Sets the next action of the converter at the beginning of a simulation step in the system.

        Args:
            action(element of action_space): The control action on the converter.
            t(float): Time at the beginning of the simulation step in seconds.

        Returns:
            list(float): Times when a switching action occurs and the conversion function must be called by the system.
        """
        self._action_start_time = t
        self._current_action = action
        return self._set_switching_pattern(action)

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
            i_out(list(float)): All currents that flow out of the converter into the motor.
            u_in(list(float) / float): All supply voltages.

        Returns:
             list(float): List of all input voltages at the motor.
        """
        raise NotImplementedError

    def _set_switching_pattern(self, action):
        """Method to calculate the switching pattern and corresponding switching times for the next time step.
        
        At least, the next time step [t + tau] is returned.

        Returns:
             list(float): Switching times.
        """
        self._switching_pattern = [action]
        return [self._action_start_time + self._tau]
