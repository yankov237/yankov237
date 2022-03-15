import gym
import numpy as np
from typing import TYPE_CHECKING, List, Tuple


if TYPE_CHECKING:
    from gym_electric_motor.core import PhysicalSystem


class PhysicalSystem:
    """The Physical System module encapsulates the physical model of the system as well as the simulation from one step
    to the next."""

    @property
    def unwrapped(self) -> 'PhysicalSystem':
        return self

    @property
    def k(self) -> int:
        """The time step of the current episode."""
        return self._k
    
    @property
    def k_cumulative(self) -> int:
        """The time step k of the system across all episodes."""
        return self._k_cumulative
    
    @property
    def t(self) -> float:
        """The current of the episode in seconds."""
        return self._t
    
    @property
    def t_cumulative(self) -> float:
        """The cumulative time of the system across all episodes."""
        return self._t_cumulative

    @property
    def tau(self) -> float:
        """The duration of one simulation step in seconds."""
        raise NotImplementedError

    @property
    def state_names(self) -> List[str]:
        """An array containing the names of the systems states."""
        raise NotImplementedError

    @property
    def action_space(self) -> gym.spaces.Space:
        """gym.Space: An OpenAI Gym Space that describes the possible actions on the system."""
        raise NotImplementedError

    @property
    def state_observation_space(self) -> gym.spaces.Box:
        """gym.Space: An OpenAI Gym Space that describes the possible states of the system."""
        raise NotImplementedError

    @property
    def limits(self) -> np.ndarray:
        """ndarray(float): An array containing the maximum allowed physical values for each state variable."""
        return NotImplementedError

    @property
    def nominal_state(self) -> np.ndarray:
        """ndarray(float): An array containing the nominal values for each state variable."""
        return NotImplementedError

    def __init__(self):      
        self._k = 0
        self._t = 0.0
        self._k_cumulative = 0
        self._t_cumulative = 0.0

    def reset(self) -> np.ndarray:
        """Resets the physical system to an initial state before a new episode starts.

        Returns:
             element of state_space: The initial systems state
        """
        self._t = 0
        self._k = 0

    def simulate(self, action) -> np.ndarray:
        """Simulates the Physical System for one time step with the input action.

        This method is called in the environment in every step to update the systems state.

        Args:
            action(element of action_space): The action to play on the system for the next time step.

        Returns:
            element of state_space: The systems state after the action was applied.
        """
        raise NotImplementedError

    def _count_step(self):
        """Helper function to increase the time and step counters."""
        self._k += 1
        self._k_cumulative += 1
        self._t += self._tau
        self._t_cumulative += self._tau

    def close(self):
        """Closes the system and all of its submodules by closing files, saving logs etc.

        Called, when the environment is closed.
        """
        pass
