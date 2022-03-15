import numpy as np

from gym_electric_motor.physical_systems import SCMLComponent
from gym_electric_motor.core.random_component import RandomComponent


class MechanicalLoad(SCMLComponent, RandomComponent):
    """The MechanicalLoad is the base class for all the mechanical systems attached
    to the electrical motors rotor.

    It contains an mechanical ode system as well as the state names, limits and
    nominal values of the mechanical quantities.
    """

    @property
    def ode_size(self):
        return 0

    @property
    def j_total(self):
        """
        Returns:
             float: Total moment of inertia affecting the motor shaft.
        """
        return self._j_total

    #: Parameter indicating if the class is implementing the optional jacobian function
    HAS_JACOBIAN = False

    def __init__(self, j_load=0.0):
        """
        Args:
            j_load(float): Moment of inertia of the load affecting the motor shaft.
        """
        SCMLComponent.__init__(self)
        RandomComponent.__init__(self)
        self._j_total = self._j_load = j_load

    def get_omega(self, load_state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self):
        """Resets the load to a new, initial state."""
        self.next_generator()

    def set_j_rotor(self, j_rotor):
        """
        Args:
            j_rotor(float): The moment of inertia of the rotor shaft of the motor.
        """
        self._j_total += j_rotor

    def mechanical_ode(self, t: float, mechanical_state: np.ndarray, torque: np.ndarray):
        """Calculates the derivative of the mechanical-ODE for each of the mechanical states.

        Args:
            t(float): Current time of the system.
            mechanical_state(np.ndarray[float]): Current state of the mechanical system.
            torque(np.ndarray[float]): Generated input torque by the electrical motor.

        Returns:
            np.ndarray[float]: Derivatives of the mechanical state for the given input torque.
        """
        return self._empty_array

    def mechanical_jacobian(self, t, mechanical_state, torque):
        """Calculates the jacobian of the mechanical-ODE for each of the mechanical state.

        Overriding this method is optional for each subclass. If it is overridden, the parameter HAS_JACOBIAN must also
        be set to True. Otherwise, the jacobian will not be called.

        Args:
            t(float): Current time of the system.
            mechanical_state(ndarray(float)): Current state of the mechanical system.
            torque(float): Generated input torque by the electrical motor.

        Returns:
            Tuple(ndarray, ndarray):
                [0]: Derivatives of the mechanical_state-odes over the mechanical_states shape:(states x states)
                [1]: Derivatives of the mechanical_state-odes over the torque shape:(states,)
        """
        return self._empty_array, self._empty_array
