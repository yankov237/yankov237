import numpy as np
from scipy.stats import truncnorm
import warnings

from ...core.random_component import RandomComponent


class MechanicalLoad(RandomComponent):
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

    @property
    def observation_names(self):
        """
        Returns:
            list(str): Names of the states in the mechanical observation.
        """
        raise NotImplementedError

    @property
    def limits(self):
        """
        Returns:
            dict(float): Mapping of the motor states to their limit values.
        """
        return self._limits

    @property
    def nominal_values(self):
        """
        Returns:
              dict(float): Mapping of the motor states to their nominal values

        """
        return self._nominal_values

    #: Parameter indicating if the class is implementing the optional jacobian function
    HAS_JACOBIAN = False

    def __init__(self, j_load=0.0):
        """
        Args:
            j_load(float): Moment of inertia of the load affecting the motor shaft.
        """
        RandomComponent.__init__(self)
        self._j_total = self._j_load = j_load
        self._limits = {}
        self._nominal_values = {}

    def get_omega(self, load_state):
        raise NotImplementedError

    def reset(self):
        """
        Reset the motors state to a new initial state. (Default 0)

        Args:
            nominal_state(list): nominal values for each state given from
                                  physical system
            state_space(gym.Box): normalized state space boundaries
            state_positions(dict): indexes of system states
        Returns:
            numpy.ndarray(float): The initial motor states.
        """
        self.next_generator()

    def set_j_rotor(self, j_rotor):
        """
        Args:
            j_rotor(float): The moment of inertia of the rotor shaft of the motor.
        """
        self._j_total += j_rotor

    def mechanical_ode(self, t, mechanical_state, torque):
        """
        Calculation of the derivatives of the mechanical-ODE for each of the mechanical states.

        Args:
            t(float): Current time of the system.
            mechanical_state(ndarray(float)): Current state of the mechanical system.
            torque(float): Generated input torque by the electrical motor.

        Returns:
            ndarray(float): Derivatives of the mechanical state for the given input torque.
        """
        return ()

    def mechanical_jacobian(self, t, mechanical_state, torque):
        """
        Calculation of the jacobians of the mechanical-ODE for each of the mechanical state.

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
        return (), ()
