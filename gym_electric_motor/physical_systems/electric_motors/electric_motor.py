import numpy as np
from scipy.stats import truncnorm

from ...core.random_component import RandomComponent

from gym_electric_motor.utils import update_parameter_dict


class ElectricMotor(RandomComponent):
    """Base class for all technical electrical motor models.

        A motor consists of the ode-state. These are the dynamic quantities of its ODE.
        For example:
            ODE-State of a DC-shunt motor: `` [i_a, i_e ] ``
                * i_a: Anchor circuit current
                * i_e: Exciting circuit current

        Each electric motor can be parametrized by a dictionary of motor parameters,
        the nominal state dictionary and the limit dictionary.
    """

    #: Parameter indicating if the class is implementing the optional jacobian function
    HAS_JACOBIAN = False

    #: _default_motor_parameter(dict): Default parameter dictionary for the motor
    _default_motor_parameter = {}
    #: _default_nominal_values(dict(float)): Default nominal motor state array
    _default_nominal_values = {}
    #: _default_limits(dict(float)): Default motor limits (0 for unbounded limits)
    _default_limits = {}

    @property
    def ode_size(self):
        raise NotImplementedError

    @property
    def motor_parameter(self):
        """
        Returns:
             dict(float): The motors parameter dictionary
        """
        return self._motor_parameter

    def __init__(self, motor_parameter=None, nominal_values=None, limit_values=None):
        """
        Args:
            motor_parameter(dict(float)): Motor parameter dictionary. Contents specified
                for each motor.
            nominal_values(dict(float)): Nominal values for the motor quantities.
            limit_values(dict(float)): Limits for the motor quantities.
        """
        RandomComponent.__init__(self)
        motor_parameter = motor_parameter or {}
        self._motor_parameter = update_parameter_dict(self._default_motor_parameter, motor_parameter)
        limit_values = limit_values or {}
        self._limits = update_parameter_dict(self._default_limits, limit_values)
        nominal_values = nominal_values or {}
        self._nominal_values = update_parameter_dict(self._default_nominal_values, nominal_values)
        self._u_in = None

    def electrical_ode(self, motor_state, omega):
        """Calculation of the derivatives of each motor state variable for the given inputs / The motors ODE-System.

        Args:
            state(ndarray(float)): The motors state.
            omega(float): Angular velocity of the motor

        Returns:
             ndarray(float): Derivatives of the motors ODE-system for the given inputs.
        """
        raise NotImplementedError

    def electrical_jacobian(self, motor_state, omega):
        """
        Calculation of the jacobian of each motor ODE for the given inputs / The motors ODE-System.

        Overriding this method is optional for each subclass. If it is overridden, the parameter HAS_JACOBIAN must also
        be set to True. Otherwise, the jacobian will not be called.

        Args:
            state(ndarray(float)): The motors state.
            omega(float): Angular velocity of the motor

        Returns:
             Tuple(ndarray, ndarray, ndarray):
                [0]: Derivatives of all electrical motor states over all electrical motor states shape:(states x states)
                [1]: Derivatives of all electrical motor states over omega shape:(states,)
                [2]: Derivative of Torque over all motor states shape:(states,)
        """
        pass

    def reset(self):
        """Resets the motor to a new initial state.

        Returns:
            numpy.ndarray(float): The initial motor states.
        """
        self.next_generator()
        
    def i_in(self, motor_state):
        """
        Args:
            motor_state(ndarray(float)): ODE state of the motor

        Returns:
             list(float): List of all currents flowing into the motor.
        """
        raise NotImplementedError

    def set_u_in(self, u_in):
        """Sets the input voltage for the next simulation step.

        Args:
            u_in: Applied input voltage
        """
        self._u_in = u_in
