class OdeSolver:
    """Interface and base class for all used OdeSolvers in gym-electric-motor."""

    #: Current system time t
    _t = 0
    #: Current system state y
    _y = None
    #: Function parameters that are passed to the system equation and the system jacobian additionally to t and y
    _f_params = None
    #: System equation in the form: _system_equation(t, y, *f_params)
    _system_equation = None
    #: System jacobian in the form _system_jacobian(t,y, *f_params)
    _system_jacobian = None

    @property
    def t(self):
        """
        Returns:
            float: Current system time t
        """
        return self._t

    @property
    def y(self):
        """
        Returns:
            float: Current system state y
        """
        return self._y

    def set_initial_value(self, initial_value, t=0):
        """Sets the new initial system state after reset.

        Args:
            initial_value(numpy.ndarray(float)): Initial system state
            t(float):  Initial system time
        """
        self._y = initial_value
        self._t = t

    def integrate(self, t):
        """Integrates the ODE-System from current time until time t

        Args:
            t(float): Time until the system shall be integrated
        Returns:
            ndarray(float): New system state at time t
        """
        raise NotImplementedError

    def set_system_equation(self, system_equation, jac=None):
        """Sets the system equation.

        Args:
            system_equation(function_pointer): Pointer to the systems equation with the parameters (t, y, *args)
            jac(function_pointer): Pointer to the systems jacobian with the parameters (t, y, *args)
        """
        self._system_equation = system_equation
        self._system_jacobian = jac

    def set_f_params(self, *args):
        """Set further arguments for the systems function call like input quantities.

        Args:
            args(list): Additional arguments for the next function calls.
        """
        self._f_params = args
