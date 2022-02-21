from scipy.integrate import odeint

from gym_electric_motor.physical_systems.ode_solvers import OdeSolver


class ScipyOdeIntSolver(OdeSolver):
    """Wrapper class for all ode-solvers in the scipy.integrate.odeint function.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs(dict): Arguments to pass to the solver. See the scipy description for further information.
        """
        self._solver_args = kwargs

    def integrate(self, t):
        # Docstring of superclass
        result = odeint(self._system_equation, self._y, [self._t, t], args=self._f_params, Dfun=self._system_jacobian,
                        tfirst=True, **self._solver_args)
        self._t = t
        self._y = result[-1]
        return self._y
