from scipy.integrate import ode

from gym_electric_motor.physical_systems.ode_solvers import OdeSolver


class ScipyOdeSolver(OdeSolver):
    """Wrapper class for all ode-solvers in the scipy.integrate.ode package.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
    """

    #: Integrator object
    _ode = None

    @property
    def t(self):
        return self._ode.t

    @property
    def y(self):
        return self._ode.y

    def __init__(self, integrator='dopri5', **kwargs):
        """
        Args:
            integrator(str): String to choose the integrator from the scipy.integrate.ode
            kwargs(dict): All parameters that can be set in the "set_integrator"-method of scipy.integrate.ode
        """
        self._solver = None
        self._solver_args = kwargs
        self._integrator = integrator

    def set_system_equation(self, system_equation, jac=None):
        # Docstring of superclass
        super().set_system_equation(system_equation, jac)
        self._ode = ode(system_equation, jac).set_integrator(self._integrator, **self._solver_args)

    def set_initial_value(self, initial_value, t=0):
        # Docstring of superclass
        self._ode.set_initial_value(initial_value, t)

    def set_f_params(self, *args):
        # Docstring of superclass
        super().set_f_params(*args)
        self._ode.set_f_params(*args)
        self._ode.set_jac_params(*args)

    def integrate(self, t):
        # Docstring of superclass
        return self._ode.integrate(t)
