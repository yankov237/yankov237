from scipy.integrate import solve_ivp

from gym_electric_motor.physical_systems.ode_solvers import OdeSolver


class ScipySolveIvpSolver(OdeSolver):
    """Wrapper class for all ode-solvers in the scipy.integrate.solve_ivp function.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    """

    def __init__(self, **kwargs):
        # Docstring of superclass
        self._solver_kwargs = kwargs

    def set_system_equation(self, system_equation, jac=None):
        # Docstring of superclass
        method = self._solver_kwargs.get('method', None)
        super().set_system_equation(system_equation, jac)

        # Only Radau BDF and LSODA support the jacobian.
        if method in ['Radau', 'BDF', 'LSODA']:
            self._solver_kwargs['jac'] = self._system_jacobian

    def integrate(self, t):
        # Docstring of superclass
        result = solve_ivp(
            self._system_equation, [self._t, t], self._y, t_eval=[t], args=self._f_params, **self._solver_kwargs
        )
        self._t = t
        self._y = result.y.T[-1]
        return self._y
