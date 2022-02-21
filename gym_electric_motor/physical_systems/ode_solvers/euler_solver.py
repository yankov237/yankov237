from gym_electric_motor.physical_systems.ode_solvers import OdeSolver


class EulerSolver(OdeSolver):
    """Solves a system of differential equations of first order for a given time step with linear approximation.

        .. math:
            x^\prime(t) = f(x(t))

        .. math:
            x(t + \\frac{\\tau}{nsteps}) = x(t) + x^\prime(t) * \\frac{\\tau}{nsteps}
    """

    def __init__(self, nsteps=1):
        """
        Args:
            nsteps(int): Number of cycles to calculate for each iteration. Higher steps make the system more accurate,
                but take also longer to compute.
        """
        self._nsteps = nsteps
        self._integrate = self._integrate_one_step if nsteps == 1 else self._integrate_nsteps

    def integrate(self, t):
        # Docstring of superclass
        return self._integrate(t)

    def _integrate_nsteps(self, t):
        """Integration method for nsteps > 1

        Args:
            t(float): Time until the system shall be calculated

        Returns:
            ndarray(float):The new state of the system.
        """
        tau = (t - self._t) / self._nsteps
        state = self._y
        current_t = t
        for _ in range(self._nsteps):
            delta = self._system_equation(current_t + tau, state, *self._f_params) * tau
            state = state + delta
            current_t += tau
        self._y = state
        self._t = t
        return self._y

    def _integrate_one_step(self, t):
        """Integration method for nsteps = 1. (For faster computation)

        Args:
            t(float): Time until the system shall be calculated

        Returns:
            ndarray(float):The new state of the system.
        """
        self._y = self._y + self._system_equation(self._t, self._y, *self._f_params) * (t - self._t)
        self._t = t
        return self._y
