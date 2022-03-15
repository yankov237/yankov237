import numpy as np

import gym_electric_motor as gem


class SquaredConstraint(gem.core.Constraint):
    """A squared constraint on multiple states as it is required oftentimes for the dq-currents in synchronous and
    asynchronous electric motors.

    .. math::
        1.0 <= \sum_{i \in S} (s_i / s_{i,max})^2

    :math:`S`: Set of the observed PhysicalSystems states
    """

    def __init__(self, states=()):
        """
        Args:
            states(iterable(str)): Names of all states to be observed within the SquaredConstraint.
        """
        self._states = states
        self._state_indices = ()
        self._limits = ()
        self._normalized = False

    def set_modules(self, ps):
        self._state_indices = [ps.state_positions[state] for state in self._states]
        self._limits = ps.limits[self._state_indices]
        self._normalized = not np.all(ps.state_space.high[self._state_indices] == self._limits)

    def __call__(self, state):
        state_ = state[self._state_indices] if self._normalized else state[self._state_indices] / self._limits
        return float(np.sum(state_**2) > 1.0)
