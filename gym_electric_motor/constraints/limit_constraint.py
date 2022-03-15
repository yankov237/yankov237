import numpy as np

import gym_electric_motor as gem


class LimitConstraint(gem.core.Constraint):
    """Constraint to observe the limits on one or more system states.

    This constraint observes if any of the systems state values exceeds the limit specified in the PhysicalSystem.

    .. math::
        1.0 >= s_i / s_{i,max}

    For all :math:`i` in the set of PhysicalSystems states :math:`S`.

    """

    def __init__(self, observed_state_names='all_states'):
        """
        Args:
            observed_state_names(['all_states']/iterable(str)): The states to observe. \n
                - ['all_states']: Shortcut for observing all states.
                - iterable(str): Pass an iterable containing all state names of the states to observe.
        """
        self._observed_state_names = observed_state_names
        self._limits = None
        self._observed_states = None

    def __call__(self, state):
        observed = state[self._observed_states]
        violation = any(abs(observed) > 1.0)
        return float(violation)

    def set_modules(self, ps):
        self._limits = ps.limits
        if 'all_states' in self._observed_state_names:
            self._observed_state_names = ps.state_names
        if self._observed_state_names is None:
            self._observed_state_names = []
        self._observed_states = gem.utils.set_state_array(
            dict.fromkeys(self._observed_state_names, 1), ps.state_names
        ).astype(bool)
