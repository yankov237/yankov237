import numpy as np
from typing import TYPE_CHECKING, List, Callable, Iterable

import gym_electric_motor as gem

if TYPE_CHECKING:
    from gym_electric_motor.core import Constraint, PhysicalSystem


class ConstraintMonitor:
    """The ConstraintMonitor is used within the ElectricMotorEnvironment to monitor the states for illegal / undesired
    values (e.g. overcurrents).

    It consists of a list of multiple independent constraints. Each constraint gets the current observation of the
    environment as input and returns a *violation degree* within :math:`[0.0, 1.0]`.
    All these are merged together and the ConstraintMonitor returns a total violation degree.

    **Soft Constraints:**
        To enable a higher flexibility, the constraints return a violation degree (float) instead of a simple violation
        flag (bool). So, even before the limits are violated, the reward function can take the limit violation degree
        into account. If the violation degree is at 0.0, no states are in a dangerous region. For values between 0.0 and
        1.0 the reward will be decreased gradually so that the agent will learn to avoid these state regions.
        If the violation degree reaches 1.0 the episode is terminated.

    **Hard Constraints:**
        With the above concept, also hard constraints can be modeled that directly terminate an episode without any 
        "danger"-region. Then, the violation degree of the constraint directly changes from 0.0 to 1.0, if a violation
        occurs.

    """

    @property
    def constraints(self) -> List[Callable]:
        """Returns the list of all constraints the ConstraintMonitor observes."""
        return self._constraints

    def __init__(
        self,
        limit_constraints: Iterable[str] or str = (),
        additional_constraints: Iterable['Constraint' or Callable] = (),
        merge_violations: str or Callable = 'max'
    ):
        """
        Args:
            limit_constraints(list(str)/'all_states'):
                Shortcut parameter to pass all states that limits shall be observed.
                    - list(str): Pass a list with state_names and all of the states will be observed to stay within
                        their limits.
                    - 'all_states': Shortcut for all states are observed to stay within the limits.

            additional_constraints(list(Constraint or Callable)):
                 Further constraints that shall be monitored. These have to be initialized first and passed to the
                 ConstraintMonitor. Alternatively, constraints can be defined as a function that takes the current
                 state and returns a float within [0.0, 1.0].
            merge_violations('max'/'product'/callable(*violation_degrees) -> float): Function to merge all single
                violation degrees to a total violation degree.
                    - 'max': Take the maximal violation degree as total violation degree.
                    - 'product': Calculates the total violation degree as one minus the product of one minus all single
                        violation degrees.
                    - callable(*violation_degrees) -> float: User defined function to calculate the total violation.
        """
        self._constraints = list(additional_constraints)
        if len(limit_constraints) > 0:
            self._constraints.append(gem.constraints.LimitConstraint(limit_constraints))

        assert all(callable(constraint) for constraint in self._constraints)
        assert merge_violations in ['max', 'product'] or callable(merge_violations)

        if len(self._constraints) == 0:
            # Without any constraint, always return 0.0 as violation
            self._merge_violations = lambda *violation_degrees: 0.0
        elif merge_violations == 'max':
            self._merge_violations = max
        elif merge_violations == 'product':
            def product_merge(*violation_degrees):
                return 1 - np.prod([(1 - violation) for violation in violation_degrees])
            self._merge_violations = product_merge
        elif callable(merge_violations):
            self._merge_violations = merge_violations

    def set_modules(self, ps: 'PhysicalSystem'):
        """The PhysicalSystem of the environment is passed to save important parameters like the index of the states.

        Args:
            ps(PhysicalSystem): The PhysicalSystem of the environment.
        """
        for constraint in self._constraints:
            if isinstance(constraint, gem.core.Constraint):
                constraint.set_modules(ps)

    def check_constraints(self, state: np.ndarray):
        """Function to check and merge all constraints.

        Args:
            state(ndarray(float)): The current environments state.

        Returns:
            float: The total violation degree in [0,1]
        """
        violations = [constraint(state) for constraint in self._constraints]
        return self._merge_violations(violations)
