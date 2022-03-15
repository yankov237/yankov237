import gym_electric_motor as gem

import numpy as np


class Constraint:
    """Base class for all constraints in the ConstraintMonitor."""

    def __call__(self, state: np.ndarray):
        """Checks the environment state for a violation of this constraint.

        Args:
            state(numpy.ndarray(float)): The current physical systems state.

        Returns:
              float in [0.0, 1.0]: Degree how much the constraint has been violated.
                0.0: No violation
                (0.0, 1.0): Undesired zone near to a full violation. No episode termination.
                1.0: Full violation and episode termination.
        """
        raise NotImplementedError

    def set_modules(self, ps: gem.core.PhysicalSystem):
        """Called by the environment that the Constraint can read information from the PhysicalSystem.

        Args:
            ps(PhysicalSystem): PhysicalSystem of the environment.
        """
        pass
