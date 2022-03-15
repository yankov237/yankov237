import numpy as np


class RewardFunction:
    """The abstract base class for reward functions in gym electric motor environments.

    The reward function is called once per step and returns reward for the current time step.

    Attributes:
        reward_range(Tuple(float, float)):Defining lowest and highest possible rewards.
    """

    #: Tuple(int,int): Lower and upper possible reward
    reward_range = (-np.inf, np.inf)

    def __call__(self, state, reference, k, action, violation_degree) -> float:
        """Calculates the reward.

        Args:
            state(numpy.ndarray(float)): State array of the environment.
            reference(numpy.ndarray(float)): Reference array of the environment.
            k(int): Systems momentary time-step
            action(element of action-space): The taken action :a_{k-1}: at the beginning of the step.
            violation_degree(float in [0.0, 1.0]): Degree of violation of the constraints. 0.0 indicates that all
                constraints are complied. 1.0 indicates that the constraints have been so much violated, that a reset is
                necessary.


        Returns:
            float: The reward for the state, reference pair
        """
        return self.reward(state, reference, k, action, violation_degree)

    def set_modules(self, physical_system, reference_generator, constraint_monitor):
        """
        Setting of the physical system, to set state arrays fitting to the environments states

        Args:
            physical_system(PhysicalSystem): The physical system of the environment
            reference_generator(ReferenceGenerator): The reference generator of the environment.
            constraint_monitor(ConstraintMonitor): The constraint monitor of the environment.
        """
        pass

    def reward(self, state, reference, k=None, action=None, violation_degree=0.0) -> float:
        """Calculates the reward for the state-rerefence pair.

        Args:
            state(ndarray(float)): Environments state array.
            reference(ndarray(float)): Environments reference array.
            k(int): Systems momentary time-step
            action(element of action space): The previously taken action.
            violation_degree(float in [0.0, 1.0]): Degree of violation of the constraints. 0.0 indicates that all
                constraints are complied. 1.0 indicates that the constraints have been so much violated, that a reset is
                necessary.

        Returns:
            float: Reward for this state, reference, action tuple.
        """

        raise NotImplementedError

    def reset(self):
        """Resets and initializes the inner state of the RewardFunction for the next episode.

        This function is called by the environment when reset.
        Inner states of the reward function can be reset here, if necessary.
        """
        pass

    def close(self):
        """Called, when the environment is closed to store logs, close files etc."""
        pass
