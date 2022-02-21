"""
On the core level the electric motor environment and the interface to its submodules are defined. By using these
interfaces further reference generators, reward functions, visualizations or physical models can be implemented.

Each ElectricMotorEnvironment contains the five following modules:

* PhysicalSystem
    - Specification and simulation of the physical model. Furthermore, specifies limits and nominal values\
    for all of its ``state_variables``.
* ReferenceGenerator
    - Calculation of reference trajectories for one or more states of the physical systems ``state_variables``.
* ConstraintMonitor
    - Observation of the PhysicalSystems state to comply to a set of user defined constraints.
* RewardFunction
    - Calculation of the reward based on the physical systems state and the reference.\
* ElectricMotorVisualization
    - Visualization of the PhysicalSystems state, reference and reward for the user.

"""
import gym
import numpy as np
from gym.spaces import Box

from .core.random_component import RandomComponent
from .constraints import Constraint, LimitConstraint
import gym_electric_motor as gem




class ReferenceGenerator:
    """The abstract base class for reference generators in gym electric motor environments.

    reference_space:
        Space of reference observations as defined in the OpenAI Gym Toolbox.

    The reference generator is called twice per step.

    Call of get_reference():
        Returns the reference array which has the same shape as the state array and contains
        values for currently referenced state variables and a default value (e.g zero) for non-referenced variables.
        This reference array is used to calculate rewards.

        Example:
            ``reference_array=np.array([0.8, 0.0, 0.0, 0.0])`` \n
            ``state_variables=['omega', 'torque', 'i', 'u', 'u_sup']`` \n
            Here, the state consists of five quantities but only ``'omega'`` is referenced during control.

    Call of get_reference_observation():
        Returns the reference observation, which is shown to the agent.
        Any shape and content is generally valid, however, values must be within the declared reference space.        
        For example, the reference observation may contain future reference values of the next ``n`` steps.

        Example:
            ``reference_observation = np.array([0.8, 0.6, 0.4])`` \n
            This reference observation may be valid for an omega-controlled environment that shows the agent not
            only the reference for the next time step omega_(t+1)=0.8 but also omega_(t+2)=0.6 and omega_(t+3)=0.4.

    """

    def __init__(self):
        self.reference_space = None
        self._physical_system = None
        self._referenced_states = None
        self._reference_names = None

    @property
    def referenced_states(self):
        """
        Returns:
            ndarray(bool): Boolean-Array with the length of the state_variables indicating which states are referenced.
        """
        return self._referenced_states

    @property
    def reference_names(self):
        """
        Returns:
            reference_names(list(str)): A list containing all names of the referenced states in the reference
            observation.
        """
        return self._reference_names

    def set_modules(self, physical_system):
        """Announcement of the PhysicalSystem to the ReferenceGenerator.

        In subclasses, store all important information from the physical system to the ReferenceGenerator here.
        The environment announces the physical system to the ReferenceGenerator during its initialization.

        Args:
            physical_system(PhysicalSystem): The physical system of the environment.
        """
        self._physical_system = physical_system

    def get_reference(self, state, *_, **__):
        """Returns the reference array of the current time step.

        The reference array needs to be in the same shape as the state variables. For referenced states the reference
        value is passed. For unreferenced states a default value (e.g. Zero) can be set in the reference array.

        Args:
            state(ndarray(float)): Current state array of the environment.

        Returns:
             ndarray(float)): Current reference array.
        """
        raise NotImplementedError

    def get_reference_observation(self, state):
        """Returns the reference observation for the next time step. This observation needs to fit in the reference space.

        Args:
            state(ndarray(float)): Current state array of the environment.

        Returns:
            value in reference_space: Observation for the next reference time step.
        """
        raise NotImplementedError

    def reset(self, initial_state):
        """Reset of references for a new episode."""
        pass

    def close(self):
        """Called by the environment, when the environment is deleted to close files, store logs, etc."""
        pass


class RewardFunction:
    """The abstract base class for reward functions in gym electric motor environments.

    The reward function is called once per step and returns reward for the current time step.

    Attributes:
        reward_range(Tuple(float, float)):Defining lowest and highest possible rewards.
    """

    #: Tuple(int,int): Lower and upper possible reward
    reward_range = (-np.inf, np.inf)

    def __call__(self, state, reference, k, action, violation_degree):
        """Call of the reward calculation.

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

    def reward(self, state, reference, k=None, action=None, violation_degree=0.0):
        """
        Reward calculation. If limits have been violated the reward is calculated with a separate function.

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
        """This function is called by the environment when reset.

        Inner states of the reward function can be reset here, if necessary.

        Args:
            initial_state(ndarray(float)): Initial state array of the Environment
            initial_reference(ndarray(float)): Initial reference array of the environment.
        """
        pass

    def close(self):
        """Called, when the environment is closed to store logs, close files etc."""
        pass


class PhysicalSystem:
    """The Physical System module encapsulates the physical model of the system as well as the simulation from one step
    to the next."""

    @property
    def unwrapped(self):
        return self

    @property
    def k(self):
        """
        Returns:
             int: The current episodes time step k.
        """
        return self._k
    
    @property
    def k_cumulative(self):
        """
        Returns:
             int: The current systems time step k across all episodes.
        """
        return self._k_cumulative
    
    @property
    def t(self):
        """
        Returns:
             float: The current episode time in seconds.
        """
        return self._t
    
    @property
    def t_cumulative(self):
        """
        Returns:
             float: The cumulative systems time across all episodes.
        """
        return self._t_cumulative

    @property
    def tau(self):
        """
        Returns:
             float: The systems time constant tau.
        """
        return self._tau

    @property
    def state_names(self):
        """
        Returns:
             ndarray(str): Array containing the names of the systems states.
        """
        return self._state_names

    @property
    def state_positions(self):
        """
        Returns:
            dict(int): Dictionary mapping the state names to its positions in the state arrays

        """
        return self._state_positions

    @property
    def action_space(self):
        """
        Returns:
            gym.Space: An OpenAI Gym Space that describes the possible actions on the system.
        """
        return self._action_space

    @property
    def state_space(self):
        """
        Returns:
             gym.Space: An OpenAI Gym Space that describes the possible states of the system.
        """
        return self._state_space

    @property
    def limits(self):
        """
        Returns:
             ndarray(float): An array containing the maximum allowed physical values for each state variable.
        """
        return NotImplementedError

    @property
    def nominal_state(self):
        """
        Returns:
             ndarray(float): An array containing the nominal values for each state variable.
        """
        return NotImplementedError

    def __init__(self, action_space, state_space, state_names, tau):
        """
        Args:
            action_space(gym.Space): The set of allowed actions on the system.
            state_space(gym.Space): The set of possible systems states.
            state_names(ndarray(str)): The names of the systems states
            tau(float): The systems simulation time interval.
        """
        self._action_space = action_space
        self._state_space = state_space
        self._state_names = state_names
        self._state_positions = {key: index for index, key in enumerate(self._state_names)}
        self._tau = tau
        self._k = 0

    def reset(self):
        """Resets the physical system to an initial state before a new episode starts.

        Returns:
             element of state_space: The initial systems state
        """
        raise NotImplementedError

    def simulate(self, action):
        """Simulation of the Physical System for one time step with the input action.

        This method is called in the environment in every step to update the systems state.

        Args:
            action(element of action_space): The action to play on the system for the next time step.

        Returns:
            element of state_space: The systems state after the action was applied.
        """
        raise NotImplementedError

    def _count_step(self):
        """Helper function to increase the time and step counters."""
        self._k += 1
        self._k_cumulative += 1
        self._t += self._tau
        self._t_cumulative += self._tau

    def close(self):
        """Closes the system and all of its submodules by closing files, saving logs etc.

        Called, when the environment is closed.
        """
        pass


class Callback:
    """The abstract base class for Callbacks in GEM.
    Each of its functions gets called at one point in the :mod:`~gym_electric_motor.core.ElectricMotorEnvironment`.
    Attributes:
        _env: The GEM environment. Use it to have full control over the environment on runtime.
    """

    def __init__(self):
        self._env = None

    def set_env(self, env):
        """Sets the environment of the motor."""
        self._env = env

    def on_reset_begin(self):
        """Gets called at the beginning of each reset"""
        pass

    def on_reset_end(self, state, reference):
        """Gets called at the end of each reset"""
        pass

    def on_step_begin(self, k, action):
        """Gets called at the beginning of each step"""
        pass

    def on_step_end(self, k, state, reference, reward, done):
        """Gets called at the end of each step"""
        pass

    def on_close(self):
        """Gets called at the beginning of a close"""
        pass


class ElectricMotorVisualization(Callback):
    """Base class for all visualizations in GEM.
    The visualization is basically only a Callback that is extended by a render() function to update the figure.
    With the function calls that are inherited by the Callback superclass (e.g. *on_step_end*),
    the data is passed from the environment to the visualization. In the render() function the passed data can be
    visualized in the desired way.
    """

    def render(self):
        """Function to update the user interface."""
        raise NotImplementedError


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
        With the above concept, also hard constraints that directly terminate an episode without any "danger"-region
        can be modeled. Then, the violation degree of the constraint directly changes from 0.0 to 1.0, if a violation
        occurs.

    """

    @property
    def constraints(self):
        """Returns the list of all constraints the ConstraintMonitor observes."""
        return self._constraints

    def __init__(self,  limit_constraints=(), additional_constraints=(), merge_violations='max'):
        """
        Args:
            limit_constraints(list(str)/'all_states'):
                Shortcut parameter to pass all states that limits shall be observed.
                    - list(str): Pass a list with state_names and all of the states will be observed to stay within
                        their limits.
                    - 'all_states': Shortcut for all states are observed to stay within the limits.

            additional_constraints(list(Constraint/callable)):
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
            self._constraints.append(LimitConstraint(limit_constraints))

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

    def set_modules(self, ps: PhysicalSystem):
        """The PhysicalSystem of the environment is passed to save important parameters like the index of the states.

        Args:
            ps(PhysicalSystem): The PhysicalSystem of the environment.
        """
        for constraint in self._constraints:
            if isinstance(constraint, Constraint):
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
