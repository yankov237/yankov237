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
        """A boolean-array of the state_variables indicating which states are referenced."""
        return self._referenced_states

    @property
    def reference_names(self):
        """A list containing all names of the referenced states in the reference observation."""
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
        """Resets the reference generator instance for a new episode."""
        pass

    def close(self):
        """Called by the environment, when the environment is deleted to close files, store logs, etc."""
        pass
