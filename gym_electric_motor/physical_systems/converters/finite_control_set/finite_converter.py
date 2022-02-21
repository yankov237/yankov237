from gym_electric_motor.physical_systems.converters import PowerElectronicConverter


class FiniteConverter(PowerElectronicConverter):
    """Base class for all finite converters."""

    #: The switching states of the converter for the current action
    _switching_pattern = []
    #: The current switching state of the converter
    _switching_state = 0
    #: The action that is the default after reset
    _reset_action = 0

    def __init__(self, tau=1e-5, interlocking_time=0.0):
        # Docstring in base class
        super().__init__(tau=tau, interlocking_time=interlocking_time)

    def set_action(self, action, t):
        assert self.action_space.contains(action), \
            f"The selected action {action} is not a valid element of the action space {self.action_space}."
        return super().set_action(action, t)

    def convert(self, i_out, t):
        # Docstring in base class
        raise NotImplementedError

    def i_sup(self, i_out):
        # Docstring in base class
        raise NotImplementedError