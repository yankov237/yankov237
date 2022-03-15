from gym_electric_motor.physical_systems.converters import PowerElectronicConverter


class FiniteConverter(PowerElectronicConverter):
    """Base class for all finite converters."""

    #: The switching states of the converter for the current action
    _switching_pattern = []
    #: The current switching state of the converter
    _switching_state = 0
    #: The action that is the default after reset
    _reset_action = 0

    def __init__(self, tau=1e-5):
        # Docstring in base class
        super().__init__()
        self.tau = tau

    def set_action(self, action, t):
        assert self.action_space.contains(action), \
            f"The selected action {action} is not a valid element of the action space {self.action_space}."
        return super().set_action(action, t)
