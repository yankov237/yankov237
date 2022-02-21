from gym_electric_motor.physical_systems.converters import PowerElectronicConverter


class ContDynamicallyAveragedConverter(PowerElectronicConverter):
    """
    Base class for all continuously controlled converters that calculate the input voltages to the motor with a
    dynamically averaged model over one time step.

    This class also implements the interlocking time of the transistors as a discount on the output voltage.
    """

    _reset_action = [0]

    def __init__(self, tau=1e-4, **kwargs):
        # Docstring in base class
        super().__init__(tau=tau, **kwargs)

    def set_action(self, action, t):
        # Docstring in base class
        return super().set_action(min(max(action, self.action_space.low), self.action_space.high), t)

    def convert(self, i_out, t):
        # Docstring in base class
        return [min(max(self._convert(i_out, t) - self._interlock(i_out, t), self.voltages.low[0]), self.voltages.high[0])]

    def _convert(self, i_in, t):
        """
        Calculate an idealized output voltage for the current active action neglecting interlocking times.

        Args:
            i_in(list(float)): Input currents of the motor
            t(float): Time of the system

        Returns:
             float: Idealized output voltage neglecting interlocking times
        """
        raise NotImplementedError

    def i_sup(self, i_out):
        # Docstring in base class
        raise NotImplementedError

    def _interlock(self, i_in, *_):
        """
        Calculate the output voltage discount due to the interlocking time of the transistors

        Args:
            i_in(list(float)): list of all currents flowing into the motor.
        """
        return np.sign(i_in[0]) / self._tau * self._interlocking_time