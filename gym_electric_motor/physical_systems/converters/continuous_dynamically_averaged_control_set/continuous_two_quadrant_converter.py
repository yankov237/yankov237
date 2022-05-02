import numpy as np
import gym

from gym_electric_motor.physical_systems.converters.continuous_dynamically_averaged_control_set import ContDynamicallyAveragedConverter


class ContTwoQuadrantConverter(ContDynamicallyAveragedConverter):
    """
    Key:
        'Cont-2QC'

    Actions:
        | Duty Cycle upper Transistor: Action
        | Duty Cycle upper Transistor: 1 - Action

    Action Space:
        Box([0,1])

    Output Voltages and Currents:
        | voltages: Box(0, 1, shape=(1,))
        | currents: Box(-1, 1, shape=(1,))
    """

    action_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float64)

    def _add_interlock(self, func):
        def _interlock(self, t, i_out, u_sup):
            u_in = func(t, i_out, u_sup) - np.sign(i_out) / self._tau * self._interlocking_time
            self._u_in[:] = np.clip(u_in, -u_sup, u_sup)
            return self._u_in
        return _interlock

    def i_sup(self, i_out):
        # Docstring in base class
        interlocking_current = 1 if i_out[0] < 0 else 0
        return self._current_action[0] * i_out[0]

    def convert(self, t, i_out, u_sup):
        """The conversion function that converts the previously set action to an input voltage for the motor.

        This function has to be called at least at every previously defined switching time, because the input voltage
        for the motor might change at these times.

        Args:
            t(float): Current time of the system.
            i_out(np.ndarray[float]): All currents that flow out of the converter into the motor.
            u_in(np.ndarray[float]): All supply voltages.

        Returns:
             np.ndarray(float): All input voltages to the motor.
        """
        return self._current_action * u_sup