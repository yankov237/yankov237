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

    def _convert(self, *_):
        # Docstring in base class
        return self._current_action[0]

    def i_sup(self, i_out):
        # Docstring in base class
        interlocking_current = 1 if i_out[0] < 0 else 0
        return (
            self._current_action[0]
            + self._interlocking_time / self._tau * (interlocking_current - self._current_action[0])
        ) * i_out[0]
