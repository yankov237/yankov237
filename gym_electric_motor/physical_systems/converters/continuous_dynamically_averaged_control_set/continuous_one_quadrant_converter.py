import gym
import numpy as np

from gym_electric_motor.physical_systems.converters.continuous_dynamically_averaged_control_set import ContinuousDynamicallyAveragedConverter


class ContOneQuadrantConverter(ContinuousDynamicallyAveragedConverter):
    """
    Key:
        'Cont-1QC'

    Action:
        Duty Cycle of the Transistor in [0,1].

    Action Space:
        Box([0,1])

    Output Voltages and Currents:
        | voltages: Box(0, 1, shape=(1,))
        | currents: Box(0, 1, shape=(1,))
    """
    voltages = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float64)
    currents = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float64)
    action_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float64)

    def _convert(self, i_in, *_):
        # Docstring in base class
        return self._current_action[0] if i_in[0] >= 0 else 1

    def _interlock(self, *_):
        # Docstring in base class
        return 0

    def i_sup(self, i_out):
        # Docstring in base class
        return self._current_action[0] * i_out[0]
