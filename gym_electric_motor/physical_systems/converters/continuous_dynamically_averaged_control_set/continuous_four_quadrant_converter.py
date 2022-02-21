import gym
import numpy as np

from gym_electric_motor.physical_systems.converters.continuous_dynamically_averaged_control_set import ContDynamicallyAveragedConverter, ContTwoQuadrantConverter


class ContFourQuadrantConverter(ContDynamicallyAveragedConverter):
    """
    The continuous four quadrant converter (4QC) is simulated with two continuous 2QC.

    Key:
        'Cont-4QC'

    Actions:
        | Duty Cycle Transistor T1: 0.5 * (Action + 1)
        | Duty Cycle Transistor T2: 1 - 0.5 * (Action + 1)
        | Duty Cycle Transistor T3: 1 - 0.5 * (Action + 1)
        | Duty Cycle Transistor T4: 0.5 * (Action + 1)

    Action Space:
        Box(-1, 1, shape=(1,))

    Output Voltages and Currents:
        | voltages: Box(-1, 1, shape=(1,))
        | currents: Box(-1, 1, shape=(1,))
    """

    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

    def __init__(self, **kwargs):
        # Docstring in base class
        super().__init__(**kwargs)
        self._subconverters = [ContTwoQuadrantConverter(**kwargs), ContTwoQuadrantConverter(**kwargs)]

    def _convert(self, *_):
        # Not used here
        pass

    def reset(self):
        # Docstring in base class
        self._subconverters[0].reset()
        self._subconverters[1].reset()
        return super().reset()

    def convert(self, i_out, t):
        # Docstring in base class
        return [self._subconverters[0].convert(i_out, t)[0] - self._subconverters[1].convert(i_out, t)[0]]

    def set_action(self, action, t):
        # Docstring in base class
        super().set_action(action, t)
        times = []
        times += self._subconverters[0].set_action([0.5 * (action[0] + 1)], t)
        times += self._subconverters[1].set_action([-0.5 * (action[0] - 1)], t)
        return sorted(list(set(times)))

    def i_sup(self, i_out):
        # Docstring in base class
        return self._subconverters[0].i_sup(i_out) + self._subconverters[1].i_sup([-i_out[0]])
