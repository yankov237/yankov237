import gym
import numpy as np

from gym_electric_motor.physical_systems.converters.continuous_dynamically_averaged_control_set import ContDynamicallyAveragedConverter, ContTwoQuadrantConverter



class ContB6BridgeConverter(ContDynamicallyAveragedConverter):
    """
    The continuous B6 bridge converter (B6C) is simulated with three continuous 2QC.

    Key:
        'Cont-B6C'

    Actions:
        The Duty Cycle for each half bridge in the range of (-1,1)

    Action Space:
        Box(-1, 1, shape=(3,))

    Output Voltages and Currents:
        | voltages: Box(-1,1, shape=(3,))
        | currents: Box(-1,1, shape=(3,))

    Output Voltage Space:
        Box(-0.5, 0.5, shape=(3,))
    """

    action_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float64)
    # Only positive voltages can be applied
    voltages = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float64)
    # Positive and negative currents are possible
    currents = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float64)

    _reset_action = [0, 0, 0]

    def __init__(self, tau=1e-4, **kwargs):
        # Docstring in base class
        super().__init__(tau=tau, **kwargs)
        self._sub_converters = [
            ContTwoQuadrantConverter(tau=tau, **kwargs),
            ContTwoQuadrantConverter(tau=tau, **kwargs),
            ContTwoQuadrantConverter(tau=tau, **kwargs),
        ]

    def reset(self):
        # Docstring in base class
        return [
            self._sub_converters[0].reset()[0] - 0.5,
            self._sub_converters[1].reset()[0] - 0.5,
            self._sub_converters[2].reset()[0] - 0.5,
        ]

    def convert(self, i_out, t):
        # Docstring in base class
        u_out = [
            self._sub_converters[0].convert([i_out[0]], t)[0] - 0.5,
            self._sub_converters[1].convert([i_out[1]], t)[0] - 0.5,
            self._sub_converters[2].convert([i_out[2]], t)[0] - 0.5
        ]
        return u_out

    def set_action(self, action, t):
        # Docstring in base class
        times = []
        times += self._sub_converters[0].set_action([0.5 * (action[0] + 1)], t)
        times += self._sub_converters[1].set_action([0.5 * (action[1] + 1)], t)
        times += self._sub_converters[2].set_action([0.5 * (action[2] + 1)], t)
        return sorted(list(set(times)))

    def i_sup(self, i_out):
        # Docstring in base class
        return sum([subconverter.i_sup([i_out_]) for subconverter, i_out_ in zip(self._sub_converters, i_out)])
