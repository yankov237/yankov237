import gym
from gym_electric_motor.physical_systems.converters.finite_control_set import FiniteConverter, FiniteTwoQuadrantConverter


class FiniteB6BridgeConverter(FiniteConverter):
    """
    The finite B6 bridge converters (B6C) is simulated with three finite 2QC.

    Key:
        'Finite-B6C'

    Actions:
        +-+-----+-----+-----+
        | |H_1  |H_2  |H_3  |
        +=+=====+=====+=====+
        |0|lower|lower|lower|
        +-+-----+-----+-----+
        |1|lower|lower|upper|
        +-+-----+-----+-----+
        |2|lower|upper|lower|
        +-+-----+-----+-----+
        |3|lower|upper|upper|
        +-+-----+-----+-----+
        |4|upper|lower|lower|
        +-+-----+-----+-----+
        |5|upper|lower|upper|
        +-+-----+-----+-----+
        |6|upper|upper|lower|
        +-+-----+-----+-----+
        |7|upper|upper|upper|
        +-+-----+-----+-----+

    Action Space:
        Discrete(8)

    Output Voltages and Currents:
        | voltages: Box(-1,1, shape=(3,))
        | currents: Box(-1,1, shape=(3,))

    Output Voltage Space:
        Box(-0.5, 0.5, shape=(3,))
    """

    action_space = gym.spaces.Discrete(8)
    
    current_space = gym.spaces.Box(-1, 1, shape=(3,))

    voltage_space = gym.spaces.Box(-1, 1, shape=(3,))


    _reset_action = 0
    _subactions = [
        [2, 2, 2],
        [2, 2, 1],
        [2, 1, 2],
        [2, 1, 1],
        [1, 2, 2],
        [1, 2, 1],
        [1, 1, 2],
        [1, 1, 1]
    ]

    def __init__(self, tau=1e-5, **kwargs):
        # Docstring in base class
        super().__init__(tau=tau, **kwargs)
        self._sub_converters = [
            FiniteTwoQuadrantConverter(tau=tau, **kwargs),
            FiniteTwoQuadrantConverter(tau=tau, **kwargs),
            FiniteTwoQuadrantConverter(tau=tau, **kwargs),
        ]

    def reset(self):
        # Docstring in base class
        for sub_conv in self._sub_converters:
            sub_conv.reset()

    def convert(self, t, i_out, u_sup):
        # Docstring in base class
        u_out = [
            self._sub_converters[0].convert([i_out[0]], t)[0] - 0.5,
            self._sub_converters[1].convert([i_out[1]], t)[0] - 0.5,
            self._sub_converters[2].convert([i_out[2]], t)[0] - 0.5
        ]
        return u_out

    def set_action(self, action, t):
        # Docstring in base class
        assert self.action_space.contains(action), \
            f"The selected action {action} is not a valid element of the action space {self.action_space}."
        subactions = self._subactions[action]
        times = set()
        times.update(self._sub_converters[0].set_action(subactions[0], t))
        times.update(self._sub_converters[1].set_action(subactions[1], t))
        times.update(self._sub_converters[2].set_action(subactions[2], t))
        return sorted(times)

    def i_sup(self, i_out):
        # Docstring in base class
        return sum([subconverter.i_sup([i_out_]) for subconverter, i_out_ in zip(self._sub_converters, i_out)])
