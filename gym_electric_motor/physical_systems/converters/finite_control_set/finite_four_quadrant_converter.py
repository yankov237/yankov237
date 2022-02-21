import gym
from gym_electric_motor.physical_systems.converters.finite_control_set import FiniteConverter, FiniteTwoQuadrantConverter


class FiniteFourQuadrantConverter(FiniteConverter):
    """
    Key:
        'Finite-4QC'

    Switching States / Actions:
        | 0: T2, T4 on.
        | 1: T1, T4 on.
        | 2: T2, T3 on.
        | 3: T1, T3 on.

    Action Space:
        Discrete(4)

    """

    action_space = gym.spaces.Discrete(4)
    
    current_space = gym.spaces.Box(0,1, shape=(1,))

    voltage_space = gym.spaces.Box(0,1, shape=(1,))

    action_space = gym.spaces.Discrete(4)

    def __init__(self, **kwargs):
        # Docstring in base class
        super().__init__(**kwargs)
        self._subconverters = [FiniteTwoQuadrantConverter(**kwargs), FiniteTwoQuadrantConverter(**kwargs)]

    def reset(self):
        # Docstring in base class
        self._subconverters[0].reset()
        self._subconverters[1].reset()

    def convert(self, t, i_out, u_sup):
        # Docstring in base class
        return [self._subconverters[0].convert(t, i_out, u_sup)[0] - self._subconverters[1].convert(t, [-i_out[0]], u_sup)[0]]

    def set_action(self, action, t):
        # Docstring in base class
        assert self.action_space.contains(action), \
            f"The selected action {action} is not a valid element of the action space {self.action_space}."
        times = set()
        action0 = [1, 1, 2, 2][action]
        action1 = [1, 2, 1, 2][action]
        times.update(self._subconverters[0].set_action(action0, t))
        times.update(self._subconverters[1].set_action(action1, t))
        return sorted(times)

    def i_sup(self, i_out):
        # Docstring in base class
        return self._subconverters[0].i_sup(i_out) + self._subconverters[1].i_sup([-i_out[0]])