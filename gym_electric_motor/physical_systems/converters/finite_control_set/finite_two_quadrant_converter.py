import gym
from gym_electric_motor.physical_systems.converters.finite_control_set import FiniteConverter


class FiniteTwoQuadrantConverter(FiniteConverter):
    """
    Key:
        'Finite-2QC'

    Switching States / Actions:
        | 0: Both Transistors off.
        | 1: Upper Transistor on.
        | 2: Lower Transistor on.

    Action Space:
        Discrete(3)

    """

    action_space = gym.spaces.Discrete(3)
    
    current_space = gym.spaces.Box(-1,1, shape=(1,))

    voltage_space = gym.spaces.Box(0,1, shape=(1,))


    def convert(self, t, i_out, u_sup):
        # Docstring in base class
        # Converter switches slightly (tau / 1000 seconds) before interlocking time due to inaccuracy of the solvers.
        if t - self._tau / 1000 > self._action_start_time + self._interlocking_time:
            self._switching_state = self._switching_pattern[-1]
        else:
            self._switching_state = self._switching_pattern[0]
        if self._switching_state == 0:
            if i_out[0] < 0:
                return u_sup
            elif i_out[0] >= 0:
                return [0.0]
        elif self._switching_state == 1:
            return u_sup
        elif self._switching_state == 2:
            return [0.0]
        else:
            raise Exception('Invalid switching state of the converter')

    def i_sup(self, i_out):
        # Docstring in base class
        if self._switching_state == 0:
            return i_out[0] if i_out[0] < 0 else 0
        elif self._switching_state == 1:
            return i_out[0]
        elif self._switching_state == 2:
            return 0
        else:
            raise Exception('Invalid switching state of the converter')

    def _set_switching_pattern(self, action):
        # Docstring in base class
        if (
                action == 0
                or self._switching_state == 0
                or action == self._switching_state
                or self._interlocking_time == 0
        ):
            self._switching_pattern = [action]
            return [self._action_start_time + self._tau]
        else:
            self._switching_pattern = [0, action]
            return [self._action_start_time + self._interlocking_time, self._action_start_time + self._tau]