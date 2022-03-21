import gym
import numpy as np

from gym_electric_motor.physical_systems.converters.continuous_dynamically_averaged_control_set\
    .continuous_dynamically_averaged_converter import ContDynamicallyAveragedConverter


class ContOneQuadrantConverter(ContDynamicallyAveragedConverter):
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

    def __init__(self, shape=(1,), **kwargs):
        super().__init__(**kwargs)
        self._u_out = np.zeros(shape)

    def convert(self, t, i_in, u_sup):
        # Docstring in base class
        return np.where(i_in > 0, self._current_action * u_sup, u_sup)

    def i_sup(self, i_out):
        # Docstring in base class
        return self._current_action * i_out
