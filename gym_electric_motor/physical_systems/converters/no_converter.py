import gym
import numpy as np

from gym_electric_motor.physical_systems.converters.power_electronic_converter import PowerElectronicConverter


class NoConverter(PowerElectronicConverter):
    """Dummy Converter class used to directly transfer the supply voltage to the motor"""
    
    action_space = gym.spaces.Box(low=np.array([]), high=np.array([]), dtype=np.float64)

    def i_sup(self, i_out):
        return i_out

    def convert(self, t, i_out, u_sup):
        return u_sup