import gym
import numpy as np

from gym_electric_motor.physical_systems.converters.continuous_dynamically_averaged_control_set import ContDynamicallyAveragedConverter


class ContMultiConverter(ContDynamicallyAveragedConverter):
    """
    Converter that allows to include an arbitrary number of independent continuous sub-converters.
    Sub-converters must be 'elementary' and can not be MultiConverters.

    Key:
        'Cont-Multi'

    Actions:
        Concatenation of the subconverters' action spaces

    Action Space:
        Box([subconverter[0].action_space.low, subconverter[1].action_space.low, ...],
            [subconverter[0].action_space.high, subconverter[1].action_space.high, ...])

    Output Voltage Space:
        Box([subconverter[0].voltages.low, subconverter[1].voltages.low, ...],
            [subconverter[0].voltages.high, subconverter[1].voltages.high, ...])
    """

    def __init__(self, sub_converters, tau=1e-4, interlocking_time=0.0):
        """
        Args:
            sub_converters(Iterable[ContDynamicallyAveragedConverter]): Sub converters to merge into one for the overlying SCML system.
        """
        super().__init__(tau=tau, interlocking_time=interlocking_time)
        self._sub_converters = list(sub_converters)

        self.subsignal_current_space_dims = []
        self.subsignal_voltage_space_dims = []
        action_space_low = []
        action_space_high = []
        currents_low = []
        currents_high = []
        voltages_low = []
        voltages_high = []

        # get the limits and space dims from each subconverter
        for subconverter in self._sub_converters:
            self.subsignal_current_space_dims.append(np.squeeze(subconverter.currents.shape) or 1)
            self.subsignal_voltage_space_dims.append(np.squeeze(subconverter.voltages.shape) or 1)

            action_space_low.append(subconverter.action_space.low)
            action_space_high.append(subconverter.action_space.high)

            currents_low.append(subconverter.currents.low)
            currents_high.append(subconverter.currents.high)

            voltages_low.append(subconverter.voltages.low)
            voltages_high.append(subconverter.voltages.high)

        # convert to 1D list
        self.subsignal_current_space_dims = np.array(self.subsignal_current_space_dims)
        self.subsignal_voltage_space_dims = np.array(self.subsignal_voltage_space_dims)

        action_space_low = np.concatenate(action_space_low)
        action_space_high = np.concatenate(action_space_high)

        currents_low = np.concatenate(currents_low)
        currents_high = np.concatenate(currents_high)

        voltages_low = np.concatenate(voltages_low)
        voltages_high = np.concatenate(voltages_high)

        # put limits into gym_space format
        self.action_space = gym.spaces.Box(action_space_low, action_space_high, dtype=np.float64)
        

    def set_action(self, action, t):
        # Docstring in base class
        times = []
        ind = 0
        for subconverter in self._sub_converters:
            sub_action = action[ind:ind + subconverter.action_space.shape[0]]
            ind += subconverter.action_space.shape[0]
            times += subconverter.set_action(sub_action, t)
        return sorted(list(set(times)))

    def reset(self):
        # Docstring in base class
        u_in = []
        for subconverter in self._sub_converters:
            u_in += subconverter.reset()
        return u_in

    def convert(self, i_out, t):
        # Docstring in base class
        u_in = []
        subsignal_idx_low = 0
        for subconverter, subsignal_space_size in zip(self._sub_converters, self.subsignal_voltage_space_dims):
            subsignal_idx_high = subsignal_idx_low + subsignal_space_size
            u_in += subconverter.convert(i_out[subsignal_idx_low:subsignal_idx_high], t)
            subsignal_idx_low = subsignal_idx_high
        return u_in

    def _convert(self, i_in, t):
        # Not used
        pass

    def i_sup(self, i_out):
        # Docstring in base class
        i_sup = 0
        subsignal_idx_low = 0
        for subconverter, subsignal_space_size in zip(self._sub_converters, self.subsignal_current_space_dims):
            subsignal_idx_high = subsignal_idx_low + subsignal_space_size
            i_sup += subconverter.i_sup(i_out[subsignal_idx_low:subsignal_idx_high])
            subsignal_idx_low = subsignal_idx_high

        return i_sup

