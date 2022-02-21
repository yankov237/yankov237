import gym
from gym_electric_motor.physical_systems.converters.finite_control_set import FiniteConverter
import gym_electric_motor as gem


class FiniteMultiConverter(FiniteConverter):
    """
    Converter that allows to include an arbitrary number of independent finite subconverters.
    Subconverters must be 'elementary' and can not be MultiConverters.

    Key:
        'Finite-Multi'

    Actions:
        Concatenation of the subconverters' action spaces

    Action Space:
        MultiDiscrete([subconverter[0].action_space.n , subconverter[1].action_space.n, ...])

    Output Voltage Space:
        Box([subconverter[0].voltages.low, subconverter[1].voltages.low, ...],
            [subconverter[0].voltages.high, subconverter[1].voltages.high, ...])
    """

    @property
    def sub_converters(self):
        return self._sub_converters

    def __init__(self, sub_converters, tau=1e-5, interlocking_time=0.0):
        """
        Args:
            subconverters(FiniteConverter): Converters to be combined as one for the SCML system
        """
        super().__init__(interlocking_time=interlocking_time, tau=tau)
        self._sub_converters = sub_converters

        # put limits into gym_space format
        self.action_space = gym.spaces.MultiDiscrete(converter.action_space.n for converter in sub_converters)
        self.voltage_space = gem.utils.concatenate_boxes([sub_conv.voltage_space for sub_conv in sub_converters])
        self.current_space = gem.utils.concatenate_boxes([sub_conv.current_space for sub_conv in sub_converters])

        self._sub_converter_slices = []
        i_start = 0
        for sub_converter in sub_converters:
            self._sub_converter_slices.append(slice(i_start, i_start + sub_converter.voltage_space.shape[0]))
            i_start = self._sub_converter_slices[-1].stop

    def convert(self, t, i_out, u_sup):
        # Docstring in base class
        u_in = []
        if len(u_sup) == 1:
          u_sup = u_sup * len(self._sub_converters)
        
        for subconverter, sub_slice, u_sup_ in zip(self._sub_converters, self._sub_converter_slices, u_sup):
            u_in += subconverter.convert(t, i_out[sub_slice], u_sup_)
        return u_in

    def reset(self):
        # Docstring in base class
        for subconverter in self._sub_converters:
            subconverter.reset()

    def set_action(self, action, t):
        # Docstring in base class
        times = set()
        for subconverter, sub_action in zip(self._sub_converters, action):
            times.update(subconverter.set_action(sub_action, t))
        return sorted(times)

    def i_sup(self, i_out):
        # Docstring in base class
        i_sup = 0
        subsignal_idx_low = 0
        for subconverter, sub_slice in zip(self._sub_converters, self._sub_converter_slices):
            i_sup += subconverter.i_sup(i_out[sub_slice])
        return i_sup