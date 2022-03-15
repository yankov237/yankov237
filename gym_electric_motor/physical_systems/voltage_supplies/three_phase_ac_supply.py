import numpy as np

from gym_electric_motor.physical_systems.voltage_supplies import ACSupply

class ThreePhaseACSupply(ACSupply):
    """AC three phase voltage supply"""


    def __init__(self, u_nominal=230, frequency=50, shape=(3,)):
        """
        Args:
        """
        super().__init__(u_nominal, frequency, shape)
        assert len(self.supply_shape) == 1, 'The Three Phase AC supply does not support multidimensional shapes.'
        assert self.supply_shape[-1] % 3 == 0, \
            'The last dimension of the supply shape has to be a multiple of three.'
        
        phi_init = np.zeros(self.supply_shape, dtype=float)
        first_phase_slice = slice(0, shape[0], 3)
        second_phase_slice = slice(1, shape[0], 3)
        third_phase_slice = slice(2, shape[0], 3)

        def reset_phi_three_phase(*_):
            phi_init[first_phase_slice] = np.random.rand(self.supply_shape[0] / 3)
            temp = phi_init[first_phase_slice]
            phi_init[second_phase_slice] = temp + 1/3
            phi_init[third_phase_slice] = temp + 2/3
            return 2 * np.pi * phi_init

        self._reset_phi = reset_phi_three_phase
