import numpy as np

from .dc_motor import DcMotor


class DcExternallyExcitedMotor(DcMotor):
    # Equals DC Base Motor
    HAS_JACOBIAN = True

    @property
    def observation_names(self):
       return ['i_a', 'i_e', 'u_a', 'u_e']

    @property
    def observation_units(self):
       return ['A', 'A', 'V', 'V']
