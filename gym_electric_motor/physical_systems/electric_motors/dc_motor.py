import gym
import numpy as np

from .electric_motor import ElectricMotor


class DcMotor(ElectricMotor):
    """
        The DcMotor and its subclasses implement the technical system of a dc motor.

        This includes the system equations, the motor parameters of the equivalent circuit diagram,
        as well as limits.

        =====================  ==========  ============= ===========================================
        Motor Parameter        Unit        Default Value Description
        =====================  ==========  ============= ===========================================
        r_a                    Ohm         16e-3         Armature circuit resistance
        r_e                    Ohm         16e-2         Exciting circuit resistance
        l_a                    H           19e-6         Armature circuit inductance
        l_e                    H           5.4e-3        Exciting circuit inductance
        l_e_prime              H           1.7e-3        Effective excitation inductance
        j_rotor                kg/m^2      0.025         Moment of inertia of the rotor
        =====================  ==========  ============= ===========================================

        =============== ====== =============================================
        Motor Currents  Unit   Description
        =============== ====== =============================================
        i_a             A      Armature circuit current
        i_e             A      Exciting circuit current
        =============== ====== =============================================
        =============== ====== =============================================
        Motor Voltages  Unit   Description
        =============== ====== =============================================
        u_a             V      Armature circuit voltage
        u_e             v      Exciting circuit voltage
        =============== ====== =============================================

        ======== ===========================================================
        Limits / Nominal Value Dictionary Entries:
        -------- -----------------------------------------------------------
        Entry    Description
        ======== ===========================================================
        i_a      Armature current
        i_e      Exciting current
        omega    Angular Velocity
        torque   Motor generated torque
        u_a      Armature Voltage
        u_e      Exciting Voltage
        ======== ===========================================================
    """

    # Indices for array accesses
    I_A_IDX = 0
    I_E_IDX = 1

    # Motor parameter, nominal values and limits are based on the following DC Motor:
    # https://www.heinzmann-electric-motors.com/en/products/dc-motors/pmg-132-dc-motor
    _default_motor_parameter = {
        'r_a': 16e-3, 'r_e': 16e-2, 'l_a': 19e-6, 'l_e_prime': 1.7e-3, 'l_e': 5.4e-3, 'j_rotor': 0.0025
    }
    _default_nominal_values = dict(omega=300, torque=16.0, i=97, i_a=97, i_e=97, u=60, u_a=60, u_e=60)
    _default_limits = dict(omega=400, torque=38.0, i=210, i_a=210, i_e=210, u=60, u_a=60, u_e=60)

    @property
    def observation_names(self):
        return ['i_a', 'i_e', 'u_a', 'u_e']
    
    @property
    def motor_state_size(self):
        return 2

    @property
    def observation_space(self):
        return gym.spaces.Box(-1, 1, shape=(4,))

    def __init__(self, motor_parameter=None, nominal_values=None, limit_values=None, motor_initializer=None):
        # Docstring of superclass
        super().__init__(motor_parameter, nominal_values, limit_values, motor_initializer)
        #: Matrix that contains the constant parameters of the systems equation for faster computation
        self._model_constants = None
        self._model_variables = None
        self._update_model()
        self._update_limits()

    def _update_model(self):
        """Updates the motor's model parameters with the motor parameters.

        Called internally when the motor parameters are changed or the motor is initialized.
        """
        mp = self._motor_parameter
        self._model_constants = np.array(
            [
                [-mp['r_a'], 0, -mp['l_e_prime'], 1, 0],
                [0, -mp['r_e'], 0, 0, 1]
            ]
        )
        self._model_constants[self.I_A_IDX] = self._model_constants[self.I_A_IDX] / mp['l_a']
        self._model_constants[self.I_E_IDX] = self._model_constants[self.I_E_IDX] / mp['l_e']
        self._model_variables = np.zeros(self._model_constants.shape[-1])

    def torque(self, motor_state):
        # Docstring of superclass
        return self._motor_parameter['l_e_prime'] * motor_state[self.I_A_IDX] * motor_state[self.I_E_IDX]

    def i_in(self, motor_state):
        # Docstring of superclass
        return list(motor_state)

    def electrical_ode(self, motor_state, omega):
        # Docstring of superclass
        self._model_variables[:] = [
            motor_state[self.I_A_IDX],
            motor_state[self.I_E_IDX],
            omega * motor_state[self.I_E_IDX],
            self._u_in[0],
            self._u_in[1],
        ]
        return np.matmul(
            self._model_constants,
            self._model_variables
        )

    def electrical_jacobian(self, state, omega):
        mp = self._motor_parameter
        return (
            np.array([
                [-mp['r_a'] / mp['l_a'], -mp['l_e_prime'] / mp['l_a'] * omega],
                [0, -mp['r_e'] / mp['l_e']]
            ]),
            np.array([-mp['l_e_prime'] * state[self.I_E_IDX] / mp['l_a'], 0]),
            np.array([mp['l_e_prime'] * state[self.I_E_IDX],
                      mp['l_e_prime'] * state[self.I_A_IDX]])
        )

    def _update_limits(self, limits_d=None, nominal_d=None):
        # Docstring of superclass
        if limits_d is None:
            limits_d = dict()

        # torque is replaced the same way for all DC motors
        limits_d.update(dict(torque=self.torque([self._limits[state] for state in self.CURRENTS])))
        super()._update_limits(limits_d)
