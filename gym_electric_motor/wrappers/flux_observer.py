import gym
import gym_electric_motor as gem
import numpy as np


class FluxObserver(gym.ObservationWrapper):

    @property
    def psi_abs(self):
        return np.abs(self._integrated)

    @property
    def psi_angle(self):
        return np.angle(self._integrated)

    @property
    def state_names(self):
        return self.env.state_names + ['psi_abs', 'psi_angle']

    @property
    def referenced_states(self):
        return np.concatenate((self.env.referenced_states, [False, False]))

    @property
    def psi_abs(self):
        return np.abs(self._integrated)

    @property
    def psi_angle(self):
        return np.angle(self._integrated)

    @property
    def psi_max(self):
        return self._psi_max

    @property
    def limits(self):
        return np.concatenate((self.env.limits, [self._psi_max, np.pi]))

    @property
    def nominal_state(self):
        return np.concatenate(self.env.nominal_state, np.array([self._psi_max, np.pi]))

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.physical_system.electrical_motor, gem.physical_systems.electric_motors.InductionMotor),\
            'The FluxObserver needs an Environment for Induction motors (SCIM or DFIM).'
        mp = env.physical_system.electrical_motor.motor_parameter
        self._l_m = mp['l_m']  # Main induction
        self._l_r = mp['l_m'] + mp['l_sigr']  # Induction of the rotor
        self._r_r = mp['r_r']  # Rotor resistance
        self._p = mp['p']  # Pole pair number
        self._psi_max = 100.0  # Todo: Calculate correct Psi max
        state_space = env.observation_space[0]
        state_space = gem.utils.concatenate_boxes(state_space, gym.spaces.Box(-1., 1, shape=(2,), dtype=np.float64))
        self.observation_space = gym.spaces.Tuple((state_space, env.observation_space[1]))
        # function to transform the currents from abc to alpha/beta coordinates
        self._abc_to_alphabeta_transformation = env.physical_system.abc_to_alphabeta_space

        # Integrated values of the flux for the two directions (Re: alpha, Im: beta)
        self._integrated = np.complex(0, 0)
        self._i_s_idx = [env.state_names.index('i_sa'), env.state_names.index('i_sb'), env.state_names.index('i_sc')]
        self._omega_idx = env.state_names.index('omega')

    def observation(self, observation):
        state = observation[0] * self.env.limits
        i_s = state[self._i_s_idx]
        omega_me = state[self._omega_idx] * self._p

        # Transform current into alpha, beta coordinates
        [i_s_alpha, i_s_beta] = self._abc_to_alphabeta_transformation(i_s)

        # Calculate delta flux
        delta = np.complex(i_s_alpha, i_s_beta) * self._r_r * self._l_m / self._l_r \
            - self._integrated * np.complex(self._r_r / self._l_r, -omega_me)

        # Integrate the flux
        self._integrated += delta * self.env.physical_system.tau
        return np.concatenate((state, [self.psi_abs / self._psi_max, self.psi_angle / np.pi])), observation[1]

    def reset(self):
        # Reset the integrated value
        self._integrated = np.complex(0, 0)
        return super().reset()
