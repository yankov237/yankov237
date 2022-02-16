import numpy as np
import warnings

import gym_electric_motor as gem
from ..random_component import RandomComponent
from ..core import PhysicalSystem


class SCMLSystem(PhysicalSystem, RandomComponent):
    """
    The SCML(Supply-Converter-Motor-Load)-System is used for the simulation of
    a technical setting consisting of these components and a solver for the electrical ODE of the motor and mechanical
    ODE of the load.
    """

    @property
    def initial_ode_value(self):
        return self._initial_ode_value

    @initial_ode_value.setter
    def initial_ode_value(self, value):
        self._initial_ode_value[:] = value

    @property
    def limits(self):
        return self._limits

    @property
    def nominal_state(self):
        return self._nominal_state

    @property
    def supply(self):
        """The voltage supply instance in the physical system"""
        return self._supply

    @property
    def converter(self):
        """The power electronic converter instance in the system"""
        return self._converter

    @property
    def electric_motor(self):
        """The electrical motor instance of the system"""
        return self._electric_motor

    @property
    def mechanical_load(self):
        """The mechanical load instance in the system"""
        return self._mechanical_load

    def __init__(self, converter, motor, load, supply, ode_solver, tau=1e-4, calc_jacobian=None):
        """
        Args:
            converter(PowerElectronicConverter): Converter for the system
            motor(ElectricMotor): Motor of the system
            load(MechanicalLoad): Mechanical Load of the System
            supply(VoltageSupply): Voltage Supply
            ode_solver(OdeSolver): Ode Solver to use in this setting
            tau(float): discrete time step of the system
            calc_jacobian(bool): If True, the jacobian matrices will be taken into account for the ode-solvers.
                Default: The jacobians are used, if available
        """
        RandomComponent.__init__(self)
        self._converter = converter
        self._electric_motor = motor
        self._mechanical_load = load
        self._supply = supply
        state_names = load.state_names + motor.state_names + converter.state_names + supply.state_names
        self._ode_solver = ode_solver
        self._motor_state = None
        self._load_state = None
        if calc_jacobian is None:
            calc_jacobian = self._electric_motor.HAS_JACOBIAN and self._mechanical_load.HAS_JACOBIAN
        if calc_jacobian and self._electric_motor.HAS_JACOBIAN and self._mechanical_load.HAS_JACOBIAN:
            jac = self._system_jacobian
        else:
            jac = None
        if calc_jacobian and jac is None:
            warnings.warn('Jacobian Matrix is not provided for either the motor or the load Model')

        self._ode_solver.set_system_equation(self._system_equation, jac)
        self._mechanical_load.set_j_rotor(self._electric_motor.motor_parameter['j_rotor'])
        self._t = 0
        self._set_indices()
        state_space = self._build_state_space(state_names)
        super().__init__(self._converter.action_space, state_space, state_names, tau)
        self._limits = np.zeros_like(state_names, dtype=float)
        self._nominal_state = np.zeros_like(state_names, dtype=float)
        self._set_limits()
        self._set_nominal_state()
        self.system_state = np.zeros_like(state_names, dtype=float)
        self._system_derivative = np.zeros((len(self._mechanical_load.state_names) + len(self._electric_motor.state_names)))
        self._components = [
            self._supply, self._converter, self._electric_motor, self._mechanical_load, self._ode_solver
        ]

    def _set_limits(self):
        """Method to set the physical limits from the components."""
        for ind, state in enumerate(self._state_names):
            motor_lim = self._electric_motor.limits.get(state, np.inf)
            mechanical_lim = self._mechanical_load.limits.get(state, np.inf)
            self._limits[ind] = min(motor_lim, mechanical_lim)
        self._limits[self._state_positions['u_sup']] = self.supply.u_nominal

    def _set_nominal_state(self):
        """Method to set the nominal values from the components."""
        for ind, state in enumerate(self._state_names):
            motor_nom = self._electric_motor.nominal_values.get(state, np.inf)
            mechanical_nom = self._mechanical_load.nominal_values.get(state, np.inf)
            self._nominal_state[ind] = min(motor_nom, mechanical_nom)
        self._nominal_state[self._state_positions['u_sup']] = self.supply.u_nominal

    def seed(self, seed=None):
        RandomComponent.seed(self, seed)
        sub_seeds = self.seed_sequence.spawn(len(self._components))
        for component, sub_seed in zip(self._components, sub_seeds):
            if isinstance(component, gem.RandomComponent):
                component.seed(sub_seed)

    def simulate(self, action):
        # Docstring of superclass        
        switching_times = self._converter.set_action(action, self._t)

        for t in switching_times:
            i_in = self._electric_motor.i_in(self._motor_state)
            i_sup = self._converter.i_sup(i_in)
            u_sup = self._supply.get_voltage(self._t, i_sup)
            u_in = self._converter.convert(i_in, self._ode_solver.t)
            u_in = [u * u_s for u in u_in for u_s in u_sup]
            self._electric_motor.set_voltage(u_in)
            ode_state = self._ode_solver.integrate(t)
        
        self._motor_state = ode_state[self._motor_state_slice]
        self._load_state = ode_state[self._load_state_slice]
        self._count_step()
        self.system_state = np.concatenate((
            self._mechanical_load.get_observation(self._load_state),
            self._electric_motor.get_observation(self._motor_state),
            self._converter.get_observation()
            self._supply.get_observation(u_sup, i_sup)
        ))
        return self.system_state / self._limits

    def _system_equation(self, t, state, u_in, **__):
        """
        Systems differential equation system.

        It is a concatenation of the motors electrical ode system and the mechanical ode system.

        Args:
            t(float): Current system time
            state(ndarray(float)): Current systems ODE-State
            u_in(list(float)): Input voltages from the converter

        Returns:
            ndarray(float): The derivatives of the ODE-State. Based on this, the Ode Solver calculates the next state.
        """
        if self._system_eq_placeholder is None:
            motor_state = state[self._motor_ode_idx]
            motor_derivative = self._electric_motor.electrical_ode(
                motor_state, u_in, state[self._omega_ode_idx]
            )
            torque = self._electric_motor.torque(motor_state)
            load_derivative = self._mechanical_load.mechanical_ode(
                t, state[self._load_ode_idx], torque
            )
            self._system_eq_placeholder = np.concatenate((load_derivative,
                                                          motor_derivative))
            self._motor_deriv_size = motor_derivative.size
            self._load_deriv_size = load_derivative.size
        else:
            motor_state = state[self._motor_ode_idx]
            self._system_eq_placeholder[:self._load_deriv_size] = \
                self._mechanical_load.mechanical_ode(
                    t, state[self._load_ode_idx],
                    self._electric_motor.torque(motor_state)
                ).ravel()
            self._system_eq_placeholder[self._load_deriv_size:] = \
                self._electric_motor.electrical_ode(
                    motor_state, u_in, state[self._omega_ode_idx]
                ).ravel()

        return self._system_eq_placeholder

    def _system_jacobian(self, t, state, u_in, **__):
        motor_state = state[self._motor_ode_idx]
        motor_jac, el_state_over_omega, torque_over_el_state = self._electric_motor.electrical_jacobian(
            motor_state, u_in, state[self._omega_ode_idx]
        )
        torque = self._electric_motor.torque(motor_state)
        load_jac, load_over_torque = self._mechanical_load.mechanical_jacobian(
            t, state[self._load_ode_idx], torque
        )
        system_jac = np.zeros((state.shape[0], state.shape[0]))
        system_jac[:load_jac.shape[0], :load_jac.shape[1]] = load_jac
        system_jac[-motor_jac.shape[0]:, -motor_jac.shape[1]:] = motor_jac
        system_jac[-motor_jac.shape[0]:, [self._omega_ode_idx]] = el_state_over_omega.reshape((-1, 1))
        system_jac[:load_jac.shape[0], load_jac.shape[1]:] = np.matmul(
            load_over_torque.reshape(-1, 1), torque_over_el_state.reshape(1, -1)
        )
        return system_jac

    def reset(self):
        """
        Reset all the systems components to an initial state.

        Returns:
             The new state of the system.
        """
        self.next_generator()
        self._motor_state = self._electric_motor.reset(
            state_space=self.state_space,
            state_positions=self.state_positions)
        self._load_state = self._mechanical_load.reset(
            state_space=self.state_space,
            state_positions=self.state_positions,
            nominal_state=self.nominal_state)
        ode_state = np.concatenate((self._load_state, self._motor_state)) 
        u_sup = self.supply.reset()
        u_in = self.converter.reset()
        u_in = [u * u_s for u in u_in for u_s in u_sup]
        torque = self.electric_motor.torque(self._motor_state)
        self._t = 0
        self._k = 0
        self._ode_solver.set_initial_value(ode_state, self._t)
        system_state = np.concatenate((
            self._mechanical_load.get_observation(self._load_state),
            self._electric_motor.get_observation(self._motor_state),
            self._converter.get_observation(),
            self._supply.get_observation()
        ))
        return system_state / self._limits