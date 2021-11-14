import numpy as np

import gym_electric_motor as gem
from gym_electric_motor.visualization.motor_dashboard_plots import StatePlot, TimePlot


class FluxObserverPlot(StatePlot):
    """Plot that displays a flux observer state (either psi_abs or psi_angle).

        Usage Example
        -------------
        .. code-block:: python
            env = gem.make(
                'DqCont-SC-SCIM-v0',
            )
            # Wrap the environment with the flux observer
            env = FluxObserver(env)

            # Generate the plot instances
            psi_abs_plot = FluxObserverPlot(env, 'psi_abs')
            psi_angle_plot = FluxObserverPlot(env, 'psi_angle')

            # Add the plots to the Dashboard
            env.visualization.add_plot(psi_abs_plot)
            env.visualization.add_plot(psi_angle_plot)

            # Run the environment as usual
            # ...
    """

    def __init__(self, flux_observer, state: str):
        assert state in ['psi_abs', 'psi_angle'], \
            f'The state has to be either "psi_abs" or "psi_angle. The passed state was {state}'
        self._flux_observer = flux_observer
        self.state_labels['psi_abs'] = r'$\hat{\Psi}_{r}$|/Vs'
        self.state_labels['psi_angle'] = r'$\measuredangle\hat{\Psi}_r$/rad'

        super().__init__(state)

    def set_env(self, env):
        # Docstring of superclass
        TimePlot.set_env(self, self._flux_observer)

        # Save the index of the state.
        self._state_idx = self._flux_observer.state_names.index(self.state)
        # The maximal values of the state.
        self._limits = self._flux_observer.limits[self._state_idx]
        self._state_space = self._flux_observer.observation_space[0].low[self._state_idx], \
            self._flux_observer.observation_space[0].high[self._state_idx]

        # Bool: if the state is referenced.
        self._referenced = self._flux_observer.referenced_states[self._state_idx]
        # Bool: if the data is already normalized to an interval of [-1, 1]
        self._normalized = self._limits != self._state_space[1]
        self.reset_data()

        min_limit = self._limits * self._state_space[0] if self._normalized else self._state_space[0]
        max_limit = self._limits * self._state_space[1] if self._normalized else self._state_space[1]
        spacing = 0.1 * (max_limit - min_limit)

        # Set the y-axis limits to fixed initial values
        self._y_lim = (min_limit - spacing, max_limit + spacing)

        # Set the y-axis label
        self._label = self.state_labels.get(self._state, self._state)

    def on_step_end(self, k, state, reference, reward, done):
        state_ = np.concatenate(
            (state, [self._flux_observer.psi_abs / self._flux_observer.psi_max, self._flux_observer.psi_angle / np.pi])
        )
        reference_ = np.concatenate((reference, np.array([0.0,0.0])))
        super().on_step_end(k, state_, reference_, reward, done)
