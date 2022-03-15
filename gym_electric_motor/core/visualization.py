import gym_electric_motor as gem


class ElectricMotorVisualization(gem.core.Callback):
    """Base class for all visualizations in gym-electric-motor.

    The visualization is basically only a Callback that is extended by a render() function to update the figure.
    With the function calls that are inherited by the Callback superclass (e.g. *on_step_end*),
    the data is passed from the environment to the visualization. In the render() function the passed data can be
    visualized in the desired way.
    """

    def render(self):
        """Function to update the user interface."""
        raise NotImplementedError
