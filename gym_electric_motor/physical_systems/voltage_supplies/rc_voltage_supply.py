import warnings
import numpy as np

from gym_electric_motor.physical_systems.ode_solvers.solvers import EulerSolver
from gym_electric_motor.physical_systems.voltage_supplies import VoltageSupply


class RCVoltageSupply(VoltageSupply):
    """DC voltage supply modeled as RC element"""
    
    def __init__(self, u_nominal=600.0, supply_parameter=None):
        """This Voltage Supply is a model of a non ideal voltage supply.
        The ideal voltage source U_0 is part of an RC element.
        
        Args: 
            supply_parameter(dict): Consists or Resistance R in Ohm and Capacitance C in Farad
            
        Additional notes:
            If the product of R and C get too small the numerical stability of the ODE is not given anymore
            typical time differences tau are only in the range of 10e-3. One might want to consider R*C as a
            time constant. The resistance R can be considered as a simplified inner resistance model.
        """
        super().__init__(u_nominal)
        supply_parameter = supply_parameter or {'R': 1, 'C': 4e-3}
        # Supply range is between 0 - capacitor completely unloaded - and u_nominal - capacitor is completely loaded
        assert 'R' in supply_parameter.keys(), "Pass key 'R' for Resistance in your dict"
        assert 'C' in supply_parameter.keys(), "Pass key 'C' for Capacitance in your dict"
        self.supply_range = (0,u_nominal) 
        self._r = supply_parameter['R']
        self._c = supply_parameter['C']
        if self._r*self._c < 1e-4:
            warnings.warn(
                "The product of R and C might be too small for the correct calculation of the supply voltage. "
                "You might want to consider R*C as a time constant."
            )
        self._u_sup = [u_nominal]
        self._u_0 = u_nominal
        self._solver = EulerSolver()
        self._solver.set_system_equation(self.system_equation)
        
    def system_equation(self, t, u_sup, u_0, i_sup, r, c):
        # ODE for derivate of u_sup
        return np.array([(u_0 - u_sup[0] - r*i_sup)/(r*c)])

    def reset(self):
        # Docstring of superclass
        # On reset the capacitor is loaded again
        self._solver.set_initial_value(np.array([self._u_0]))
        self._u_sup = [self._u_0]
        return self._u_sup
    
    def get_voltage(self, t, i_sup):
        # Docstring of superclass
        self._solver.set_f_params(self._u_0, i_sup, self._r, self._c)
        self._u_sup = self._solver.integrate(t)
        return self._u_sup
