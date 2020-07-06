import gym_electric_motor as gem
import gym_electric_motor.physical_systems.voltage_supplies as vs
from ..testing_utils import DummyOdeSolver
import numpy as np
import pytest

class TestVoltageSupply:

    key = ''
    class_to_test = vs.VoltageSupply

    def test_registered(self):
        """If a key is provided, tests if the class can be initialized from registry"""
        if self.key != '':
            supply = gem.utils.instantiate(vs.VoltageSupply, self.key)
            assert type(supply) == self.class_to_test

    def test_initialization(self):
        """Test the initalization and correct setting of values"""
        u_nominal = 600.0
        voltage_supply = self.class_to_test(u_nominal)
        assert voltage_supply.u_nominal == u_nominal


class TestIdealVoltageSupply(TestVoltageSupply):

    key = 'IdealVoltageSupply'
    class_to_test = vs.IdealVoltageSupply

    def test_default_initialization(self):
        """Test for default initialization values"""
        voltage_supply = vs.IdealVoltageSupply()
        assert voltage_supply._u_nominal == 600.0
        assert voltage_supply.supply_range == (600.0, 600.0)

    def test_get_voltage(self, u_nominal=450.0):
        """Test the get voltage function.
            It must return u_nominal."""
        supply = vs.IdealVoltageSupply(u_nominal)
        assert supply.get_voltage() == [u_nominal]

    def test_reset(self, u_nominal=450.0):
        """Test the reset function.
            It must return u_nominal."""
        supply = vs.IdealVoltageSupply(u_nominal)
        assert supply.reset() == [u_nominal]
        
class TestRCVoltageSupply(TestVoltageSupply):
    
    key = 'RCVoltageSupply'
    class_to_test = vs.RCVoltageSupply
    
    def test_default_initialization(self):
        """Test for default initialization values"""
        voltage_supply = vs.RCVoltageSupply()
        assert voltage_supply._u_0 == 600.0
        assert voltage_supply._u_sup == [600.0]
        assert voltage_supply.supply_range == (0.0, 600.0)
        assert voltage_supply._r == 1
        assert voltage_supply._c == 4e-3
        
    def test_reset(self, u_nominal=450.0):
        """Test the reset function.
            It must return u_nominal."""
        voltage_supply = vs.RCVoltageSupply(u_nominal)
        assert voltage_supply.reset() == [u_nominal]

        
    def test_initialization(self, u_nominal=450.0,supply_parameter={'R':3,'C':6e-2}):
        voltage_supply = vs.RCVoltageSupply(u_nominal, supply_parameter)
        assert voltage_supply._u_0 == u_nominal
        assert voltage_supply._u_sup == [u_nominal]
        assert voltage_supply.supply_range == (0.0, u_nominal) 
        assert voltage_supply._r == supply_parameter['R']
        assert voltage_supply._c == supply_parameter['C']
        
    def test_get_voltage(self, monkeypatch, u_nominal=450.0):
        """Test the get voltage function.
            It must return the right voltage added by its change given by the time delta in this example."""
        supply = vs.RCVoltageSupply(u_nominal)
        solver = DummyOdeSolver()
        solver._y = np.array([u_nominal])
        monkeypatch.setattr(supply, '_solver', solver)
        times = [0.5,1,1.5,1.78,2.1,3]
        for time in times:
            assert supply.get_voltage(time,0) == [u_nominal + time]
            assert supply._u_sup == u_nominal + time
            
    def test_system_equation(self):
         """Tests the correct behavior of the system equation by hand calculated values"""
         supply = vs.RCVoltageSupply()
         system_equation = supply.system_equation
         assert system_equation(0,[10],50,1,1,1) == 39
         assert system_equation(0,[20],50,2,2,2) == 6.5
         assert system_equation(0,[30],50,3,3,3) == 1 + 2/9
         #time invariance
         assert system_equation(0,[30],50,3,3,3) == system_equation(5,[30],50,3,3,3)     

class TestAC1PhaseSupply(TestVoltageSupply):
    key = 'AC1PhaseSupply'
    class_to_test = vs.AC1PhaseSupply
    
    def test_default_initialization(self):
        """Test for default initialization values"""
        voltage_supply = vs.AC1PhaseSupply()
        assert voltage_supply.u_nominal == 230.0
        assert voltage_supply._max_amp == 230.0 * np.sqrt(2)
        assert voltage_supply.supply_range == [-230.0 * np.sqrt(2), 230.0 * np.sqrt(2)]
        assert voltage_supply._fixed_phi == False
        assert voltage_supply._f == 50
        
    def test_reset(self, u_nominal=230.0):
        """Test the reset function for correct behavior on fixed phase""" 
        supply_parameter = {'frequency': 50, 'phase': 0, 'fixed_phase': False}
        voltage_supply = vs.AC1PhaseSupply(u_nominal, supply_parameter)
        assert voltage_supply._phi == 0
        _ = voltage_supply.reset()
        assert voltage_supply._phi != 0, "Test this again and if this error doesn't appear next time you should consider playing lotto"

        supply_parameter = {'frequency': 50, 'phase': 0, 'fixed_phase': True}
        voltage_supply = vs.AC1PhaseSupply(u_nominal, supply_parameter)
        assert voltage_supply._phi == 0
        assert voltage_supply.reset() == [0.0]
        assert voltage_supply._phi == 0

        
    def test_initialization(self, u_nominal = 300.0):
        supply_parameter = {'frequency': 35, 'phase': 0, 'fixed_phase': True}
        voltage_supply = vs.AC1PhaseSupply(u_nominal, supply_parameter)
        assert voltage_supply.u_nominal == 300.0
        assert voltage_supply._max_amp == 300.0 * np.sqrt(2)
        assert voltage_supply.supply_range == [-300.0 * np.sqrt(2), 300.0 * np.sqrt(2)]
        assert voltage_supply._fixed_phi == True
        assert voltage_supply._phi == 0
        assert voltage_supply._f == 35
        
    def test_get_voltage(self):
        """Test the get voltage function for different times t."""
        supply_parameter = {'frequency': 1, 'phase': 0, 'fixed_phase': True}
        supply = vs.AC1PhaseSupply(supply_parameter = supply_parameter)
        
        times = [0, 2*np.pi, 4*np.pi]
        for time in times:
            assert supply.get_voltage(time) == pytest.approx([0.0])
            
        times = [np.pi/2, 5*np.pi/2, 9*np.pi/2]
        for time in times:
            assert supply.get_voltage(time) == pytest.approx([230.0 * np.sqrt(2)])
            
        times = [3*np.pi/2, 7*np.pi/2, 11*np.pi/2]
        for time in times:
            assert supply.get_voltage(time) == pytest.approx([-230.0 * np.sqrt(2)])


    
    
