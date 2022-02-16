class OnePhaseACSupply(VoltageSupply):
    """AC one phase voltage supply"""

    def __init__(self, u_nominal=230, supply_parameter=None):
        """
        Args:
            u_nominal(float): Single phasic effective value of the voltage supply
            supply_parameter(dict): Consists of frequency f in Hz and phase phi in range of [0,2*pi) in case you wish for a fixed phase
        """

        super().__init__(u_nominal)
        
        self._fixed_phi = False
        if supply_parameter is not None:
            assert isinstance(supply_parameter, dict), "supply_parameter should be a dict"
            assert 'frequency' in supply_parameter.keys(), "Pass key 'frequency' for frequency f in Hz in your dict"
            if 'phase' in supply_parameter.keys():
                assert 0<= supply_parameter['phase'] < 2*np.pi, "The phase angle has to be given in rad in range [0,2*pi)"
                self._fixed_phi = True
                supply_parameter = supply_parameter
            else:
                supply_parameter['phase'] = np.random.rand()*2*np.pi
        else:
            supply_parameter = {'frequency': 50, 'phase': np.random.rand()*2*np.pi}

        self._f = supply_parameter['frequency']
        self._phi = supply_parameter['phase']
        self._max_amp = self._u_nominal*np.sqrt(2)
        self.supply_range = [-1*self._max_amp, self._max_amp]
        
    def reset(self):
        if not self._fixed_phi:
            self._phi = np.random.rand()*2*np.pi
        return self.get_voltage(0)
    
    def get_voltage(self, t, *_, **__):
        # Docstring of superclass
        self._u_sup = [self._max_amp*np.sin(2*np.pi*self._f*t + self._phi)]
        return self._u_sup