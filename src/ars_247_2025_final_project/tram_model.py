from types import NoneType
from typing import Union
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from .tools import gaussian_pulse, double_sigmoid_pulse, pulse_model

# Thoughts so far: when all parameters are physiological (0.47, 2.5, 0.15, 0.15, 0.00001, 0.01), amplification and delay is seen, but signal is very broad.
# IF: radius is significantly increased (2.5 --> 20), output signal seems to sharpen somewhat, and further time delays are seen.
# IF: 

class TRAM_Model():
    def __init__(
            self,
            n_domains: int,
            pulse: Union[float, str] = None, # possible values: 'gaussian' & 'sigmoid'
            pulse_params: Union[NoneType, dict] = None, # parameters for the promoter function (excluding t)
            sys_config: Union[NoneType, dict] = None
        ) -> NoneType:
        self.n_domains = n_domains

        # Names of the dZ/dt part outputs
        self.variable_dictionary = self._build_variable_dictionary()
        
        # Building transcriptional (input) pulse parameters
        if pulse == 'gaussian': 
            assert pulse_params is not None, f"If promoter is 'gaussian' pulse, must provide pulse parameters"
            assert set(list(pulse_params.keys()) + ['t']) == set(['t', 'peak_time', 'peak_value', 'width'])
        elif pulse == 'sigmoid':
            assert pulse_params is not None, f"If promoter is 'sigmoid' pulse, must provide pulse parameters"
            assert set(list(pulse_params.keys()) + ['t']) == set(['t', 't_on', 't_off', 'peak_value', 'comp'])
        assert pulse is not None, f"If pulse is not gaussian or sigmoid, a constant value (e.g. 0.01) must be provided."
        self.pulse = pulse
        self.pulse_params = pulse_params

        # validating system configuration variables
        self.sys_config = self._validate_sys_config(sys_config)

        # building the model
        self.model = self._tram_model_builder()

    def solve_pulse(self) -> OdeResult:
        model = pulse_model(self.pulse, self.pulse_params)
        soln = solve_ivp(
            fun = model,
            t_span = [0, 4 * 3600],
            y0 = [0],
            # t_eval = np.linspace(0, 4 * 3600, 100000)
            max_step = 1
        )
        return soln

    def solve_tram_ivp(self, t_start, t_end) -> OdeResult:
        soln = solve_ivp(
            fun = self.model,
            t_span = [t_start, t_end],
            # y0 = np.zeros(shape=(self.n_domains * 4 + 1))
            y0 = np.zeros(shape=(self.n_domains * 4 + 1)),
            max_step = 1
        )
        return soln

    def _build_variable_dictionary(self) -> dict:
        variables = {}
        for i in range(0, self.n_domains * 4, 4):
            tn = i // 4 + 1 # nth tram counter
            variables[i] = f"TF{i // 4 + 1}n"
            variables[i + 1] = f"TF{i // 4 + 1}m"
            variables[i + 2] = f"TF{i // 4 + 1}Cm"
            variables[i + 3] = f"TF{i // 4 + 1}Cn"
        variables[len(variables)] = "TF"
        return variables
    
    def _validate_sys_config(self, val_this: dict) -> dict:
        must_haves = ['PROM', 'V_A', 'R_M', 'K_TEV', 'K_TVMV', 'K_TRAM_DEG', 'K_TF_DEG']
        assert set(must_haves) == set(val_this.keys()), f"System config missing {set(must_haves) - set(val_this.keys())}"
        return val_this


    def _tram_model_builder(self) -> NoneType:
        assert self.n_domains < 9, f"Number of tram domains must be less than 4, not {self.n_domains}"
        for attr in dir(self):
            if attr == f"_build_tram_model_{self.n_domains}_domain":
                return getattr(self, attr)()
        else:
            raise ValueError(f"Model builder function not found for a model with {self.n_domains}")
        
    def _build_tram_model_0_domain(self) -> Callable:
        def model(t: float, y):
            TF = y

            if self.pulse == 'gaussian':
                this_pulse = gaussian_pulse(t, **self.pulse_params)
            elif self.pulse == 'sigmoid':
                this_pulse = double_sigmoid_pulse(t, **self.pulse_params)
            else:
                this_pulse = self.pulse

            dydt = [
                this_pulse * self.sys_config['PROM'] - self.sys_config['K_TF_DEG'] * TF
            ]

            return dydt
        
        return model

        
    def _build_tram_model_1_domain(self) -> Callable:
        def model(t: float, y):
            TF1n, TF1m, TF1Cm, TF1Cn, \
            TF = y

            if self.pulse == 'gaussian':
                this_pulse = gaussian_pulse(t, **self.pulse_params)
            elif self.pulse == 'sigmoid':
                this_pulse = double_sigmoid_pulse(t, **self.pulse_params)
            else:
                this_pulse = self.pulse
            
            dydt = [
                # Production + initiate first round
                this_pulse * self.sys_config['PROM'] - self.sys_config['V_A'] / self.sys_config['R_M'] * TF1n - self.sys_config['K_TRAM_DEG'] * TF1n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF1n - self.sys_config['K_TEV'] * TF1m - self.sys_config['K_TRAM_DEG'] * TF1m,
                self.sys_config['K_TEV'] * TF1m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF1Cm - self.sys_config['K_TRAM_DEG'] * TF1Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF1Cm - self.sys_config['K_TVMV'] * TF1Cn - self.sys_config['K_TRAM_DEG'] * TF1Cn,
                
                # Final TRAM domain cleaved, regular TF degradation (in nucleus)
                self.sys_config['K_TVMV'] * TF1Cn - self.sys_config['K_TF_DEG'] * TF
            ]
            return dydt
        
        return model
        
    def _build_tram_model_2_domain(self) -> Callable:
        def model(t: float, y):
            TF1n, TF1m, TF1Cm, TF1Cn, \
            TF2n, TF2m, TF2Cm, TF2Cn, \
            TF = y

            if self.pulse == 'gaussian':
                this_pulse = gaussian_pulse(t, **self.pulse_params)
            elif self.pulse == 'sigmoid':
                this_pulse = double_sigmoid_pulse(t, **self.pulse_params)
            else:
                this_pulse = self.pulse

            dydt = [
                # Production + initiate first round
                this_pulse * self.sys_config['PROM'] - self.sys_config['V_A'] / self.sys_config['R_M'] * TF1n - self.sys_config['K_TRAM_DEG'] * TF1n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF1n - self.sys_config['K_TEV'] * TF1m - self.sys_config['K_TRAM_DEG'] * TF1m,
                self.sys_config['K_TEV'] * TF1m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF1Cm - self.sys_config['K_TRAM_DEG'] * TF1Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF1Cm - self.sys_config['K_TVMV'] * TF1Cn - self.sys_config['K_TRAM_DEG'] * TF1Cn,

                # Second round
                self.sys_config['K_TVMV'] * TF1Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF2n - self.sys_config['K_TRAM_DEG'] * TF2n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF2n - self.sys_config['K_TEV'] * TF2m - self.sys_config['K_TRAM_DEG'] * TF2m,
                self.sys_config['K_TEV'] * TF2m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF2Cm - self.sys_config['K_TRAM_DEG'] * TF2Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF2Cm - self.sys_config['K_TVMV'] * TF2Cn - self.sys_config['K_TRAM_DEG'] * TF2Cn,
                
                # Final TRAM domain cleaved, regular TF degradation (in nucleus)
                self.sys_config['K_TVMV'] * TF2Cn - self.sys_config['K_TF_DEG'] * TF
            ]
            return dydt
        
        return model
    
    def _build_tram_model_3_domain(self) -> Callable:
        def model(t: float, y):
            TF1n, TF1m, TF1Cm, TF1Cn, \
            TF2n, TF2m, TF2Cm, TF2Cn, \
            TF3n, TF3m, TF3Cm, TF3Cn, TF = y

            if self.pulse == 'gaussian':
                this_pulse = gaussian_pulse(t, **self.pulse_params)
            elif self.pulse == 'sigmoid':
                this_pulse = double_sigmoid_pulse(t, **self.pulse_params)
            else:
                this_pulse = self.pulse

            dydt = [
                # Production + initiate first round
                this_pulse * self.sys_config['PROM'] - self.sys_config['V_A'] / self.sys_config['R_M'] * TF1n - self.sys_config['K_TRAM_DEG'] * TF1n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF1n - self.sys_config['K_TEV'] * TF1m - self.sys_config['K_TRAM_DEG'] * TF1m,
                self.sys_config['K_TEV'] * TF1m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF1Cm - self.sys_config['K_TRAM_DEG'] * TF1Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF1Cm - self.sys_config['K_TVMV'] * TF1Cn - self.sys_config['K_TRAM_DEG'] * TF1Cn,

                # Second round
                self.sys_config['K_TVMV'] * TF1Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF2n - self.sys_config['K_TRAM_DEG'] * TF2n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF2n - self.sys_config['K_TEV'] * TF2m - self.sys_config['K_TRAM_DEG'] * TF2m,
                self.sys_config['K_TEV'] * TF2m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF2Cm - self.sys_config['K_TRAM_DEG'] * TF2Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF2Cm - self.sys_config['K_TVMV'] * TF2Cn - self.sys_config['K_TRAM_DEG'] * TF2Cn,

                # Third round
                self.sys_config['K_TVMV'] * TF2Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF3n - self.sys_config['K_TRAM_DEG'] * TF3n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF3n - self.sys_config['K_TEV'] * TF3m - self.sys_config['K_TRAM_DEG'] * TF3m,
                self.sys_config['K_TEV'] * TF3m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF3Cm - self.sys_config['K_TRAM_DEG'] * TF3Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF3Cm - self.sys_config['K_TVMV'] * TF3Cn - self.sys_config['K_TRAM_DEG'] * TF3Cn,
                
                # Final TRAM domain cleaved, regular TF degradation (in nucleus)
                self.sys_config['K_TVMV'] * TF3Cn - self.sys_config['K_TF_DEG'] * TF
            ]
            return dydt
        
        return model
        
    
    def _build_tram_model_4_domain(self) -> Callable:
        def model(t: float, y):
            TF1n, TF1m, TF1Cm, TF1Cn, \
            TF2n, TF2m, TF2Cm, TF2Cn, \
            TF3n, TF3m, TF3Cm, TF3Cn, \
            TF4n, TF4m, TF4Cm, TF4Cn, \
            TF = y

            if self.pulse == 'gaussian':
                this_pulse = gaussian_pulse(t, **self.pulse_params)
            elif self.pulse == 'sigmoid':
                this_pulse = double_sigmoid_pulse(t, **self.pulse_params)
            else:
                this_pulse = self.pulse
            
            dydt = [
                # Production + initiate first round
                this_pulse * self.sys_config['PROM'] - self.sys_config['V_A'] / self.sys_config['R_M'] * TF1n - self.sys_config['K_TRAM_DEG'] * TF1n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF1n - self.sys_config['K_TEV'] * TF1m - self.sys_config['K_TRAM_DEG'] * TF1m,
                self.sys_config['K_TEV'] * TF1m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF1Cm - self.sys_config['K_TRAM_DEG'] * TF1Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF1Cm - self.sys_config['K_TVMV'] * TF1Cn - self.sys_config['K_TRAM_DEG'] * TF1Cn,

                # Second round
                self.sys_config['K_TVMV'] * TF1Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF2n - self.sys_config['K_TRAM_DEG'] * TF2n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF2n - self.sys_config['K_TEV'] * TF2m - self.sys_config['K_TRAM_DEG'] * TF2m,
                self.sys_config['K_TEV'] * TF2m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF2Cm - self.sys_config['K_TRAM_DEG'] * TF2Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF2Cm - self.sys_config['K_TVMV'] * TF2Cn - self.sys_config['K_TRAM_DEG'] * TF2Cn,

                # Third round
                self.sys_config['K_TVMV'] * TF2Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF3n - self.sys_config['K_TRAM_DEG'] * TF3n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF3n - self.sys_config['K_TEV'] * TF3m - self.sys_config['K_TRAM_DEG'] * TF3m,
                self.sys_config['K_TEV'] * TF3m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF3Cm - self.sys_config['K_TRAM_DEG'] * TF3Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF3Cm - self.sys_config['K_TVMV'] * TF3Cn - self.sys_config['K_TRAM_DEG'] * TF3Cn,

                # Fourth round
                self.sys_config['K_TVMV'] * TF3Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF4n - self.sys_config['K_TRAM_DEG'] * TF4n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF4n - self.sys_config['K_TEV'] * TF4m - self.sys_config['K_TRAM_DEG'] * TF4m,
                self.sys_config['K_TEV'] * TF4m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF4Cm - self.sys_config['K_TRAM_DEG'] * TF4Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF4Cm - self.sys_config['K_TVMV'] * TF4Cn - self.sys_config['K_TRAM_DEG'] * TF4Cn,
                
                # Final TRAM domain cleaved, regular TF degradation (in nucleus)
                self.sys_config['K_TVMV'] * TF4Cn - self.sys_config['K_TF_DEG'] * TF
            ]
            return dydt
        
        return model
    
    def _build_tram_model_5_domain(self) -> Callable:
        def model(t: float, y):
            TF1n, TF1m, TF1Cm, TF1Cn, \
            TF2n, TF2m, TF2Cm, TF2Cn, \
            TF3n, TF3m, TF3Cm, TF3Cn, \
            TF4n, TF4m, TF4Cm, TF4Cn, \
            TF5n, TF5m, TF5Cm, TF5Cn, \
            TF = y

            if self.pulse == 'gaussian':
                this_pulse = gaussian_pulse(t, **self.pulse_params)
            elif self.pulse == 'sigmoid':
                this_pulse = double_sigmoid_pulse(t, **self.pulse_params)
            else:
                this_pulse = self.pulse
            
            dydt = [
                # Production + initiate first round
                this_pulse * self.sys_config['PROM'] - self.sys_config['V_A'] / self.sys_config['R_M'] * TF1n - self.sys_config['K_TRAM_DEG'] * TF1n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF1n - self.sys_config['K_TEV'] * TF1m - self.sys_config['K_TRAM_DEG'] * TF1m,
                self.sys_config['K_TEV'] * TF1m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF1Cm - self.sys_config['K_TRAM_DEG'] * TF1Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF1Cm - self.sys_config['K_TVMV'] * TF1Cn - self.sys_config['K_TRAM_DEG'] * TF1Cn,

                # Second round
                self.sys_config['K_TVMV'] * TF1Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF2n - self.sys_config['K_TRAM_DEG'] * TF2n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF2n - self.sys_config['K_TEV'] * TF2m - self.sys_config['K_TRAM_DEG'] * TF2m,
                self.sys_config['K_TEV'] * TF2m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF2Cm - self.sys_config['K_TRAM_DEG'] * TF2Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF2Cm - self.sys_config['K_TVMV'] * TF2Cn - self.sys_config['K_TRAM_DEG'] * TF2Cn,

                # Third round
                self.sys_config['K_TVMV'] * TF2Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF3n - self.sys_config['K_TRAM_DEG'] * TF3n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF3n - self.sys_config['K_TEV'] * TF3m - self.sys_config['K_TRAM_DEG'] * TF3m,
                self.sys_config['K_TEV'] * TF3m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF3Cm - self.sys_config['K_TRAM_DEG'] * TF3Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF3Cm - self.sys_config['K_TVMV'] * TF3Cn - self.sys_config['K_TRAM_DEG'] * TF3Cn,

                # Fourth round
                self.sys_config['K_TVMV'] * TF3Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF4n - self.sys_config['K_TRAM_DEG'] * TF4n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF4n - self.sys_config['K_TEV'] * TF4m - self.sys_config['K_TRAM_DEG'] * TF4m,
                self.sys_config['K_TEV'] * TF4m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF4Cm - self.sys_config['K_TRAM_DEG'] * TF4Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF4Cm - self.sys_config['K_TVMV'] * TF4Cn - self.sys_config['K_TRAM_DEG'] * TF4Cn,

                # Fifth round
                self.sys_config['K_TVMV'] * TF4Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF5n - self.sys_config['K_TRAM_DEG'] * TF5n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF5n - self.sys_config['K_TEV'] * TF5m - self.sys_config['K_TRAM_DEG'] * TF5m,
                self.sys_config['K_TEV'] * TF5m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF5Cm - self.sys_config['K_TRAM_DEG'] * TF5Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF5Cm - self.sys_config['K_TVMV'] * TF5Cn - self.sys_config['K_TRAM_DEG'] * TF5Cn,
                
                # Final TRAM domain cleaved, regular TF degradation (in nucleus)
                self.sys_config['K_TVMV'] * TF5Cn - self.sys_config['K_TF_DEG'] * TF
            ]
            return dydt
        
        return model
    
    def _build_tram_model_6_domain(self) -> Callable:
        def model(t: float, y):
            TF1n, TF1m, TF1Cm, TF1Cn, \
            TF2n, TF2m, TF2Cm, TF2Cn, \
            TF3n, TF3m, TF3Cm, TF3Cn, \
            TF4n, TF4m, TF4Cm, TF4Cn, \
            TF5n, TF5m, TF5Cm, TF5Cn, \
            TF6n, TF6m, TF6Cm, TF6Cn, \
            TF = y

            if self.pulse == 'gaussian':
                this_pulse = gaussian_pulse(t, **self.pulse_params)
            elif self.pulse == 'sigmoid':
                this_pulse = double_sigmoid_pulse(t, **self.pulse_params)
            else:
                this_pulse = self.pulse
            
            dydt = [
                # Production + initiate first round
                this_pulse * self.sys_config['PROM'] - self.sys_config['V_A'] / self.sys_config['R_M'] * TF1n - self.sys_config['K_TRAM_DEG'] * TF1n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF1n - self.sys_config['K_TEV'] * TF1m - self.sys_config['K_TRAM_DEG'] * TF1m,
                self.sys_config['K_TEV'] * TF1m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF1Cm - self.sys_config['K_TRAM_DEG'] * TF1Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF1Cm - self.sys_config['K_TVMV'] * TF1Cn - self.sys_config['K_TRAM_DEG'] * TF1Cn,

                # Second round
                self.sys_config['K_TVMV'] * TF1Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF2n - self.sys_config['K_TRAM_DEG'] * TF2n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF2n - self.sys_config['K_TEV'] * TF2m - self.sys_config['K_TRAM_DEG'] * TF2m,
                self.sys_config['K_TEV'] * TF2m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF2Cm - self.sys_config['K_TRAM_DEG'] * TF2Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF2Cm - self.sys_config['K_TVMV'] * TF2Cn - self.sys_config['K_TRAM_DEG'] * TF2Cn,

                # Third round
                self.sys_config['K_TVMV'] * TF2Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF3n - self.sys_config['K_TRAM_DEG'] * TF3n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF3n - self.sys_config['K_TEV'] * TF3m - self.sys_config['K_TRAM_DEG'] * TF3m,
                self.sys_config['K_TEV'] * TF3m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF3Cm - self.sys_config['K_TRAM_DEG'] * TF3Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF3Cm - self.sys_config['K_TVMV'] * TF3Cn - self.sys_config['K_TRAM_DEG'] * TF3Cn,

                # Fourth round
                self.sys_config['K_TVMV'] * TF3Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF4n - self.sys_config['K_TRAM_DEG'] * TF4n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF4n - self.sys_config['K_TEV'] * TF4m - self.sys_config['K_TRAM_DEG'] * TF4m,
                self.sys_config['K_TEV'] * TF4m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF4Cm - self.sys_config['K_TRAM_DEG'] * TF4Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF4Cm - self.sys_config['K_TVMV'] * TF4Cn - self.sys_config['K_TRAM_DEG'] * TF4Cn,

                # Fifth round
                self.sys_config['K_TVMV'] * TF4Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF5n - self.sys_config['K_TRAM_DEG'] * TF5n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF5n - self.sys_config['K_TEV'] * TF5m - self.sys_config['K_TRAM_DEG'] * TF5m,
                self.sys_config['K_TEV'] * TF5m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF5Cm - self.sys_config['K_TRAM_DEG'] * TF5Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF5Cm - self.sys_config['K_TVMV'] * TF5Cn - self.sys_config['K_TRAM_DEG'] * TF5Cn,

                # Sixth round
                self.sys_config['K_TVMV'] * TF5Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF6n - self.sys_config['K_TRAM_DEG'] * TF6n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF6n - self.sys_config['K_TEV'] * TF6m - self.sys_config['K_TRAM_DEG'] * TF6m,
                self.sys_config['K_TEV'] * TF6m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF6Cm - self.sys_config['K_TRAM_DEG'] * TF6Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF6Cm - self.sys_config['K_TVMV'] * TF6Cn - self.sys_config['K_TRAM_DEG'] * TF6Cn,
                
                # Final TRAM domain cleaved, regular TF degradation (in nucleus)
                self.sys_config['K_TVMV'] * TF6Cn - self.sys_config['K_TF_DEG'] * TF
            ]
            return dydt
        
        return model
    
    def _build_tram_model_7_domain(self) -> Callable:
        def model(t: float, y):
            TF1n, TF1m, TF1Cm, TF1Cn, \
            TF2n, TF2m, TF2Cm, TF2Cn, \
            TF3n, TF3m, TF3Cm, TF3Cn, \
            TF4n, TF4m, TF4Cm, TF4Cn, \
            TF5n, TF5m, TF5Cm, TF5Cn, \
            TF6n, TF6m, TF6Cm, TF6Cn, \
            TF7n, TF7m, TF7Cm, TF7Cn, \
            TF = y

            if self.pulse == 'gaussian':
                this_pulse = gaussian_pulse(t, **self.pulse_params)
            elif self.pulse == 'sigmoid':
                this_pulse = double_sigmoid_pulse(t, **self.pulse_params)
            else:
                this_pulse = self.pulse
            
            dydt = [
                # Production + initiate first round
                this_pulse * self.sys_config['PROM'] - self.sys_config['V_A'] / self.sys_config['R_M'] * TF1n - self.sys_config['K_TRAM_DEG'] * TF1n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF1n - self.sys_config['K_TEV'] * TF1m - self.sys_config['K_TRAM_DEG'] * TF1m,
                self.sys_config['K_TEV'] * TF1m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF1Cm - self.sys_config['K_TRAM_DEG'] * TF1Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF1Cm - self.sys_config['K_TVMV'] * TF1Cn - self.sys_config['K_TRAM_DEG'] * TF1Cn,

                # Second round
                self.sys_config['K_TVMV'] * TF1Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF2n - self.sys_config['K_TRAM_DEG'] * TF2n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF2n - self.sys_config['K_TEV'] * TF2m - self.sys_config['K_TRAM_DEG'] * TF2m,
                self.sys_config['K_TEV'] * TF2m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF2Cm - self.sys_config['K_TRAM_DEG'] * TF2Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF2Cm - self.sys_config['K_TVMV'] * TF2Cn - self.sys_config['K_TRAM_DEG'] * TF2Cn,

                # Third round
                self.sys_config['K_TVMV'] * TF2Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF3n - self.sys_config['K_TRAM_DEG'] * TF3n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF3n - self.sys_config['K_TEV'] * TF3m - self.sys_config['K_TRAM_DEG'] * TF3m,
                self.sys_config['K_TEV'] * TF3m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF3Cm - self.sys_config['K_TRAM_DEG'] * TF3Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF3Cm - self.sys_config['K_TVMV'] * TF3Cn - self.sys_config['K_TRAM_DEG'] * TF3Cn,

                # Fourth round
                self.sys_config['K_TVMV'] * TF3Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF4n - self.sys_config['K_TRAM_DEG'] * TF4n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF4n - self.sys_config['K_TEV'] * TF4m - self.sys_config['K_TRAM_DEG'] * TF4m,
                self.sys_config['K_TEV'] * TF4m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF4Cm - self.sys_config['K_TRAM_DEG'] * TF4Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF4Cm - self.sys_config['K_TVMV'] * TF4Cn - self.sys_config['K_TRAM_DEG'] * TF4Cn,

                # Fifth round
                self.sys_config['K_TVMV'] * TF4Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF5n - self.sys_config['K_TRAM_DEG'] * TF5n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF5n - self.sys_config['K_TEV'] * TF5m - self.sys_config['K_TRAM_DEG'] * TF5m,
                self.sys_config['K_TEV'] * TF5m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF5Cm - self.sys_config['K_TRAM_DEG'] * TF5Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF5Cm - self.sys_config['K_TVMV'] * TF5Cn - self.sys_config['K_TRAM_DEG'] * TF5Cn,

                # Sixth round
                self.sys_config['K_TVMV'] * TF5Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF6n - self.sys_config['K_TRAM_DEG'] * TF6n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF6n - self.sys_config['K_TEV'] * TF6m - self.sys_config['K_TRAM_DEG'] * TF6m,
                self.sys_config['K_TEV'] * TF6m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF6Cm - self.sys_config['K_TRAM_DEG'] * TF6Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF6Cm - self.sys_config['K_TVMV'] * TF6Cn - self.sys_config['K_TRAM_DEG'] * TF6Cn,

                # Seventh round
                self.sys_config['K_TVMV'] * TF6Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF7n - self.sys_config['K_TRAM_DEG'] * TF7n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF7n - self.sys_config['K_TEV'] * TF7m - self.sys_config['K_TRAM_DEG'] * TF7m,
                self.sys_config['K_TEV'] * TF7m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF7Cm - self.sys_config['K_TRAM_DEG'] * TF7Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF7Cm - self.sys_config['K_TVMV'] * TF7Cn - self.sys_config['K_TRAM_DEG'] * TF7Cn,
                
                # Final TRAM domain cleaved, regular TF degradation (in nucleus)
                self.sys_config['K_TVMV'] * TF7Cn - self.sys_config['K_TF_DEG'] * TF
            ]
            return dydt
        
        return model
    
    def _build_tram_model_8_domain(self) -> Callable:
        def model(t: float, y):
            TF1n, TF1m, TF1Cm, TF1Cn, \
            TF2n, TF2m, TF2Cm, TF2Cn, \
            TF3n, TF3m, TF3Cm, TF3Cn, \
            TF4n, TF4m, TF4Cm, TF4Cn, \
            TF5n, TF5m, TF5Cm, TF5Cn, \
            TF6n, TF6m, TF6Cm, TF6Cn, \
            TF7n, TF7m, TF7Cm, TF7Cn, \
            TF8n, TF8m, TF8Cm, TF8Cn, \
            TF = y

            if self.pulse == 'gaussian':
                this_pulse = gaussian_pulse(t, **self.pulse_params)
            elif self.pulse == 'sigmoid':
                this_pulse = double_sigmoid_pulse(t, **self.pulse_params)
            else:
                this_pulse = self.pulse
            
            dydt = [
                # Production + initiate first round
                this_pulse * self.sys_config['PROM'] - self.sys_config['V_A'] / self.sys_config['R_M'] * TF1n - self.sys_config['K_TRAM_DEG'] * TF1n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF1n - self.sys_config['K_TEV'] * TF1m - self.sys_config['K_TRAM_DEG'] * TF1m,
                self.sys_config['K_TEV'] * TF1m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF1Cm - self.sys_config['K_TRAM_DEG'] * TF1Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF1Cm - self.sys_config['K_TVMV'] * TF1Cn - self.sys_config['K_TRAM_DEG'] * TF1Cn,

                # Second round
                self.sys_config['K_TVMV'] * TF1Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF2n - self.sys_config['K_TRAM_DEG'] * TF2n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF2n - self.sys_config['K_TEV'] * TF2m - self.sys_config['K_TRAM_DEG'] * TF2m,
                self.sys_config['K_TEV'] * TF2m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF2Cm - self.sys_config['K_TRAM_DEG'] * TF2Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF2Cm - self.sys_config['K_TVMV'] * TF2Cn - self.sys_config['K_TRAM_DEG'] * TF2Cn,

                # Third round
                self.sys_config['K_TVMV'] * TF2Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF3n - self.sys_config['K_TRAM_DEG'] * TF3n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF3n - self.sys_config['K_TEV'] * TF3m - self.sys_config['K_TRAM_DEG'] * TF3m,
                self.sys_config['K_TEV'] * TF3m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF3Cm - self.sys_config['K_TRAM_DEG'] * TF3Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF3Cm - self.sys_config['K_TVMV'] * TF3Cn - self.sys_config['K_TRAM_DEG'] * TF3Cn,

                # Fourth round
                self.sys_config['K_TVMV'] * TF3Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF4n - self.sys_config['K_TRAM_DEG'] * TF4n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF4n - self.sys_config['K_TEV'] * TF4m - self.sys_config['K_TRAM_DEG'] * TF4m,
                self.sys_config['K_TEV'] * TF4m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF4Cm - self.sys_config['K_TRAM_DEG'] * TF4Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF4Cm - self.sys_config['K_TVMV'] * TF4Cn - self.sys_config['K_TRAM_DEG'] * TF4Cn,

                # Fifth round
                self.sys_config['K_TVMV'] * TF4Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF5n - self.sys_config['K_TRAM_DEG'] * TF5n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF5n - self.sys_config['K_TEV'] * TF5m - self.sys_config['K_TRAM_DEG'] * TF5m,
                self.sys_config['K_TEV'] * TF5m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF5Cm - self.sys_config['K_TRAM_DEG'] * TF5Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF5Cm - self.sys_config['K_TVMV'] * TF5Cn - self.sys_config['K_TRAM_DEG'] * TF5Cn,

                # Sixth round
                self.sys_config['K_TVMV'] * TF5Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF6n - self.sys_config['K_TRAM_DEG'] * TF6n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF6n - self.sys_config['K_TEV'] * TF6m - self.sys_config['K_TRAM_DEG'] * TF6m,
                self.sys_config['K_TEV'] * TF6m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF6Cm - self.sys_config['K_TRAM_DEG'] * TF6Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF6Cm - self.sys_config['K_TVMV'] * TF6Cn - self.sys_config['K_TRAM_DEG'] * TF6Cn,

                # Seventh round
                self.sys_config['K_TVMV'] * TF6Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF7n - self.sys_config['K_TRAM_DEG'] * TF7n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF7n - self.sys_config['K_TEV'] * TF7m - self.sys_config['K_TRAM_DEG'] * TF7m,
                self.sys_config['K_TEV'] * TF7m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF7Cm - self.sys_config['K_TRAM_DEG'] * TF7Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF7Cm - self.sys_config['K_TVMV'] * TF7Cn - self.sys_config['K_TRAM_DEG'] * TF7Cn,

                # Eighth round
                self.sys_config['K_TVMV'] * TF7Cn - self.sys_config['V_A'] / self.sys_config['R_M'] * TF8n - self.sys_config['K_TRAM_DEG'] * TF8n,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF8n - self.sys_config['K_TEV'] * TF8m - self.sys_config['K_TRAM_DEG'] * TF8m,
                self.sys_config['K_TEV'] * TF8m - self.sys_config['V_A'] / self.sys_config['R_M'] * TF8Cm - self.sys_config['K_TRAM_DEG'] * TF8Cm,
                self.sys_config['V_A'] / self.sys_config['R_M'] * TF8Cm - self.sys_config['K_TVMV'] * TF8Cn - self.sys_config['K_TRAM_DEG'] * TF8Cn,
                
                # Final TRAM domain cleaved, regular TF degradation (in nucleus)
                self.sys_config['K_TVMV'] * TF8Cn - self.sys_config['K_TF_DEG'] * TF
            ]
            return dydt
        
        return model
    
    def __str__(self):
        return f"{self.__class__.__name__} Object @0x{id(self)} \u007B\n\tn_domains: {self.n_domains}\n\tsolved: {self.solved}\n \u007D"