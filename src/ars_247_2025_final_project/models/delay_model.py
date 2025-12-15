from typing import Union
from types import NoneType

from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
import numpy as np
from numpy.typing import NDArray

from ..tools import pulse_model

class Delay_Model():
    def __init__(
            self,
            n_delays,
            pulse: Union[float, str] = None, # possible values: 'gaussian' & 'sigmoid'
            pulse_params: Union[NoneType, dict] = None, # parameters for the promoter function (excluding t)
            sys_config: Union[NoneType, dict] = None,
            random_k_d: bool = False,
            k_d_params: Union[NoneType, dict] = None,
        ) -> NoneType:
        
        self.n_delays = n_delays
        self.sys_config = sys_config

        # Building transcriptional (input) pulse parameters
        if pulse == 'gaussian': 
            assert pulse_params is not None, f"If promoter is 'gaussian' pulse, must provide pulse parameters"
            assert set(list(pulse_params.keys()) + ['t']) == set(['t', 'peak_time', 'peak_value', 'width'])
        elif pulse == 'sigmoid':
            assert pulse_params is not None, f"If promoter is 'sigmoid' pulse, must provide pulse parameters"
            assert set(list(pulse_params.keys()) + ['t']) == set(['t', 't_on', 't_off', 'peak_value', 'comp'])
        assert pulse is not None, f"If pulse is not gaussian or sigmoid, a constant value (e.g. 0.01) must be provided."

        self.random_k_d = random_k_d
        if self.random_k_d:
            assert k_d_params is not None, f"If random_k_d is True, then k_d_params must be provided"
            assert set(list(k_d_params.keys())) == set(['sigma', 'mu']), f"k_d_params must be 'sigma' and 'nu'"
        self.k_d_params = k_d_params

        self.pulse = pulse
        self.pulse_params = pulse_params
        self._model_builder = None
        self.solved = False
        self.soln = None
        self.n_odes = None

    def solve_model_ivp(
            self, t_start, t_end, 
            sys_input: Union[NoneType, NDArray]=None,
            y0: Union[NoneType, NDArray]=None
        ) -> OdeResult:

        # Build the model with correct system input
        self.model = self._model_builder(sys_input=sys_input)

        # Solving the ivp
        soln = solve_ivp(
            fun = self.model,
            t_span = [t_start, t_end - 1],
            y0 = np.zeros(shape=(self.n_odes)) if y0 is None else y0,
            t_eval = np.arange(t_start, t_end)
        )
        
        # Careful! these paremeters are dangerous since soln doesn't know what sys_input is
        self.solved = True
        self.soln = soln

        return self.soln
    
    def _validate_sys_config(self, must_haves: list[str], val_this: dict) -> dict:
        assert set(must_haves) == set(val_this.keys()), f"System config missing {set(must_haves) - set(val_this.keys())}"
        return val_this
    
    def _build_variable_dictionary(self):
        pass

    def _get_k_ds(self, n_k_ds: int) -> NDArray:
        if self.random_k_d:
            return np.random.normal(loc=self.k_d_params['mu'], scale=self.k_d_params['sigma'], size=(n_k_ds,))
        else:
            # If self.random_k_d is false, then fill with the identical provided value
            return np.full(shape=(n_k_ds,), fill_value=self.sys_config['K_D'])