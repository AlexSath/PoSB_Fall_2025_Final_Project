from types import NoneType
from typing import Union
from collections.abc import Callable

from numpy.typing import NDArray

from .delay_model import Delay_Model
from ..tools import gaussian_pulse, double_sigmoid_pulse

class Transcriptional_Delay_Model(Delay_Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Names of the dZ/dt part outputs
        self.variable_dictionary = self._build_variable_dictionary()

        # validating system configuration variables
        self.sys_config = self._validate_sys_config(
            must_haves = ['PROM', 'K_D', 'K_DEG', 'N_COOP'],
            val_this = self.sys_config
        )

        # building the model
        self._model_builder = self._td_model_builder
        self.model = None

        # indicating number of ODEs expected in solution
        self.n_odes = self.n_delays + 1

    def _td_model_builder(self, sys_input: Union[NoneType, NDArray]=None) -> Callable:
        assert self.n_delays < 9, f"{self.__class__.__name__} only supports an even number of delays up to 8"
        assert self.n_delays % 2 == 0, f"{self.__class__.__name__} only supports an even number of delays up to 8"
        for attr in dir(self):
            if attr == f"_build_td_model_{self.n_delays}_layers":
                return getattr(self, attr)(sys_input=sys_input)
        else:
            raise ValueError(f"Model builder function not found for a model with {self.n_delays}")
        
    def _build_td_model_0_layers(self, sys_input: Union[NoneType, NDArray]=None) -> Callable:
        def model(t: float, y):
            TF = y

            if sys_input is None:
                if self.pulse == 'gaussian':
                    this_pulse = gaussian_pulse(t, **self.pulse_params)
                elif self.pulse == 'sigmoid':
                    this_pulse = double_sigmoid_pulse(t, **self.pulse_params)
                else:
                    this_pulse = self.pulse
            else: this_pulse = sys_input[int(t)]

            prom = self.sys_config['PROM']
            k_deg = self.sys_config['K_DEG']

            dydt = [
                this_pulse * prom - k_deg * TF
            ]

            return dydt

        return model
    
    def _build_td_model_2_layers(self, sys_input: Union[NoneType, NDArray]=None) -> Callable:
        def model(t: float, y):
            TD1, TD2, TF = y

            if sys_input is None:
                if self.pulse == 'gaussian':
                    this_pulse = gaussian_pulse(t, **self.pulse_params)
                elif self.pulse == 'sigmoid':
                    this_pulse = double_sigmoid_pulse(t, **self.pulse_params)
                else:
                    this_pulse = self.pulse
            else: this_pulse = sys_input[int(t)]

            prom = self.sys_config['PROM']
            k_deg = self.sys_config['K_DEG']
            k_d: NDArray = self._get_k_ds(n_k_ds=self.n_delays)
            n = self.sys_config['N_COOP']

            dydt = [
                this_pulse * prom - k_deg * TD1,
                prom * TD1**n / (k_d[0]**n + TD1**n) - k_deg * TD2,
                prom * TD2**n / (k_d[1]**n + TD2**n) - k_deg * TF
            ]

            return dydt
        
        return model
    
    def _build_td_model_4_layers(self, sys_input: Union[NoneType, NDArray]=None) -> Callable:
        def model(t: float, y):
            TD1, TD2, TD3, TD4, TF = y

            if sys_input is None:
                if self.pulse == 'gaussian':
                    this_pulse = gaussian_pulse(t, **self.pulse_params)
                elif self.pulse == 'sigmoid':
                    this_pulse = double_sigmoid_pulse(t, **self.pulse_params)
                else:
                    this_pulse = self.pulse
            else: this_pulse = sys_input[int(t)]

            prom = self.sys_config['PROM']
            k_deg = self.sys_config['K_DEG']
            k_d: NDArray = self._get_k_ds(n_k_ds=self.n_delays)
            n = self.sys_config['N_COOP']

            dydt = [
                this_pulse * prom - k_deg * TD1,
                prom * TD1**n / (k_d[0]**n + TD1**n) - k_deg * TD2,
                prom * TD2**n / (k_d[1]**n + TD2**n) - k_deg * TD3,
                prom * TD3**n / (k_d[2]**n + TD3**n) - k_deg * TD4,
                prom * TD4**n / (k_d[3]**n + TD4**n) - k_deg * TF
            ]

            return dydt
        
        return model
    
    def _build_td_model_6_layers(self, sys_input: Union[NoneType, NDArray]=None) -> Callable:
        def model(t: float, y):
            TD1, TD2, TD3, TD4, TD5, TD6, TF = y

            if sys_input is None:
                if self.pulse == 'gaussian':
                    this_pulse = gaussian_pulse(t, **self.pulse_params)
                elif self.pulse == 'sigmoid':
                    this_pulse = double_sigmoid_pulse(t, **self.pulse_params)
                else:
                    this_pulse = self.pulse
            else: this_pulse = sys_input[int(t)]

            prom = self.sys_config['PROM']
            k_deg = self.sys_config['K_DEG']
            k_d: NDArray = self._get_k_ds(n_k_ds=self.n_delays)
            n = self.sys_config['N_COOP']

            dydt = [
                this_pulse * prom - k_deg * TD1,
                prom * TD1**n / (k_d[0]**n + TD1**n) - k_deg * TD2,
                prom * TD2**n / (k_d[1]**n + TD2**n) - k_deg * TD3,
                prom * TD3**n / (k_d[2]**n + TD3**n) - k_deg * TD4,
                prom * TD4**n / (k_d[3]**n + TD4**n) - k_deg * TD5,
                prom * TD5**n / (k_d[4]**n + TD5**n) - k_deg * TD6,
                prom * TD6**n / (k_d[5]**n + TD6**n) - k_deg * TF
            ]

            return dydt
        
        return model
    
    def _build_td_model_8_layers(self, sys_input: Union[NoneType, NDArray]=None) -> Callable:
        def model(t: float, y):
            TD1, TD2, TD3, TD4, TD5, TD6, TD7, TD8, TF = y

            if sys_input is None:
                if self.pulse == 'gaussian':
                    this_pulse = gaussian_pulse(t, **self.pulse_params)
                elif self.pulse == 'sigmoid':
                    this_pulse = double_sigmoid_pulse(t, **self.pulse_params)
                else:
                    this_pulse = self.pulse
            else: this_pulse = sys_input[int(t)]

            prom = self.sys_config['PROM']
            k_deg = self.sys_config['K_DEG']
            k_d: NDArray = self._get_k_ds(n_k_ds=self.n_delays)
            n = self.sys_config['N_COOP']

            dydt = [
                this_pulse * prom - k_deg * TD1,
                prom * TD1**n / (k_d[0]**n + TD1**n) - k_deg * TD2,
                prom * TD2**n / (k_d[1]**n + TD2**n) - k_deg * TD3,
                prom * TD3**n / (k_d[2]**n + TD3**n) - k_deg * TD4,
                prom * TD4**n / (k_d[3]**n + TD4**n) - k_deg * TD5,
                prom * TD5**n / (k_d[4]**n + TD5**n) - k_deg * TD6,
                prom * TD6**n / (k_d[5]**n + TD6**n) - k_deg * TD7,
                prom * TD7**n / (k_d[6]**n + TD7**n) - k_deg * TD8,
                prom * TD8**n / (k_d[7]**n + TD8**n) - k_deg * TF
            ]

            return dydt
        
        return model
    
    def __str__(self):
        return f"{self.__class__.__name__} Object @0x{id(self)} \u007B\n\tn_layers: {self.n_delays}\n\tsolved: {self.solved}\n \u007D"
