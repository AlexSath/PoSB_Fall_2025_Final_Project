from typing import Union
from types import NoneType
from collections.abc import Callable

from numpy.typing import NDArray
import numpy as np

from .delay_model import Delay_Model
from ..tools import hill

class Stress_Model(Delay_Model):
    def __init__(
            self, 
            input1: NDArray, input2: NDArray, 
            input_names: list[str],
            model_type: str = 'are_hse',
            **kwargs,
        ):
        super().__init__(**kwargs)

        # Names of the dZ/dt part outputs
        self.variable_dictionary = self._build_variable_dictionary()

        # getting inputs
        self.input1: NDArray = input1
        self.input2: NDArray = input2
        assert self.input1.shape == self.input2.shape, \
            f"Input shapes must be the same. Received {self.input1.shape} and {self.input2.shape}"

        # getting input_names
        self.input_names = input_names
        if model_type == 'are_hse': 
            assert set(self.input_names) == set(['hse', 'are']), \
                f"For model type 'hse_are', inputs must be 'hse' and 'are' signal"
        self.model_type = model_type

        # validating system configuration variables
        self.sys_config = self._validate_sys_config(
            must_haves = ['K_TX', 'K_DEG', 'N_COOP'] if self.random_k_d \
                else ['K_TX', 'K_DEG', 'N_COOP', 'K_D', 'K_D_DOWN'],
            val_this = self.sys_config
        ) 

        # building the model
        self._model_builder = self._stress_model_builder
        self.model = None

        # indicating number of ODEs expected in solution
        self.n_odes = None

    def _stress_model_builder(self, sys_input: Union[NoneType, NDArray]=None):
        try:
            return getattr(self, f"_build_{self.model_type}_model")()
        except Exception as e:
            raise ValueError(f"Could not find model of name {self.model_type}\n.Error: {e}")
        
    def _build_are_hse_model(self) -> Callable:
        # by convention, input 1 will be are
        # by convention, input 2 will be hse
        self.n_odes=12

        def model(t: float, y):
            VanA1, TVP64, TetR, CymR, \
            TgtA1, Gal4, PhlF, VanR, \
            ScbR, QF2, GFP, mCh = y

            k_d: NDArray = self._get_k_ds(n_k_ds=16)
            k_tx = self.sys_config['K_TX']
            k_deg = self.sys_config['K_DEG']
            n = self.sys_config['N_COOP']
            k_down = 0.2

            i1 = self.input1
            i2 = self.input2
            t=int(t)

            dydt = [
                # Input layer
                # k_tx*hill(True, i1[t], k_d[0], n) - k_deg*VanA1, #VanA1
                # k_tx*hill(True, i2[t], k_d[1], n) - k_deg*TVP64, #TVP64
                # k_tx*hill(True, i1[t], k_d[2], n) - k_deg*TetR, #TetR
                # k_tx*hill(True, i2[t], k_d[3], n) - k_deg*CymR, #CymR

                k_tx*hill(True, i1[t], k_d[0], n) - k_deg*VanA1, #VanA1
                k_tx*hill(True, i2[t], k_d[1], n) - k_deg*TVP64, #TVP64
                k_tx*hill(True, i1[t], k_d[2], n) - k_deg*TetR, #TetR
                k_tx*hill(True, i2[t], k_d[3], n) - k_deg*CymR, #CymR
                
                # Not layer
                k_tx*hill(False, TetR, k_down, n) - k_deg*TgtA1, #TgtA1
                k_tx*hill(False, CymR, k_down, n) - k_deg*Gal4, #Gal4

                # AND / XOR layer
                k_tx*(hill(True, TVP64, k_down, n)+hill(True, TgtA1, k_down, n)) \
                    - k_deg*PhlF, # PhlF
                k_tx*(hill(True, VanA1, k_down, n)+hill(True, Gal4, k_down, n)) \
                    - k_deg*VanR, #VanR
                k_tx*(hill(True, TgtA1, k_down, n)+hill(True, Gal4, k_down, n)) \
                    - k_deg*ScbR, #ScbR
                k_tx*(hill(False, PhlF, k_down, n)+hill(False, VanR, k_down, n)) \
                    - k_deg*QF2, #QF2

                # Output
                k_tx*hill(True, QF2, k_down, n) - k_deg*GFP, # GFP
                k_tx*hill(False, ScbR, k_down, n) - k_deg*mCh # mCh
            ]

            return dydt
        
        return model

    # def _build_are_hse_model(self) -> Callable:
    #     # by convention, input 1 will be are
    #     # by convention, input 2 will be hse
    #     self.n_odes=10

    #     def model(t: float, y):
    #         VanA1, TVP64, TetR, CymR, \
    #         TgtA1, Gal4, QF2, \
    #         ScbR, GFP, mCh = y

    #         k_d: NDArray = self._get_k_ds(n_k_ds=4)
    #         k_tx = self.sys_config['K_TX']
    #         k_deg = self.sys_config['K_DEG']
    #         n = self.sys_config['N_COOP']
    #         k_down = self.sys_config['K_D_DOWN']

    #         i1 = self.input1
    #         i2 = self.input2
    #         t=int(t)

    #         dydt = [
    #             # Input layer
    #             # k_tx*hill(True, i1[t], k_d[0], n) - k_deg*VanA1, #VanA1
    #             # k_tx*hill(True, i2[t], k_d[1], n) - k_deg*TVP64, #TVP64
    #             # k_tx*hill(True, i1[t], k_d[2], n) - k_deg*TetR, #TetR
    #             # k_tx*hill(True, i2[t], k_d[3], n) - k_deg*CymR, #CymR

    #             k_tx*hill(True, i1[t], k_d[0], n) - k_deg*VanA1, #VanA1
    #             k_tx*hill(True, i2[t], k_d[1], n) - k_deg*TVP64, #TVP64
    #             k_tx*hill(True, i1[t], k_d[2], n) - k_deg*TetR, #TetR
    #             k_tx*hill(True, i2[t], k_d[3], n) - k_deg*CymR, #CymR
                
    #             # Not layer
    #             k_tx*hill(False, TetR, k_down, n) - k_deg*TgtA1, #TgtA1
    #             k_tx*hill(False, CymR, k_down, n) - k_deg*Gal4, #Gal4

    #             # AND / XOR layer
    #             k_tx*(hill(True, TVP64, k_down, n)+hill(True, TgtA1, k_down, n)) \
    #             + k_tx*(hill(True, VanA1, k_down, n)+hill(True, Gal4, k_down, n)) \
    #                 - k_deg*QF2, #QF2
    #             k_tx*(hill(True, TgtA1, k_down, n)+hill(True, Gal4, k_down, n)) \
    #                 - k_deg*ScbR, #ScbR
                
    #             # Output
    #             k_tx*hill(True, QF2, k_down, n) - k_deg*GFP, # GFP
    #             k_tx*hill(False, ScbR, k_down, n) - k_deg*mCh # mCh
    #         ]

    #         return dydt
        
    #     return model
            