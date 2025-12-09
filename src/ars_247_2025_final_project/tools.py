from collections.abc import Callable

import numpy as np

def transpose_dict(this_dict) -> dict:
    new_dict = {}
    for key, value in this_dict.values():
        new_dict[value] = key
    return new_dict

def gaussian_pulse(t, peak_time, peak_value, width) -> float:
    return peak_value * np.exp(-((t - peak_time) ** 2) / (2 * width ** 2))

def double_sigmoid_pulse(t, t_on, t_off, peak_value, comp) -> float:
    rise = 1 / (1 + np.exp(-comp * (t - t_on)))
    fall = 1 / (1 + np.exp(-comp * (t_off - t)))
    return peak_value * rise * fall

def pulse_model(pulse_type: str, params: dict) -> Callable:
    if pulse_type == 'gaussian': 
        assert params is not None, f"If promoter is 'gaussian' pulse, must provide pulse parameters"
        assert set(list(params.keys()) + ['t']) == set(['t', 'peak_time', 'peak_value', 'width'])
        this_pulse = gaussian_pulse
    elif pulse_type == 'sigmoid':
        assert params is not None, f"If promoter is 'sigmoid' pulse, must provide pulse parameters"
        assert set(list(params.keys()) + ['t']) == set(['t', 't_on', 't_off', 'peak_value', 'comp'])
        this_pulse = double_sigmoid_pulse
    else: raise ValueError(f"Pulse type {pulse_type} not recognized. Try 'gaussian' or 'sigmoid'")

    def model(t, y):
        this_pulse_value = this_pulse(t, **params)

        dydt = [
            this_pulse_value
        ]

        return dydt
    
    return model