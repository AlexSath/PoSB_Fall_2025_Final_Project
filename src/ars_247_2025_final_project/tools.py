from collections.abc import Callable

import numpy as np
from scipy.interpolate import interp1d

from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
# from numpy import vectorize

def transpose_dict(this_dict) -> dict:
    new_dict = {}
    for key, value in this_dict.values():
        new_dict[value] = key
    return new_dict

def gaussian_pulse(t, peak_time, peak_value, width):
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

def hill(act: bool, g: float, k_d: float, n: float):
    if act:
        return g**n / (k_d**n + g**n)
    else:
        return k_d**n / (k_d**n + g**n)
    

def solve_pulse(t_start, t_end, pulse_type, pulse_params, dt=1) -> OdeResult:
    model = pulse_model(pulse_type, pulse_params)
    soln = solve_ivp(
        fun = model,
        t_span = [t_start, t_end],
        y0 = [0],
        t_eval = np.arange(t_start, t_end, dt)
    )
    return soln


def geometric_ou_noise(theta, sigma, mu, t_total, dt_sde, dt_output):
    """
    Generates Geometric Ornstein-Uhlenbeck noise - no negative transcriptional noise!.
    The process evolves in log-space, then exponentiates.
    
    Args:
     - theta: Rate of mean reversion.
     - sigma: Volatility in log-space.
     - mu_log: Long-term mean in log-space (output centers around exp(mu_log)).
     - t_total, dt_sde, dt_output: Time parameters as before.
    """
    
    t_sde = np.arange(0, t_total + dt_sde, dt_sde)
    n_steps_sde = len(t_sde)
    
    # Evolve in log-space
    log_noise = np.zeros(n_steps_sde)
    log_noise[0] = mu  # Start at the mean
    
    random_increments = np.random.normal(0, 1, n_steps_sde) * np.sqrt(dt_sde)
    
    for i in range(1, n_steps_sde):
        drift = theta * (mu - log_noise[i-1]) * dt_sde
        diffusion = sigma * random_increments[i]
        log_noise[i] = log_noise[i-1] + drift + diffusion
    
    
    # ou_noise_sde = np.maximum(log_noise, 0)

    # def softplus(x, sharpness=1.0):
    #     baseline = np.log(2) / sharpness  # softplus(0) for given sharpness
    #     return np.log(1 + np.exp(sharpness * x)) / sharpness - baseline

    ou_noise_sde = np.abs(log_noise)
    
    # Interpolation
    interpolation_function = interp1d(t_sde, ou_noise_sde, kind='cubic')
    t_output = np.arange(0, t_total, dt_output)
    ou_noise_interp = interpolation_function(t_output)
    
    return t_output, ou_noise_interp