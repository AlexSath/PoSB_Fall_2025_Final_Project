import numpy as np
import itertools

def analyze_single_var_axis(input_data: dict, key: str, default_params: dict, start_time: int=None):
   """
   Args:
    - input_data [dict]: will be the dictionary containing all keys. Dictionary must have:
       - 'n_ranges' key indicating the number of parameter ranges in the dictionary
       - 'combinations' key which will contain the index of all combinations
       - 'solutions' key which will contain a 3-dimensional array containing all solutions of shape (N, COMPOSITOR, T)
    - key [str]: will be a string containing a key that the function should iterate through for the 1D analysis
    - default_params [dict]: the default parameters that must be matched to the parameter space of the single key

   Returns:
    - key [str]: equal to the input value 'key'.
    - key_param_range [NDArray, (K,)]: The parameter ranges that 'key' took on in the 'input_data'. K is variable because the number could be variable.
    - delay [NDarray, (K,)]: Delay values relative to t=0 for peak final output pulse time with provided parameters
    - amplitude [NDarray, (K,)]: Maximum mplitude for peak final output pulse
   """
   assert set(['n_ranges', 'combinations', 'solutions']) <= set(list(input_data.keys())), f"'input_data' must contain 'n_ranges, 'combinations', and 'solutions'"
   n_ranges = input_data['n_ranges']
   input_data_combinations = input_data['combinations']
   input_data_solutions = input_data['solutions']
   # input_data_output_index = input_data_solutions.shape[1] - 1
   data_ranges_keys = list(input_data.keys())[:n_ranges]
   assert key in data_ranges_keys, f"Found {data_ranges_keys} in 'input_data'. Provided key '{key}' is missing."
   
   key_ranges = input_data[key]
   target_parameter_combinations = np.zeros(shape=(len(key_ranges), n_ranges))
   for kdx, k in enumerate(key_ranges): # looping through the parameter space of the specified key
      this_parameter_combination = np.zeros(shape=(n_ranges,))
      for ddx, other_ranges in enumerate(data_ranges_keys): # looping through the other keys in the space
         if other_ranges == key: # if this is the key we are looping through, ass the 
            this_parameter_combination[ddx] = k
         else: # if this is not the key, take the default parameter value from 'default_params'
            this_parameter_combination[ddx] = default_params[other_ranges]
      target_parameter_combinations[kdx,:] = this_parameter_combination

   desired_indices = np.zeros(shape=(target_parameter_combinations.shape[0],), dtype=int)
   for pdx, parameter_combination in enumerate(target_parameter_combinations):
      this_idx = np.where((input_data_combinations == parameter_combination).all(axis=1) == 1)[0][0]
      desired_indices[pdx] = this_idx
   
   delays = np.zeros(shape=(target_parameter_combinations.shape[0],))
   amplitudes = np.zeros(shape=(target_parameter_combinations.shape[0],))
   for sdx, solution_idx in enumerate(desired_indices):
      delays[sdx] = np.argmax(input_data_solutions[solution_idx,-1,:])
      if start_time is not None: delays[sdx] -= start_time
      amplitudes[sdx] = np.max(input_data_solutions[solution_idx,-1,:])

   return key, key_ranges, delays, amplitudes


def analyze_multiple_var_axis(input_data: dict, keys: list[str], default_params: dict, start_time: int=None):
   """
   Args:
    - input_data [dict]: will be the dictionary containing all keys. Dictionary must have:
       - 'n_ranges' key indicating the number of parameter ranges in the dictionary (N, )
       - 'combinations' key which will contain the index of all combinations of shape (I, N)
       - 'solutions' key which will contain a 3-dimensional array containing all solutions of shape (I, COMPOSITORS, T)
    - keys [list[str]]: will be a string containing a key that the function should iterate through for the 2D analysis
    - default_params [dict]: the default parameters that must be matched to the parameter space of the single key

   Returns:
    - keys [list[str]]: equal to the input values of key1 and key2.
    - key_param_range [NDArray, (L, K)]: L = number of input keys. K is the number of variations in each key.
    - delay [NDarray, (K,...,K)]: Delay values relative to t=0 for peak final output pulse time with provided parameters. L = number of K dimensions.
    - amplitude [NDarray, (K,...,K)]: Maximum mplitude for peak final output pulse. L = number of K dimensions
   """
   # assert len(keys) <= 2, f"Function 'analyze_multiple_var_axis' only tested up to 2 keys!"
   assert set(['n_ranges', 'combinations', 'solutions']) <= set(list(input_data.keys())), f"'input_data' must contain 'n_ranges, 'combinations', and 'solutions'"
   n_ranges = input_data['n_ranges']
   data_ranges_keys = list(input_data.keys())[:n_ranges]
   for key in keys: assert key in data_ranges_keys, f"Found {data_ranges_keys} in 'input_data'. Provided key '{key}' is missing."
   input_data_combinations = input_data['combinations']
   input_data_solutions = input_data['solutions']
   
   # Getting the desired indices (hard)
   key_ranges = np.stack([input_data[k] for k in keys], axis=1).transpose(1, 0) # Will have shape (L, K)
   delays = np.zeros(shape=tuple(key_ranges.shape[0]*[key_ranges.shape[1]])) # shape (K,...,K)
   amplitudes = np.zeros(shape=tuple(key_ranges.shape[0]*[key_ranges.shape[1]])) # shape (K,...,K)
   for combination in itertools.product(*[r for r in key_ranges]):

      # get indices of (K,...,K) dimensions in the output array
      this_slice = []
      for cdx, comb_value in enumerate(combination):
         this_slice.append(np.where(key_ranges[cdx,...] == comb_value)[0][0])
      # this_slice.append(None) # None required here because last dim on slice is N in the parameter combinations array
      this_slice = np.s_[*this_slice]
      
      # get the full parameter combination (size N) of provided indices
      this_parameter_combination = np.zeros(shape=(n_ranges,))
      for kdx, key in enumerate(data_ranges_keys):
         if key in keys:
            key_comb_index = keys.index(key)
            this_parameter_combination[kdx] = combination[key_comb_index]
         else:
            this_parameter_combination[kdx] = default_params[key]

      # Finding correct index and outputting delays and amplitudes for desired parameter combination
      this_idx = np.where((input_data_combinations == this_parameter_combination).all(axis=-1))[0][0]
      delays[this_slice] = np.argmax(input_data_solutions[this_idx,-1,:])
      if start_time is not None: delays[this_slice] -= start_time
      amplitudes[this_slice] = np.max(input_data_solutions[this_idx,-1,:])

   return keys, key_ranges, delays, amplitudes
   
