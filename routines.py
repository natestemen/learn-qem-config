import json
import numpy as np
import re
from experiment import make_experiment_list, batch_execute

DEFAULT_METHODS = ["global", "local_random", "local_all", "identity_scaling"]
DEFAULT_EXTRAPS = ["linear", "richardson", "polynomial", "exponential", "poly-exp", "adaptive-exp"]
DEFAULT_SEARCH_FACTORS = [[1, 2, 3], [1, 3, 5], [1, 5, 9]]
BASE_FACTORS = [1, 3, 5, 7]

def _print_result(best_config, ideal_result, best_result, error):

    print("BEST CONFIGURATION:")
    
    # Json manipulation for better print
    json_str = json.dumps(best_config, indent=4)
    json_str = re.sub(r'\[\s+([^]]+?)\s+\]', lambda x: "[" + " ".join(x.group(1).split()) + "]", json_str)
    
    print(json_str)
    print(f"Ideal result:      {ideal_result}")
    print(f"Best result:       {best_result}")
    print(f"Absolute error:    {error}")

def _get_default_batch():
    return {
        "noise_scaling_factors": DEFAULT_SEARCH_FACTORS,
        "noise_scaling_method": DEFAULT_METHODS,
        "extrapolation": DEFAULT_EXTRAPS
    }


def brute_force_search(circ, exe, ideal_result, zne_batch_test=None):

    # Set the default search space if not provided
    if zne_batch_test is None:
        zne_batch_test = _get_default_batch()
        
    exp_results = batch_execute(zne_batch_test, circ, exe)

    # Error handling
    results_array = np.array(exp_results, dtype=float)
    num_failures = np.isnan(results_array).sum()
    if num_failures > 0:
        print(f"{num_failures} configurations failed due to extrapolation errors")

    configs = make_experiment_list(zne_batch_test)
    errors = np.abs(np.array(exp_results) - ideal_result)
    best_idx = np.nanargmin(errors)

    _print_result(best_config=configs[best_idx], ideal_result=ideal_result, best_result=exp_results[best_idx], error=errors[best_idx])


def light_brute_force_search(circ, exe, ideal_result, zne_batch_test=None, max_iter=10, printing=True):

    # Set the default search space if not provided
    if zne_batch_test is None:
        zne_batch_test = _get_default_batch()

    # Starting configuration: first value of each parameter
    current_config = {
        "noise_scaling_factors": zne_batch_test["noise_scaling_factors"][0],
        "noise_scaling_method": zne_batch_test["noise_scaling_method"][0],
        "extrapolation": zne_batch_test["extrapolation"][0]
    }

    iter = 0
    converged = False
    max_reached = False
    
    best_result_value = np.nan

    while not converged and not max_reached:
        iter += 1
        previous_config = current_config.copy()
        
        # Optimize locally each parameter
        for param_name in ["noise_scaling_factors", "noise_scaling_method", "extrapolation"]:
            
            # Skip if there is only 1 option for this parameter 
            if len(zne_batch_test[param_name]) <= 1:
                continue

            # Set up an experiment with all parameters fixed except one
            batch_dict = {
                "noise_scaling_factors": [current_config["noise_scaling_factors"]],
                "noise_scaling_method": [current_config["noise_scaling_method"]],
                "extrapolation": [current_config["extrapolation"]]
            }
            batch_dict[param_name] = zne_batch_test[param_name]

            # Run this batch of experiments
            results = batch_execute(batch_dict, circ, exe)
            
            # Error handling
            # Convert to numpy array to handle NaNs correctly
            results_array = np.array(results, dtype=float)
            errors = np.abs(results_array - ideal_result)
            # If all experiments in this batch failed (all are NaN), skip update
            if np.all(np.isnan(errors)):
                continue

            best_idx = np.nanargmin(errors)
            
            best_value_for_param = zne_batch_test[param_name][best_idx]
            current_config[param_name] = best_value_for_param
            best_result_value = results[best_idx]

        # Check if the configuration has changed
        if json.dumps(previous_config, sort_keys=True) == json.dumps(current_config, sort_keys=True):
            converged = True

        # Limit on iterations
        if iter == max_iter:
            max_reached = True

    final_error = np.nan
    if not np.isnan(best_result_value):
        final_error = abs(best_result_value - ideal_result)

    # Only print results when it is used as an independet routine
    if printing:
        if max_reached:
            print("Max number of iterations reached")
        if np.isnan(best_result_value):
            print("\nLight Brute Force failed: No valid configuration found.")
        else:
            _print_result(best_config=current_config, ideal_result=ideal_result, best_result=best_result_value, error=final_error)
            
    return current_config, best_result_value, final_error


def adaptive_search(circ, exe, ideal_result, zne_batch_test=None, tolerance=1e-3, max_factors=10):

    stop_reason=0

    # Setup searching space, use user methods but force the default scaling factors    
    if zne_batch_test is None:
        methods_to_test = DEFAULT_METHODS
        extraps_to_test = DEFAULT_EXTRAPS
    else:
        methods_to_test = zne_batch_test.get("noise_scaling_method", DEFAULT_METHODS)
        extraps_to_test = zne_batch_test.get("extrapolation", DEFAULT_EXTRAPS)

    # Force the base factors for the initial search
    initial_batch = {
        "noise_scaling_factors": [BASE_FACTORS],
        "noise_scaling_method": methods_to_test,
        "extrapolation": extraps_to_test
    }

    # Find best methods using light_brute_force 
    best_config, best_val, best_err = light_brute_force_search( circ, exe, ideal_result, initial_batch, max_iter=10, printing=False)

    # Adaptive expansion of scale factors

    current_factors = list(best_config["noise_scaling_factors"])
    current_best_error = best_err
    current_best_result = best_val
    
    next_factor = 9
    
    while len(current_factors) < max_factors:
        
        # Add one scale factor
        new_factors_list = current_factors + [next_factor]
        
        # Build single experiment for validation
        test_batch = {
            "noise_scaling_factors": [new_factors_list],
            "noise_scaling_method": [best_config["noise_scaling_method"]],
            "extrapolation": [best_config["extrapolation"]]
        }

        # Execute
        results = batch_execute(test_batch, circ, exe)
        new_val = results[0]
        
        # Check if there is any reasonable improvement
        new_err = abs(new_val - ideal_result)
        improvement = current_best_error - new_err
        if new_err > current_best_error:
            stop_reason=1
            break 
        if improvement < tolerance:
            stop_reason=2
            break
        
        # Update scale factor list
        current_factors = new_factors_list
        current_best_error = new_err
        current_best_result = new_val
        best_config["noise_scaling_factors"] = current_factors
        next_factor += 2

    # Print the stopping criteria
    if stop_reason==1:
        print("Adaptive search stopped: error worsened")
    elif stop_reason==2:
        print("Adaptive search: Convergence reached")
    elif stop_reason==0:
        print("Adaptive search: Maximum limit of scale factors reached ")

    _print_result(best_config, ideal_result, current_best_result, current_best_error)