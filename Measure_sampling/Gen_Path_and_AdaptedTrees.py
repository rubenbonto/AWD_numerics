import os
import sys
import numpy as np

# Add Trees path to sys.path
trees_path = os.path.abspath('/Users/rubenbontorno/Documents/Master_Thesis/Code/AWD_numerics/Trees')
kmeans_meas_path = os.path.abspath('/Users/rubenbontorno/Documents/Master_Thesis/Code/AWD_numerics/Adapted_empirical_measure')

if trees_path not in sys.path:
    sys.path.append(trees_path)

if kmeans_meas_path not in sys.path:
    sys.path.append(kmeans_meas_path)

# Now import modules after adding Trees to sys.path
from Build_trees_from_paths import build_tree_from_paths

from AEM_kMeans import empirical_k_means_measure
import time



def generate_brownian_motion(num_paths=3000, time_steps=4, scale=1, return_time=False):
    """
    Generates sample paths of Brownian motion.

    Parameters:
    - num_paths (int): Number of sample paths.
    - time_steps (int): Number of time steps per path.
    - scale (float): Scaling factor for the standard deviation.
    - return_time (bool): Whether to return the time taken for generation.

    Returns:
    - np.ndarray: Generated Brownian motion sample paths.
    - float (optional): Time taken to generate sample paths.
    """
    start_time = time.time()
    sample_paths = np.zeros((num_paths, time_steps))
    for i in range(num_paths):
        sample_paths[i, 0] = 0  # Initial point
        for t in range(1, time_steps):
            sample_paths[i, t] = np.random.normal(loc=sample_paths[i, t-1], scale=scale)
    end_time = time.time()
    
    if return_time:
        return sample_paths, end_time - start_time
    return sample_paths


def generate_financial_model_paths(num_paths=3000, time_steps=4, scale=1, mean_reversion=0.1, volatility=0.2, drift=0.05, return_time=False):
    """
    Generates sample paths from a financial model based on a mean-reverting stochastic process.

    Parameters:
    - num_paths (int): Number of sample paths.
    - time_steps (int): Number of time steps per path.
    - scale (float): Scaling factor for the standard deviation.
    - mean_reversion (float): Speed of mean reversion (theta in the Ornstein-Uhlenbeck process).
    - volatility (float): Volatility term for randomness.
    - drift (float): Long-term drift component.
    - return_time (bool): Whether to return the time taken for generation.

    Returns:
    - np.ndarray: Generated financial model sample paths.
    - float (optional): Time taken to generate sample paths.
    """
    start_time = time.time()
    sample_paths = np.zeros((num_paths, time_steps))
    
    # Ensure all sample paths start with the same initial value
    initial_value = 0  # Fixed initial value for all paths
    sample_paths[:, 0] = initial_value
    
    for i in range(num_paths):
        for t in range(1, time_steps):
            previous_value = sample_paths[i, t-1]
            mean_reverting_term = mean_reversion * (drift - previous_value)
            random_shock = np.random.normal(loc=0, scale=volatility)
            sample_paths[i, t] = previous_value + mean_reverting_term + random_shock
    
    end_time = time.time()
    
    if return_time:
        return sample_paths, end_time - start_time
    return sample_paths


def generate_adapted_tree(num_paths=3000, time_steps=4, scale=1, use_weights=1, model='brownian', return_times=False):
    """
    Generates an adapted tree from sample paths based on the selected stochastic model.

    Parameters:
    - num_paths (int): Number of sample paths.
    - time_steps (int): Number of time steps per path.
    - scale (float): Scaling factor for the standard deviation.
    - use_weights (int): Whether to use weights in the clustering step.
    - model (str): Choice between 'brownian' and 'financial'.
    - return_times (bool): Whether to return timing details.

    Returns:
    - TreeNode: Root node of the generated tree.
    - tuple (optional): Timings for sample generation, measure adaptation, and tree building.
    """
    # Generate sample paths based on the selected model
    start_sample_time = time.time()
    if model == 'brownian':
        sample_paths, sample_time = generate_brownian_motion(num_paths, time_steps, scale, return_time=True)
    elif model == 'financial':
        sample_paths, sample_time = generate_financial_model_paths(num_paths, time_steps, scale, return_time=True)
    else:
        raise ValueError("Invalid model selection. Choose either 'brownian' or 'financial'.")
    end_sample_time = time.time()
    
    # Adapt the measure
    start_adapt_time = time.time()
    new_sample_paths, new_weights = empirical_k_means_measure(sample_paths, use_weights=use_weights)
    end_adapt_time = time.time()
    adapt_time = end_adapt_time - start_adapt_time
    
    # Build tree from the adapted sample paths
    start_tree_time = time.time()
    tree_root = build_tree_from_paths(new_sample_paths, new_weights)
    end_tree_time = time.time()
    tree_time = end_tree_time - start_tree_time
    
    if return_times:
        return tree_root, (sample_time, adapt_time, tree_time)
    return tree_root