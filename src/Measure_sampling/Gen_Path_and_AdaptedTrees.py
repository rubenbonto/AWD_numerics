import numpy as np
import time

# Import from src (no need to modify sys.path)
from trees.Build_trees_from_paths import build_tree_from_paths
from adapted_empirical_measure.AEM_kMeans import (
    empirical_k_means_measure,
    empirical_k_means_measure_new,
    empirical_k_means_measure_grid,
)
from adapted_empirical_measure.AEM_grid import empirical_grid_measure


def generate_brownian_motion(
    num_paths=3000, time_steps=4, x_init=0, scale=1, return_time=False
):
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
        sample_paths[i, 0] = x_init  # Initial point
        for t in range(1, time_steps):
            sample_paths[i, t] = np.random.normal(
                loc=sample_paths[i, t - 1], scale=scale
            )
    end_time = time.time()

    if return_time:
        return sample_paths, end_time - start_time
    return sample_paths


#### NOT USED AS NO IDEA WHAT'S THE REAL DISTANCE COMPARED TO WHAT WE ESTIMATE
def generate_financial_model_paths(
    num_paths=3000,
    time_steps=4,
    scale=1,
    mean_reversion=0.1,
    volatility=0.2,
    drift=0.05,
    return_time=False,
):
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
            previous_value = sample_paths[i, t - 1]
            mean_reverting_term = mean_reversion * (drift - previous_value)
            random_shock = np.random.normal(loc=0, scale=volatility)
            sample_paths[i, t] = previous_value + mean_reverting_term + random_shock

    end_time = time.time()

    if return_time:
        return sample_paths, end_time - start_time
    return sample_paths


#### SAMPLE PATHS AND GENERATE AND "ADAPTED" MEASURE FROM IT AND FINALLY CREAT A TREE FROM THIS MEASURE
def generate_adapted_tree(
    num_paths=3000,
    time_steps=4,
    x_init=0,
    scale=1,
    use_weights=1,
    model="brownian",
    return_times=False,
    cluster_method="kmeans",
    N=None,
):
    """
    Generates an adapted tree from sample paths based on the selected stochastic model.

    Parameters:
    - num_paths (int): Number of sample paths.
    - time_steps (int): Number of time steps per path.
    - x_init (float): Initial value for sample paths.
    - scale (float): Scaling factor for the standard deviation.
    - use_weights (int): Whether to use weights in the clustering step.
    - model (str): Choice between 'brownian' and 'financial'.
    - return_times (bool): Whether to return timing details.
    - cluster_method (str): Clustering method to use ('kmeans' or 'grid').
    - N (int): Number of grid points for the grid method. If None, defaults to int(round(sqrt(num_paths))).

    Returns:
    - TreeNode: Root node of the generated tree.
    - tuple (optional): Timings for sample generation, measure adaptation, and tree building.
    """
    # Generate sample paths based on the selected model
    start_sample_time = time.time()
    if model == "brownian":
        sample_paths, sample_time = generate_brownian_motion(
            num_paths, time_steps, x_init, scale, return_time=True
        )
    elif model == "financial":
        sample_paths, sample_time = generate_financial_model_paths(
            num_paths, time_steps, scale, return_time=True
        )
    else:
        raise ValueError(
            "Invalid model selection. Choose either 'brownian' or 'financial'."
        )
    end_sample_time = time.time()

    # Adapt the measure
    start_adapt_time = time.time()
    if cluster_method == "kmeans":
        new_sample_paths, new_weights = empirical_k_means_measure(
            sample_paths, use_weights=use_weights
        )
    elif cluster_method == "kmeans_new":
        new_sample_paths, new_weights = empirical_k_means_measure_new(
            sample_paths, use_weights=use_weights
        )
    elif cluster_method == "kmeans_grid":
        new_sample_paths, new_weights = empirical_k_means_measure_grid(
            sample_paths, use_weights=use_weights
        )
    elif cluster_method == "grid":
        if N is None:
            N = int(np.round(np.sqrt(num_paths)))
        new_sample_paths, new_weights = empirical_grid_measure(
            sample_paths, N=N, use_weights=use_weights
        )
    else:
        raise ValueError("Invalid cluster method. Choose either 'kmeans' or 'grid'.")
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
