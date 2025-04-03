import numpy as np

# In practice, we focus more on uniform grid measures. Both functions achieve the same goal using different approaches:
# - One function fixes the number of grid support points.
# - The other function fixes the grid size.
# Using a fixed grid size and changing the number is in practice what we will be doing (as it leads to more stable results and explicitly gives you the grid size which you can think of as an error bound on your approximation).


def uniform_empirical_grid_measure(data, delta_n=None, use_weights=False):
    """
    Computes an empirical measure approximation using grid quantization.

    Parameters:
    - data (np.ndarray): (num_samples, time_steps) array representing sample paths.
    - delta_n (float, optional): Grid spacing. Defaults to 1 / num_path^(1/time_steps).
    - use_weights (bool): If True, returns weights for unique paths.

    Returns:
    - np.ndarray: Quantized sample paths.
    - np.ndarray (optional): Weights if use_weights is True.
    """
    num_path, time_steps = data.shape
    delta_n = delta_n or 1 / (num_path ** (1 / time_steps))

    quantized_data = np.round(data / delta_n) * delta_n
    quantized_data[:, 0] = data[:, 0]

    if not use_weights:
        return quantized_data

    unique_paths, counts = np.unique(quantized_data, axis=0, return_counts=True)
    weights = counts / num_path
    return unique_paths, weights


def empirical_grid_measure(data, N=None, use_weights=False):
    """
    Computes an empirical measure approximation of sample paths using a common grid.

    Parameters:
    - data (np.ndarray): (num_samples, time_steps) array representing sample paths.
    - N (int, optional): Number of grid points. Defaults to sqrt(num_samples).
    - use_weights (bool): If True, returns weights for unique paths.

    Returns:
    - np.ndarray: Quantized sample paths.
    - np.ndarray (optional): Weights if use_weights is True.
    """
    num_samples, time_steps = data.shape
    N = N or int(np.sqrt(num_samples))

    global_min, global_max = data.min(), data.max()
    grid = np.linspace(global_min, global_max, N)

    quantized_data = grid[np.abs(data[..., None] - grid).argmin(axis=-1)]
    quantized_data[:, 0] = data[:, 0]

    if not use_weights:
        return quantized_data

    unique_paths, counts = np.unique(quantized_data, axis=0, return_counts=True)
    weights = counts / num_samples
    return unique_paths, weights
