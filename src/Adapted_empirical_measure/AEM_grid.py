import numpy as np


# replicate the grid as done in: Estimating Processes in Adapted Wasserstein Distance but not with T+1 but with sqrt n !
def empirical_grid_measure(data, N=None, use_weights=False):
    """
    Computes an empirical measure approximation of sample paths using grid quantization.
    Instead of k-means, a single uniform grid (based on the overall minimum and maximum values
    of the data across all samples and time steps) is used to quantize the sample paths.

    IMPORTANT: The first time step (column) is assumed to be identical across samples and is left unchanged.

    Parameters:
    - data (np.ndarray): A (num_samples, time_steps) array representing sample paths.
    - N (int, optional): Number of grid points. Defaults to int(round(sqrt(num_samples))).
    - use_weights (bool): If True, returns weights for each unique quantized sample path.

    Returns:
    - np.ndarray: New sample paths after quantization.
    - list (optional): Weights associated with each unique sample path if use_weights is True.
    """
    num_samples, time_steps = data.shape
    if N is None:
        N = int(np.round(np.sqrt(num_samples)))

    # Compute global min and max over the entire dataset (maybe I can try to optimize grid point a bit more)
    global_min = data.min()
    global_max = data.max()

    # Create a common grid for all time steps
    grid = np.linspace(global_min, global_max, N)

    # For each value in the data, find the closest grid point
    quantized_data = grid[np.abs(data[..., None] - grid).argmin(axis=-1)]

    # Restore the first time step to its original value!
    quantized_data[:, 0] = data[:, 0]

    if not use_weights:
        return quantized_data

    # If weights are needed, aggregate identical sample paths and compute their frequencies
    unique_paths, indices, counts = np.unique(
        quantized_data, axis=0, return_inverse=True, return_counts=True
    )
    weights = counts / num_samples
    return unique_paths, weights


# replicate the grid as done in: Estimating Processes in Adapted Wasserstein
def uniform_empirical_grid_measure(data, delta_n=None, use_weights=False):
    """
    Computes an empirical measure approximation of sample paths using grid quantization.
    If delta_n is not provided, it is initialized using the formula:

        delta_n = 1/(num_path**(1/t))

    where num_path and t are the dimensions of the input data. Note that since the first
    time step is trivial (identical across samples), the non-trivial time steps count is (t-1)
    and we add one for the formula, yielding t in the exponent denominator.

    Parameters:
    - data (np.ndarray): A (num_samples, time_steps) array representing sample paths (assumed to be iid).
    - delta_n (float, optional): Grid spacing used for quantization. If None, it is initialized
                                 using the formula above.
    - use_weights (bool): If True, returns weights for each unique quantized sample path.

    Returns:
    - quantized_data (np.ndarray): New sample paths after quantization.
    - (optional) unique_paths (np.ndarray) and weights (np.ndarray): If use_weights is True,
      returns the unique quantized paths and their corresponding frequencies.
    """
    num_path, t = data.shape

    if delta_n is None:
        delta_n = 1 / (
            num_path ** (1 / t)
        )  # It is for d = 1 the formula is 1/N^r where r is 1/(T+1) but as we assume deterministic first step T+1 = "t".

    # Define the grid function (round to nearest multiple of delta_n)
    grid_func = lambda x: np.floor(x / delta_n + 0.5) * delta_n

    # Quantize the data using the grid function
    quantized_data = grid_func(data)

    # Restore the first time step (assumed to be identical across samples) to its original value
    quantized_data[:, 0] = data[:, 0]

    if not use_weights:
        return quantized_data
    else:
        # Compute unique quantized sample paths along with their counts.
        unique_paths, indices, counts = np.unique(
            quantized_data, axis=0, return_inverse=True, return_counts=True
        )
        weights = counts / num_path
        return unique_paths, weights
