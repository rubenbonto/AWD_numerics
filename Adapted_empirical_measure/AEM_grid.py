import numpy as np



# replicate the grid as done in: Estimating Processes in Adapted Wasserstein Distance but not with T+1 but with sqrt n
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
    
    # Compute global min and max over the entire dataset
    global_min = data.min()
    global_max = data.max()
    
    # Create a common grid for all time steps
    grid = np.linspace(global_min, global_max, N)
    
    # For each value in the data, find the closest grid point using broadcasting
    quantized_data = grid[np.abs(data[..., None] - grid).argmin(axis=-1)]
    
    # Restore the first time step to its original value (it remains unchanged)
    quantized_data[:, 0] = data[:, 0]
    
    if not use_weights:
        return quantized_data

    # If weights are needed, aggregate identical sample paths and compute their frequencies
    unique_paths, indices, counts = np.unique(quantized_data, axis=0, 
                                              return_inverse=True, return_counts=True)
    weights = counts / num_samples
    return unique_paths, weights

# the issue bellow is that it modify the first value which is something we do no want
def empirical_grid_measure_old_old(data, N=None, use_weights=False):
    """
    Computes an empirical measure approximation of sample paths using grid quantization.
    Instead of k-means, a single uniform grid (based on the overall minimum and maximum values 
    of the data across all samples and time steps) is used to quantize the sample paths.

    Parameters:
    - data (np.ndarray): A (num_samples, time_steps) array representing sample paths.
    - N (int, optional): Number of grid points. Defaults to int(round(sqrt(num_samples))).
    - use_weights (bool): If True, returns weights for each unique quantized sample path.

    Returns:
    - np.ndarray: New sample paths after quantization.
    - list (optional): Weights associated with each unique sample path if use_weights=True.
    """
    num_samples, time_steps = data.shape
    if N is None:
        N = int(np.round(np.sqrt(num_samples)))
    
    # Compute global min and max over the entire dataset
    # (Alternatively, one could compute per-sample min and max and then take the overall min and max)
    global_min = data.min()
    global_max = data.max()
    
    # Create a common grid for all time steps
    grid = np.linspace(global_min, global_max, N)
    
    # For each value in the data, find the closest grid point
    # This uses broadcasting: data[..., None] creates an extra dimension so that we can subtract the grid
    quantized_data = grid[np.abs(data[..., None] - grid).argmin(axis=-1)]
    
    if not use_weights:
        return quantized_data

    # If weights are needed, aggregate identical sample paths and compute their frequencies
    # Using np.unique with axis=0 returns unique rows (sample paths) and counts of each unique path
    unique_paths, indices, counts = np.unique(quantized_data, axis=0, return_inverse=True, return_counts=True)
    weights = counts / num_samples
    return unique_paths, weights


## different grid for each time step
def empirical_grid_measure_old(data, N=None, use_weights=False):
    """
    Computes an empirical measure approximation of sample paths using grid quantization.
    Instead of using one global grid, a separate grid is computed for each time step based
    on the minimum and maximum values in that time step.

    Parameters:
    - data (np.ndarray): A (num_samples, time_steps) array representing sample paths.
    - N (int, optional): Number of grid points for each time step. Defaults to int(round(sqrt(num_samples))).
    - use_weights (bool): If True, returns weights for each unique quantized sample path.

    Returns:
    - np.ndarray: New sample paths after quantization.
    - list (optional): Weights associated with each unique sample path if use_weights is True.
    """
    num_samples, time_steps = data.shape
    if N is None:
        N = int(np.round(np.sqrt(num_samples)))
    
    # Initialize an array to hold the quantized sample paths
    quantized_data = np.empty_like(data)
    
    # Process each time step separately
    for t in range(time_steps):
        # Compute min and max for this time step
        col_min = data[:, t].min()
        col_max = data[:, t].max()
        
        # Create a grid of N points between col_min and col_max
        grid = np.linspace(col_min, col_max, N)
        
        # For each sample, find the closest grid point
        differences = np.abs(data[:, t, np.newaxis] - grid)  # Shape: (num_samples, N)
        nearest_idx = differences.argmin(axis=1)
        quantized_data[:, t] = grid[nearest_idx]
    
    if not use_weights:
        return quantized_data

    # Compute unique paths and their frequencies if weights are requested
    unique_paths, indices, counts = np.unique(quantized_data, axis=0, return_inverse=True, return_counts=True)
    weights = counts / num_samples
    return unique_paths, weights

