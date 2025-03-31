import numpy as np


def multidim_uniform_empirical_grid_measure(data, delta_n=None, use_weights=False):
    if data.ndim == 3:
        num_path, t, d = data.shape
        if delta_n is None:
            delta_n = 1 / (num_path ** (1 / t))
        grid_func = lambda x: np.floor(x / delta_n + 0.5) * delta_n
        quantized_data = grid_func(data)
        quantized_data[:, 0, :] = data[:, 0, :]
        if not use_weights:
            return quantized_data
        else:
            # Convert each sample path (2D array) into a hashable tuple of tuples.
            quantized_paths = [
                tuple(map(tuple, quantized_data[i])) for i in range(num_path)
            ]
            # Preserve the original order.
            unique_paths_list = []
            counts_list = []
            for p in quantized_paths:
                if p in unique_paths_list:
                    counts_list[unique_paths_list.index(p)] += 1
                else:
                    unique_paths_list.append(p)
                    counts_list.append(1)
            weights = np.array(counts_list) / num_path
            # Convert unique paths back to numpy arrays.
            unique_paths = np.array([np.array(up) for up in unique_paths_list])
            return unique_paths, weights
    else:
        raise ValueError("Data must be 3D (d>1).")


import numpy as np
from sklearn.cluster import KMeans


def multidim_empirical_k_means_measure_new(
    data, tol_decimals=6, use_weights=False, delta_n=None, k_start=1
):
    """
    Computes an empirical measure approximation of multi-dimensional sample paths using k-means clustering.
    For each time step t>=1, clustering starts with one cluster and is refined until the maximum
    Euclidean deviation from the assigned cluster center is ≤ delta_n.

    The first time step (t == 0) is assumed to be deterministic across samples and is not clustered.

    Parameters:
    - data (np.ndarray): A (num_samples, time_steps, d) array representing sample paths.
    - tol_decimals (int): Number of decimals to round cluster centers to.
    - use_weights (bool): Whether to weight cluster centers by frequency.
    - delta_n (float, optional): If None, it is initialized as
          1 / (num_samples ** (1 / time_steps)).
          Otherwise, it enforces that the maximum Euclidean deviation (in each time step) is below this threshold.

    Returns:
    - np.ndarray: New sample paths approximating an empirical measure.
    - list (optional): Weights associated with each sample path if use_weights is True.
    """
    if data.ndim != 3:
        raise ValueError("Data must be 3D with shape (num_samples, time_steps, d).")

    num_samples, time_steps, d = data.shape

    if delta_n is None:
        delta_n = 1 / (num_samples ** (1 / time_steps))

    label_list = []
    support_list = []

    # Process each time step independently
    for t in range(time_steps):
        if t == 0:
            # For the deterministic initial condition, use the original value
            unique_val = data[0, 0, :]
            labels = np.zeros(num_samples, dtype=int)
            cluster_centers = np.array([unique_val])
        else:
            # Begin with one cluster and refine until the maximum deviation is within delta_n.
            k = k_start
            while True:
                km = KMeans(n_clusters=k, n_init="auto").fit(data[:, t, :])
                # Round the cluster centers to avoid spurious differences.
                cluster_centers = np.round(km.cluster_centers_, decimals=tol_decimals)
                labels = km.labels_
                # Compute the Euclidean distances for each point from its assigned cluster center.
                distances = np.linalg.norm(
                    data[:, t, :] - cluster_centers[labels], axis=1
                )
                max_distance = distances.max()
                if max_distance <= delta_n or k >= num_samples:
                    break
                else:
                    k += 1
        label_list.append(labels)
        support_list.append(cluster_centers)

    if not use_weights:
        # Reconstruct the quantized sample paths
        output = np.zeros((num_samples, time_steps, d))
        for t in range(time_steps):
            centers = support_list[t]
            labels = label_list[t]
            output[:, t, :] = centers[labels]
        return output
    else:
        # Collapse identical paths and assign weights.
        # To make sample paths hashable, convert each path (a 2D array) into a tuple of tuples.
        unique_paths_dict = {}
        for i in range(num_samples):
            current_path = tuple(
                tuple(support_list[t][label_list[t][i]]) for t in range(time_steps)
            )
            unique_paths_dict[current_path] = unique_paths_dict.get(current_path, 0) + (
                1 / num_samples
            )

        unique_paths_list = [np.array(path) for path in unique_paths_dict.keys()]
        weights_list = list(unique_paths_dict.values())
        unique_paths_array = np.array(unique_paths_list)
        return unique_paths_array, weights_list


import numpy as np
from sklearn.cluster import KMeans
import concurrent.futures


def process_time_step(t, data, tol_decimals, delta_n):
    """
    Process a single time step t of the data.
    """
    num_samples = data.shape[0]
    if t == 0:
        # For the deterministic initial condition
        unique_val = data[0, 0, :]
        labels = np.zeros(num_samples, dtype=int)
        cluster_centers = np.array([unique_val])
    else:
        # Start with one cluster and increase until max distance ≤ delta_n
        k = 1
        while True:
            km = KMeans(n_clusters=k, n_init="auto").fit(data[:, t, :])
            # Round centers to avoid spurious differences
            cluster_centers = np.round(km.cluster_centers_, decimals=tol_decimals)
            labels = km.labels_
            # Compute Euclidean distances from each point to its assigned cluster center
            distances = np.linalg.norm(data[:, t, :] - cluster_centers[labels], axis=1)
            max_distance = distances.max()
            if max_distance <= delta_n or k >= num_samples:
                break
            k += 1
    return t, labels, cluster_centers


def parallel_multidim_empirical_k_means_measure_new(
    data, tol_decimals=6, use_weights=False, delta_n=None
):
    """
    Computes an empirical measure approximation of multi-dimensional sample paths using
    k-means clustering in parallel across time steps.

    Parameters:
    - data (np.ndarray): Array of shape (num_samples, time_steps, d)
    - tol_decimals (int): Rounding precision for cluster centers.
    - use_weights (bool): Whether to compute and return weights for unique paths.
    - delta_n (float, optional): Maximum allowed Euclidean deviation per time step.
                                 If None, set to 1/(num_samples^(1/time_steps)).

    Returns:
    - If use_weights is False:
         np.ndarray of quantized sample paths.
      If use_weights is True:
         tuple (np.ndarray of unique paths, list of weights).
    """
    num_samples, time_steps, d = data.shape
    if delta_n is None:
        delta_n = 1 / (num_samples ** (1 / time_steps))

    results = {}
    # Use a ProcessPoolExecutor to parallelize over time steps
    with concurrent.futures.ProcessPoolExecutor(max_workers=time_steps) as executor:
        # Submit each time step for parallel processing
        future_to_t = {
            executor.submit(process_time_step, t, data, tol_decimals, delta_n): t
            for t in range(time_steps)
        }
        for future in concurrent.futures.as_completed(future_to_t):
            t, labels, centers = future.result()
            results[t] = (labels, centers)

    # Reassemble results in the original order of time steps
    label_list = []
    support_list = []
    for t in range(time_steps):
        labels, centers = results[t]
        label_list.append(labels)
        support_list.append(centers)

    if not use_weights:
        # Reconstruct the quantized sample paths
        output = np.zeros((num_samples, time_steps, d))
        for t in range(time_steps):
            centers = support_list[t]
            labels = label_list[t]
            output[:, t, :] = centers[labels]
        return output
    else:
        # Collapse identical paths and compute weights
        unique_paths_dict = {}
        for i in range(num_samples):
            current_path = tuple(
                tuple(support_list[t][label_list[t][i]]) for t in range(time_steps)
            )
            unique_paths_dict[current_path] = unique_paths_dict.get(current_path, 0) + (
                1 / num_samples
            )
        unique_paths_list = [np.array(path) for path in unique_paths_dict.keys()]
        weights_list = list(unique_paths_dict.values())
        unique_paths_array = np.array(unique_paths_list)
        return unique_paths_array, weights_list
