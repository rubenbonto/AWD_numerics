import numpy as np
from sklearn.cluster import KMeans
import math

"""
This function generates new weighted sample paths that represent an empirical measure capturing time evolution.
The method is based on the paper:

Eckstein, S., & Pammer, G. (2023). "Computational Methods for Adapted Optimal Transport."
   - arXiv:2203.05005 [math.PR]. Available at: https://arxiv.org/abs/2203.05005

Inspired by:
Backhoff, J., Bartl, D., Beiglböck, M., & Wiesel, J. (2021). "Estimating Processes in Adapted Wasserstein Distance."
   - arXiv:2002.07261 [math.PR]. Available at: https://arxiv.org/abs/2002.07261

Original implementation source: https://github.com/stephaneckstein/aotnumerics

"""


# This is the original function I took from the above github
def empirical_k_means_measure(
    data, use_klist=False, klist=(), tol_decimals=6, use_weights=False, heuristic=False
):
    """
    Computes an empirical measure approximation of sample paths using k-means clustering.

    Parameters:
    - data (np.ndarray): A (num_samples, time_steps) array representing sample paths.
    - use_klist (bool): If True, uses a predefined list of cluster sizes for each time step.
    - klist (tuple): A list of cluster sizes per time step. If not provided, defaults to sqrt(num_samples).
    - tol_decimals (int): Number of decimals to round cluster centers to.
    - use_weights (bool): Whether to weight cluster centers by frequency.
    - heuristic (bool): If True, uses a heuristic clustering method instead of k-means.

    Returns:
    - np.ndarray: New weighted sample paths approximating an empirical measure.
    - list (optional): Weights associated with each sample path if use_weights=True.
    """
    num_samples, time_steps = data.shape

    if not use_klist:
        klist = (np.ones(time_steps) * int(np.round(np.sqrt(num_samples)))).astype(int)

    label_list = []
    support_list = []
    output_samples = np.zeros([0, time_steps])
    output_weights = []

    if heuristic:
        for t in range(time_steps):
            sorted_indices = np.argsort(data[:, t])
            sorted_data = data[sorted_indices, t]
            cluster_size = int(np.round(num_samples / klist[t]))
            cluster_cutoff = cluster_size * klist[t]
            clustered_means = np.mean(
                sorted_data[:cluster_cutoff].reshape(-1, cluster_size), axis=1
            )
            remainder_mean = (
                np.mean(sorted_data[cluster_cutoff:])
                if cluster_cutoff < num_samples
                else None
            )
            cluster_centers = (
                np.append(clustered_means, remainder_mean)
                if remainder_mean is not None
                else clustered_means
            )
            labels = np.digitize(sorted_data, cluster_centers, right=True)
            label_list.append(labels)
            support_list.append(cluster_centers)
    else:
        for t in range(time_steps):
            km = KMeans(n_clusters=klist[t], n_init="auto").fit(data[:, t : t + 1])
            cluster_centers = np.round(
                km.cluster_centers_, decimals=tol_decimals
            ).flatten()
            labels = km.labels_
            label_list.append(labels)
            support_list.append(cluster_centers)

    if not use_weights:
        output = np.zeros([num_samples, time_steps])
        for t in range(time_steps):
            output[:, t] = support_list[t][label_list[t]]
        return output

    for i in range(num_samples):
        current_path = np.array(
            [support_list[t][label_list[t][i]] for t in range(time_steps)]
        )
        existing_index = next(
            (
                j
                for j, path in enumerate(output_samples)
                if np.all(path == current_path)
            ),
            None,
        )
        if existing_index is not None:
            output_weights[existing_index] += 1 / num_samples
        else:
            output_samples = (
                np.vstack([output_samples, current_path])
                if output_samples.size
                else np.expand_dims(current_path, axis=0)
            )
            output_weights.append(1 / num_samples)

    return output_samples, output_weights


def empirical_k_means_measure_new(
    data, tol_decimals=6, use_weights=False, delta_n=None
):
    """
    Computes an empirical measure approximation of sample paths using k-means clustering.
    For each time step t>=1, clustering starts with one cluster and is refined until the maximum
    absolute deviation from the assigned cluster center is ≤ delta_n.

    The first time step (t == 0) is assumed to be deterministic and is not clustered.

    Parameters:
    - data (np.ndarray): A (num_samples, time_steps) array representing sample paths.
    - tol_decimals (int): Number of decimals to round cluster centers to.
    - use_weights (bool): Whether to weight cluster centers by frequency.
    - delta_n (float, optional): If None, it is initialized as
          1 / (num_samples ** (1 / time_steps)).
          Otherwise, it enforces that the maximum deviation in each time step (between a point
          and its cluster center) is below this threshold.

    Returns:
    - np.ndarray: New weighted sample paths approximating an empirical measure.
    - list (optional): Weights associated with each sample path if use_weights is True.
    """
    num_samples, time_steps = data.shape

    # Initialize delta_n if not provided
    if delta_n is None:
        delta_n = 1 / (num_samples ** (1 / time_steps))

    label_list = []
    support_list = []
    output_samples = np.zeros([0, time_steps])
    output_weights = []

    for t in range(time_steps):
        if t == 0:
            # For deterministic initial condition, assign the same value to every sample.
            unique_val = data[0, 0]
            labels = np.zeros(num_samples, dtype=int)
            cluster_centers = np.array([unique_val])
        else:
            # Start with one cluster.
            k = 1
            while True:
                km = KMeans(n_clusters=k, n_init="auto").fit(data[:, t : t + 1])
                cluster_centers = np.round(
                    km.cluster_centers_, decimals=tol_decimals
                ).flatten()
                labels = km.labels_
                # Compute absolute distances for each point to its assigned cluster center.
                distances = np.abs(data[:, t] - cluster_centers[labels])
                max_distance = distances.max()
                # Stop if maximum deviation is below delta_n or if k reaches num_samples.
                if max_distance <= delta_n or k >= num_samples:
                    break
                else:
                    k += 1
        label_list.append(labels)
        support_list.append(cluster_centers)

    if not use_weights:
        output = np.zeros([num_samples, time_steps])
        for t in range(time_steps):
            output[:, t] = support_list[t][label_list[t]]
        return output

    # Collapse identical paths and assign weights if use_weights is True.
    for i in range(num_samples):
        current_path = np.array(
            [support_list[t][label_list[t][i]] for t in range(time_steps)]
        )
        existing_index = next(
            (
                j
                for j, path in enumerate(output_samples)
                if np.all(path == current_path)
            ),
            None,
        )
        if existing_index is not None:
            output_weights[existing_index] += 1 / num_samples
        else:
            output_samples = (
                np.vstack([output_samples, current_path])
                if output_samples.size
                else np.expand_dims(current_path, axis=0)
            )
            output_weights.append(1 / num_samples)

    return output_samples, output_weights


#########################
# Bellow is somthing i keep just in case


# This is a function that for smaller spread values cluster into proportionnaly to the largest spread to which we assign sqrt(number of sample)
def empirical_k_means_measure_new_old(
    data, tol_decimals=6, use_weights=False, heuristic=False
):
    """
    Computes an empirical measure approximation of sample paths using k-means clustering.
    Uses a different number of clusters per time step based on the spread (max-min) of the data.

    The number of clusters for each time step t>=1 is set to:
        k[t] = max(2, ceil(sqrt(num_samples) * (spread[t] / max_spread)))
    where spread[t] = max(data[:, t]) - min(data[:, t]) and max_spread is the largest spread among t>=1.
    For time step 0 (deterministic initial condition) no clustering is applied.

    Parameters:
    - data (np.ndarray): A (num_samples, time_steps) array representing sample paths.
    - tol_decimals (int): Number of decimals to round cluster centers to.
    - use_weights (bool): Whether to weight cluster centers by frequency.
    - heuristic (bool): If True, uses a heuristic clustering method instead of k-means.

    Returns:
    - np.ndarray: New weighted sample paths approximating an empirical measure.
    - list (optional): Weights associated with each sample path if use_weights=True.
    """
    num_samples, time_steps = data.shape

    # Precompute the number of clusters per time step.
    # For time step 0, no clustering is needed (deterministic starting value).
    klist = np.zeros(time_steps, dtype=int)
    klist[0] = 1  # Only one unique value at the start.

    # Compute spreads for each time step.
    spreads = np.ptp(data, axis=0)  # ptp = max - min for each column
    # Exclude time 0 when determining the maximum spread.
    max_spread = spreads[1:].max() if time_steps > 1 else 0

    # For t>=1, assign clusters proportional to the spread relative to the maximum spread.
    for t in range(1, time_steps):
        # Compute the provisional number of clusters based on spread ratio.
        if max_spread > 0:
            proposed = np.sqrt(num_samples) * (spreads[t] / max_spread)
        else:
            proposed = np.sqrt(num_samples)
        # Round up and ensure at least 2 clusters.
        klist[t] = max(2, math.ceil(proposed))

    label_list = []
    support_list = []
    output_samples = np.zeros([0, time_steps])
    output_weights = []

    if heuristic:
        for t in range(time_steps):
            sorted_indices = np.argsort(data[:, t])
            sorted_data = data[sorted_indices, t]
            cluster_size = int(np.round(num_samples / klist[t]))
            cluster_cutoff = cluster_size * klist[t]
            clustered_means = np.mean(
                sorted_data[:cluster_cutoff].reshape(-1, cluster_size), axis=1
            )
            remainder_mean = (
                np.mean(sorted_data[cluster_cutoff:])
                if cluster_cutoff < num_samples
                else None
            )
            cluster_centers = (
                np.append(clustered_means, remainder_mean)
                if remainder_mean is not None
                else clustered_means
            )
            labels = np.digitize(sorted_data, cluster_centers, right=True)
            label_list.append(labels)
            support_list.append(cluster_centers)
    else:
        for t in range(time_steps):
            km = KMeans(n_clusters=klist[t], n_init="auto").fit(data[:, t : t + 1])
            cluster_centers = np.round(
                km.cluster_centers_, decimals=tol_decimals
            ).flatten()
            labels = km.labels_
            label_list.append(labels)
            support_list.append(cluster_centers)

    if not use_weights:
        output = np.zeros([num_samples, time_steps])
        for t in range(time_steps):
            output[:, t] = support_list[t][label_list[t]]
        return output

    for i in range(num_samples):
        current_path = np.array(
            [support_list[t][label_list[t][i]] for t in range(time_steps)]
        )
        existing_index = next(
            (
                j
                for j, path in enumerate(output_samples)
                if np.all(path == current_path)
            ),
            None,
        )
        if existing_index is not None:
            output_weights[existing_index] += 1 / num_samples
        else:
            output_samples = (
                np.vstack([output_samples, current_path])
                if output_samples.size
                else np.expand_dims(current_path, axis=0)
            )
            output_weights.append(1 / num_samples)

    return output_samples, output_weights


## This is a function that try to simulate a grid with r as presented in: Estimating Processes in Adapted Wasserstein Distance but scaled by the spread
def empirical_k_means_measure_grid(
    data, tol_decimals=6, use_weights=False, heuristic=False
):
    """
    Computes an empirical measure approximation of sample paths using k-means clustering.
    Uses a different number of clusters per time step based on the spread (max-min) of the data.

    The number of clusters for each time step t>=1 is set to:
        k[t] = max(2, ceil((spread[t] / max_spread) * (time_steps + 1)))
    where spread[t] = max(data[:, t]) - min(data[:, t]) and max_spread is the largest spread among t>=1.
    For time step 0 (deterministic initial condition) no clustering is applied.

    Parameters:
    - data (np.ndarray): A (num_samples, time_steps) array representing sample paths.
    - tol_decimals (int): Number of decimals to round cluster centers to.
    - use_weights (bool): Whether to weight cluster centers by frequency.
    - heuristic (bool): If True, uses a heuristic clustering method instead of k-means.

    Returns:
    - np.ndarray: New weighted sample paths approximating an empirical measure.
    - list (optional): Weights associated with each sample path if use_weights=True.
    """
    num_samples, time_steps = data.shape

    # Precompute the number of clusters per time step.
    # For time step 0, no clustering is needed (deterministic starting value).
    klist = np.zeros(time_steps, dtype=int)
    klist[0] = 1  # Only one unique value at the start.

    # Compute spreads for each time step.
    spreads = np.ptp(data, axis=0)  # ptp = max - min for each column
    # Exclude time 0 when determining the maximum spread.
    max_spread = spreads[1:].max() if time_steps > 1 else 0

    # For t>=1, assign clusters proportional to the spread relative to the maximum spread.
    # Here the maximum number of clusters is set to (time_steps+1)
    for t in range(1, time_steps):
        if max_spread > 0:
            proposed = spreads[t] * (time_steps + 1)
        else:
            proposed = time_steps + 1
        klist[t] = max(2, math.ceil(proposed))

    label_list = []
    support_list = []
    output_samples = np.zeros([0, time_steps])
    output_weights = []

    if heuristic:
        for t in range(time_steps):
            sorted_indices = np.argsort(data[:, t])
            sorted_data = data[sorted_indices, t]
            cluster_size = int(np.round(num_samples / klist[t]))
            cluster_cutoff = cluster_size * klist[t]
            clustered_means = np.mean(
                sorted_data[:cluster_cutoff].reshape(-1, cluster_size), axis=1
            )
            remainder_mean = (
                np.mean(sorted_data[cluster_cutoff:])
                if cluster_cutoff < num_samples
                else None
            )
            cluster_centers = (
                np.append(clustered_means, remainder_mean)
                if remainder_mean is not None
                else clustered_means
            )
            labels = np.digitize(sorted_data, cluster_centers, right=True)
            label_list.append(labels)
            support_list.append(cluster_centers)
    else:
        for t in range(time_steps):
            # Handle deterministic or constant steps (if only one unique value, use one cluster)
            unique_vals = np.unique(data[:, t])
            if t == 0 or unique_vals.size == 1:
                labels = np.zeros(num_samples, dtype=int)
                support = unique_vals
                label_list.append(labels)
                support_list.append(support)
                continue
            km = KMeans(n_clusters=klist[t], n_init="auto").fit(data[:, t : t + 1])
            cluster_centers = np.round(
                km.cluster_centers_, decimals=tol_decimals
            ).flatten()
            labels = km.labels_
            label_list.append(labels)
            support_list.append(cluster_centers)

    if not use_weights:
        output = np.zeros([num_samples, time_steps])
        for t in range(time_steps):
            output[:, t] = support_list[t][label_list[t]]
        return output

    for i in range(num_samples):
        current_path = np.array(
            [support_list[t][label_list[t][i]] for t in range(time_steps)]
        )
        existing_index = next(
            (
                j
                for j, path in enumerate(output_samples)
                if np.all(path == current_path)
            ),
            None,
        )
        if existing_index is not None:
            output_weights[existing_index] += 1 / num_samples
        else:
            output_samples = (
                np.vstack([output_samples, current_path])
                if output_samples.size
                else np.expand_dims(current_path, axis=0)
            )
            output_weights.append(1 / num_samples)

    return output_samples, output_weights
