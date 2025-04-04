import numpy as np

"""
This script computes the Adapted Wasserstein squared distance \( AW_2^2 \) 
following Theorem 1.1 of the paper:

"Adapted optimal transport between Gaussian processes in discrete time" 
by Madhu Gunasingam and Ting-Kam Leonard Wong (2025).

For the generalization to Gaussian processes in \(\mathbb{R}^{dT}\), see Theorem 2.4 (with \(\lambda = 0\)) in:
"Entropic adapted Wasserstein distance on Gaussians"
by Beatrice Acciaio, Songyan Hou, Gudmund Pammer

References:
    https://arxiv.org/abs/2404.06625
    https://arxiv.org/abs/2412.18794
"""


###############################
# Functions for matrix and vector generation
###############################


def build_mean_and_cov(t, mean_val=0.0, var_factor=4.0):
    """
    Generates mean vector and covariance matrix for 1D discrete-time process.

    Parameters:
    - t (int): Time length (dimension).
    - mean_val (float): Mean value (default=0.0).
    - var_factor (float): Variance scaling factor (default=4.0).

    Returns:
    - a (np.ndarray): Mean vector of shape (t,).
    - A (np.ndarray): Covariance matrix of shape (t, t).
    """
    a = np.full(t, mean_val)
    A = var_factor * np.fromfunction(
        lambda i, j: np.minimum(i + 1, j + 1), (t, t), dtype=int
    )
    return a, A


def build_full_covariance(var, d, T):
    """
    Generates covariance matrix for \(\mathbb{R}^{dT}\) with block structure.

    Parameters:
    - var (float): Variance factor (e.g., 1.1^2 or 0.1^2).
    - d (int): Spatial dimension.
    - T (int): Number of time steps.

    Returns:
    - Cov (np.ndarray): Covariance matrix of shape (d*T, d*T).
    """
    Sigma = var * np.eye(d)
    Cov = np.zeros((d * T, d * T))
    for i in range(T):
        for j in range(T):
            factor = min(i + 1, j + 1)
            Cov[i * d : (i + 1) * d, j * d : (j + 1) * d] = factor * Sigma
    return Cov


###############################
# Solvers for AW_2^2
###############################


def adapted_wasserstein_squared_1d(A, B, a=None, b=None):
    """
    Computes AW_2^2 for a one-dimensional process.

    Parameters:
    - a, b (np.ndarray or None): Mean vectors. If None, set to zero.
    - A, B (np.ndarray): Covariance matrices.

    Returns:
    - float: AW_2^2 distance.
    """
    t = A.shape[0]
    if a is None:
        a = np.zeros(t)
    if b is None:
        b = np.zeros(t)

    L = np.linalg.cholesky(A)
    M = np.linalg.cholesky(B)

    mean_diff = np.sum((a - b) ** 2)
    trace_sum = np.trace(A) + np.trace(B)
    l1_diag = np.sum(np.abs(np.diag(L.T @ M)))

    return mean_diff + trace_sum - 2 * l1_diag


def adapted_wasserstein_squared_multidim(A, B, d, T, a=None, b=None):
    """
    Computes AW_2^2 for a multi-dimensional process in \(\mathbb{R}^{dT}\).

    Parameters:
    - a, b (np.ndarray or None): Mean vectors. If None, set to zero.
    - A, B (np.ndarray): Covariance matrices.
    - d (int): Spatial dimension.
    - T (int): Number of time steps.

    Returns:
    - float: AW_2^2 distance.
    """
    dt = A.shape[0]
    if a is None:
        a = np.zeros(dt)
    if b is None:
        b = np.zeros(dt)

    L = np.linalg.cholesky(A)
    M = np.linalg.cholesky(B)

    P = M.T @ L
    singular_sum = 0.0

    for t in range(T):
        block = P[t * d : (t + 1) * d, t * d : (t + 1) * d]
        s = np.linalg.svd(block, compute_uv=False)
        singular_sum += np.sum(s)

    mean_diff = np.sum((a - b) ** 2)
    trace_sum = np.trace(A) + np.trace(B)

    return mean_diff + trace_sum - 2 * singular_sum
