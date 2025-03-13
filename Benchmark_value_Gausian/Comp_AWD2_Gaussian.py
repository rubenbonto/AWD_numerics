import numpy as np

"""
This script computes the Adapted Wasserstein squared distance \( AW_2^2 \) 
following Theorem 1.1 of the paper:

"Adapted optimal transport between Gaussian processes in discrete time" 
by Madhu Gunasingam and Ting-Kam Leonard Wong (2025).

Reference: https://arxiv.org/abs/2404.06625
"""


def build_mean_and_cov(t, mean_val=0.0, var_factor=4.0):
    """
    Constructs a mean vector and a covariance matrix in \(\mathbb{R}^t\).

    The covariance matrix follows the structure:
        \( A[i, j] = \text{var_factor} \times \min(i+1, j+1) \).

    Parameters
    ----------
    t : int
        Dimension (or time length).
    mean_val : float, optional
        Value to fill the mean vector (default is 0.0).
    var_factor : float, optional
        Scalar factor for covariance computation (default is 4.0).

    Returns
    -------
    a : np.ndarray of shape (t,)
        Mean vector.
    A : np.ndarray of shape (t, t)
        Covariance matrix.
    """
    a = np.full(t, mean_val)
    A = var_factor * np.fromfunction(
        lambda i, j: np.minimum(i + 1, j + 1), (t, t), dtype=int
    )
    return a, A


def adapted_wasserstein_squared(a, A, b, B):
    """
    Computes the Adapted Wasserstein squared distance \( AW_2^2 \)
    following Theorem 1.1 of the paper:

    "Adapted optimal transport between Gaussian processes in discrete time"
    by Madhu Gunasingam and Ting-Kam Leonard Wong (2025).

    Parameters
    ----------
    a, b : np.ndarray, shape (t,)
        Mean vectors.
    A, B : np.ndarray, shape (t, t)
        Covariance matrices.

    Returns
    -------
    float
        Adapted Wasserstein squared distance.
    """
    # Cholesky decompositions: A = L L^T, B = M M^T
    L = np.linalg.cholesky(A)
    M = np.linalg.cholesky(B)

    # Mean squared difference
    mean_diff = np.sum((a - b) ** 2)

    # Trace terms
    trace_sum = np.trace(A) + np.trace(B)

    # L1 norm of diagonal elements of L^T M
    l1_diag = np.sum(np.abs(np.diag(L.T @ M)))

    # Final adapted Wasserstein squared distance
    return mean_diff + trace_sum - 2 * l1_diag
