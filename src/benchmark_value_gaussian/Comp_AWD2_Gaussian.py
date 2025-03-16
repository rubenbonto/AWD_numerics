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
# For d = 1
###############################


def build_mean_and_cov(t, mean_val=0.0, var_factor=4.0):
    """
    Constructs the mean vector and covariance matrix for a discrete-time process
    where
        x_0 = mean_val,
        and x_{t+1} = x_t + \(\gamma_t\) with \(\gamma_t \sim \mathcal{N}(0, \text{var_factor})\).

    The covariance matrix \(A\) is given by:
        A[i, j] = var_factor * min(i+1, j+1)

    Parameters
    ----------
    t : int
        Time length (dimension).
    mean_val : float, optional
        Value for each entry in the mean vector (default 0.0).
    var_factor : float, optional
        Scalar factor for covariance (default 4.0).

    Returns
    -------
    a : np.ndarray, shape (t,)
        Mean vector.
    A : np.ndarray, shape (t, t)
        Covariance matrix.
    """
    a = np.full(t, mean_val)
    A = var_factor * np.fromfunction(
        lambda i, j: np.minimum(i + 1, j + 1), (t, t), dtype=int
    )
    return a, A


def adapted_wasserstein_squared_1d(a, A, b, B):
    """
    Computes the Adapted Wasserstein squared distance \( AW_2^2 \) for a one-dimensional process.

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
    # Cholesky decompositions
    L = np.linalg.cholesky(A)
    M = np.linalg.cholesky(B)

    # Mean difference and trace terms
    mean_diff = np.sum((a - b) ** 2)
    trace_sum = np.trace(A) + np.trace(B)

    # L1 norm of diagonal elements of L^T M
    l1_diag = np.sum(np.abs(np.diag(L.T @ M)))

    return mean_diff + trace_sum - 2 * l1_diag


###############################
# For d > 1
###############################


def adapted_wasserstein_squared(a, A, b, B, d, T):
    """
    Computes the Adapted 2-Wasserstein squared distance between two Gaussian measures
    \(\mu = \mathcal{N}(a,A)\) and \(\nu = \mathcal{N}(b,B)\) in \(\mathbb{R}^{dT}\).

    Parameters
    ----------
    a : np.ndarray, shape (d*T,)
        Mean vector of the first Gaussian.
    A : np.ndarray, shape (d*T, d*T)
        Covariance matrix of the first Gaussian.
    b : np.ndarray, shape (d*T,)
        Mean vector of the second Gaussian.
    B : np.ndarray, shape (d*T, d*T)
        Covariance matrix of the second Gaussian.
    d : int
        Intrinsic spatial dimension.
    T : int
        Number of time steps (so that the total dimension is d*T).

    Returns
    -------
    float
        Adapted Wasserstein squared distance.
    """
    # Cholesky decompositions
    L = np.linalg.cholesky(A)
    M = np.linalg.cholesky(B)

    # Compute product and partition into T diagonal blocks
    P = M.T @ L
    singular_sum = 0.0
    for t in range(T):
        block = P[t * d : (t + 1) * d, t * d : (t + 1) * d]
        s = np.linalg.svd(block, compute_uv=False)
        singular_sum += np.sum(s)

    mean_diff = np.sum((a - b) ** 2)
    trace_sum = np.trace(A) + np.trace(B)
    correction_term = 2 * singular_sum

    return mean_diff + trace_sum - correction_term


def build_full_covariance(var, d, T):
    """
    Builds the full covariance matrix for a process in \(\mathbb{R}^{dT}\) with block structure.

    The block (i, j) of the covariance is given by:

        Block(i, j) = min(i+1, j+1) * (var * I_d)

    This generalizes the one-dimensional structure to d-dimensional increments.

    Parameters
    ----------
    var : float
        Variance parameter (e.g., 1.1^2 or 0.1^2).
    d : int
        Spatial dimension.
    T : int
        Number of time steps.

    Returns
    -------
    Cov : np.ndarray, shape (d*T, d*T)
        Covariance matrix with the specified block structure.
    """
    Sigma = var * np.eye(d)
    Cov = np.zeros((d * T, d * T))
    for i in range(T):
        for j in range(T):
            factor = min(i + 1, j + 1)
            Cov[i * d : (i + 1) * d, j * d : (j + 1) * d] = factor * Sigma
    return Cov
