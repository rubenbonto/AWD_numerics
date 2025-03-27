import numpy as np
from collections import defaultdict
import ot
from tqdm import tqdm
import concurrent.futures

# --- Multi-dimensional path generator ---
import numpy as np


def Lmatrix2paths_flat(L, n_sample, d, T, normalize=False, seed=0, verbose=False):
    r"""
    Generate multi-dimensional process paths using a flattened matrix L.

    Parameters:
        L : np.ndarray of shape (d*T, d*T)
            The transformation matrix.
        n_sample : int
            Number of sample paths.
        d : int
            Dimension of the process.
        T : int
            Number of time steps (increments). The paths will have T+1 time points.
        normalize : bool, optional
            If True, normalize L so that the total variance equals 1.
        seed : int, optional
            Random seed for reproducibility.
        verbose : bool, optional
            If True, print debugging information.

    Returns:
        paths : np.ndarray of shape (n_sample, T+1, d)
            The generated sample paths with X₀ = 0.
        A : np.ndarray of shape (d*T, d*T)
            The covariance matrix A = L Lᵀ.
    """
    dim = d * T  # total dimension
    # Compute covariance and optionally normalize
    A0 = L @ L.T
    if normalize:
        L = L / np.sqrt(np.trace(A0))
        A = L @ L.T
    else:
        A = A0

    np.random.seed(seed)
    # Generate all noise samples at once, shape (n_sample, dim)
    noise = np.random.normal(size=(n_sample, dim))
    # Apply the transformation: result is shape (n_sample, dim)
    transformed = noise @ L.T
    # Reshape to (n_sample, T, d) so that each path has T increments in R^d
    increments = transformed.reshape(n_sample, T, d)
    # Prepend the zero initial condition: shape becomes (n_sample, T+1, d)
    paths = np.concatenate([np.zeros((n_sample, 1, d)), increments], axis=1)

    if verbose:
        print("Transformation matrix L (possibly normalized):")
        print(L)
        print("Covariance matrix A = L Lᵀ:")
        print(A)

    return paths, A


def path2adaptedpath_multidim(samples, delta_n):
    r"""
    Project paths to the grid.

    Works elementwise so that it applies also to multi-dimensional arrays.
    """
    grid_func = lambda x: np.floor(x / delta_n + 0.5) * delta_n
    adapted_samples = grid_func(samples)
    return adapted_samples


def sort_qpath_multidim(path):
    r"""
    Lexicographically sort quantized paths.

    Assumes 'path' is an array of shape (T+1, n_sample) where each entry is an integer.
    """
    T = path.shape[-1] - 1
    sorting_keys = [path[:, i] for i in range(T, -1, -1)]
    return path[np.lexsort(tuple(sorting_keys))]


def qpath2mu_x_multidim(qpath, markovian=False):
    r"""
    From quantized paths to conditional measures.
    For each time step, build a dictionary mapping the history (or last value in the Markovian case)
    to the distribution of the next quantized value.
    """
    T = qpath.shape[-1] - 1
    mu_x = [defaultdict(dict) for t in range(T)]
    for path in qpath:
        for t in range(T):
            if markovian:
                pre_path = path[t]
            else:
                pre_path = tuple(int(x) for x in path[: t + 1])
            next_val = int(path[t + 1])
            if pre_path not in mu_x[t] or next_val not in mu_x[t][pre_path]:
                mu_x[t][pre_path][next_val] = 1
            else:
                mu_x[t][pre_path][next_val] += 1
    return mu_x


def list_repr_mu_x_multidim(mu_x, q2v, quantized_value=False):
    r"""
    Represent the conditional measures in list format.
    Returns:
        mu_x_c : list of histories at each time step.
        mu_x_cn : counts (number of histories) per time step.
        mu_x_v : list of next-value arrays for each history.
        mu_x_w : list of corresponding weight arrays (normalized counts).
        mu_x_cumn : cumulative indices (for use in dynamic programming).
    """
    mu_x_c = [list(mu_x_t.keys()) for mu_x_t in mu_x]
    mu_x_cn = [len(mu_x_t) for mu_x_t in mu_x]
    if quantized_value:
        mu_x_v = [
            [np.array(list(d.keys())) for d in mu_x_t.values()] for mu_x_t in mu_x
        ]
    else:
        mu_x_v = [
            [np.array([q2v[k] for k in d.keys()]) for d in mu_x_t.values()]
            for mu_x_t in mu_x
        ]
    mu_x_w0 = [[list(d.values()) for d in mu_x_t.values()] for mu_x_t in mu_x]
    mu_x_w = [[np.array(l) / sum(l) for l in mu_x_w0_t] for mu_x_w0_t in mu_x_w0]
    mu_x_n = [[len(d) for d in mu_x_t.values()] for mu_x_t in mu_x]
    mu_x_cumn = [np.cumsum([0] + mu_x_n_t) for mu_x_n_t in mu_x_n]
    return mu_x_c, mu_x_cn, mu_x_v, mu_x_w, mu_x_cumn
