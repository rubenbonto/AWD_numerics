import numpy as np
from collections import defaultdict
import ot
from tqdm import tqdm
import concurrent.futures

###############################
# Multi-dimensional Path Generator
###############################


def Lmatrix2paths_flat(L, n_sample, d, T, normalize=False, seed=0, verbose=False):
    """
    Generate multi-dimensional process paths using a flattened matrix L.

    Parameters:
        L : np.ndarray of shape (d*T, d*T)
        n_sample : int
        d : int
        T : int
        normalize : bool, optional
        seed : int, optional
        verbose : bool, optional

    Returns:
        paths : np.ndarray of shape (n_sample, T+1, d)
        A : np.ndarray of shape (d*T, d*T)
    """
    dim = d * T
    A0 = L @ L.T
    if normalize:
        L = L / np.sqrt(np.trace(A0))
        A = L @ L.T
    else:
        A = A0

    np.random.seed(seed)
    noise = np.random.normal(size=(n_sample, dim))
    transformed = noise @ L.T
    increments = transformed.reshape(n_sample, T, d)
    paths = np.concatenate([np.zeros((n_sample, 1, d)), increments], axis=1)

    if verbose:
        print("Transformation matrix L:", L)
        print("Covariance matrix A:", A)

    return paths, A


def path2adaptedpath_multidim(samples, delta_n):
    """
    Project paths to the grid.
    """
    grid_func = lambda x: np.floor(x / delta_n + 0.5) * delta_n
    return grid_func(samples)


def sort_qpath_multidim(path):
    """
    Lexicographically sort quantized paths.
    """
    T = path.shape[-1] - 1
    sorting_keys = [path[:, i] for i in range(T, -1, -1)]
    return path[np.lexsort(tuple(sorting_keys))]


def qpath2mu_x_multidim(qpath, markovian=False):
    """
    From quantized paths to conditional measures.
    """
    T = qpath.shape[-1] - 1
    mu_x = [defaultdict(dict) for _ in range(T)]
    for path in qpath:
        for t in range(T):
            pre_path = path[t] if markovian else tuple(int(x) for x in path[: t + 1])
            next_val = int(path[t + 1])
            if next_val not in mu_x[t][pre_path]:
                mu_x[t][pre_path][next_val] = 1
            else:
                mu_x[t][pre_path][next_val] += 1
    return mu_x


def list_repr_mu_x_multidim(mu_x, q2v, quantized_value=False):
    """
    Represent the conditional measures in list format.
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
