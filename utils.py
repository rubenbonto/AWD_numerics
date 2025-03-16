from collections import defaultdict
import random

import matplotlib.pyplot as plt
import numpy as np
import ot
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


def Lmatrix2paths(L, n_sample, normalize=False, seed=0):
    r"""
    Lower triangular matrix L to covariance matrix A and generated paths
    """
    A0 = L @ L.T  # A = LL^T
    L = L / np.sqrt(np.trace(A0)) if normalize else L
    A = L @ L.T

    # np.linalg.cholesky(A) -  L (sanity check)

    print("Cholesky:")
    print(L)
    print("Covariance:")
    print(A)

    T = len(L)

    np.random.seed(seed)
    noise1 = np.random.normal(size=[T, n_sample])  # (T, n_sample)
    X = L @ noise1  # (T, n_sample)
    X = np.concatenate([np.zeros_like(X[:1]), X], axis=0)  # (T+1, n_sample)
    return X, A


def adapted_empirical_measure(samples, delta_n):
    r"""
    Project paths to adapted grids
    """
    grid_func = lambda x: np.floor(x / delta_n + 0.5) * delta_n
    adapted_samples = grid_func(samples)
    return adapted_samples


def adapted_wasserstein_squared(A, B, a=0, b=0):
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


def quantization(adaptedX, adaptedY, markovian=False):
    print("Quantization ......")
    T = len(adaptedX) - 1

    # Global quantization for X union Y samples on grid
    q2v = np.unique(np.concatenate([adaptedX, adaptedY], axis=0))
    v2q = {k: v for v, k in enumerate(q2v)}  # Value to Quantization
    print("Number of distinct values in global quantization: ", len(q2v))

    def adapted_path2conditional_measure(adaptedpath, v2q, markovian):
        r"""
        Path to Conditional Measure
        non-Markovian:
        mu_x[0] = {(3,): {1: 1, 2: 5}}
        Markovian:
        mu_x[0] = {3: {1: 1, 2: 5}}
        """
        T = len(adaptedpath) - 1
        mu_x = [defaultdict(dict) for t in range(T)]
        for path in adaptedpath.T:
            for t in range(T):
                if markovian:
                    pre_path = v2q[path[t]]
                else:
                    pre_path = tuple(v2q[v] for v in path[: t + 1])  #
                next_val = v2q[path[t + 1]]
                if pre_path not in mu_x[t] or next_val not in mu_x[t][pre_path]:
                    mu_x[t][pre_path][next_val] = 1
                else:
                    mu_x[t][pre_path][next_val] += 1
        return mu_x

    mu_x = adapted_path2conditional_measure(adaptedX, v2q, markovian=markovian)
    nu_y = adapted_path2conditional_measure(adaptedY, v2q, markovian=markovian)

    print("Number of condition subpaths of mu_x")
    for t in range(T):
        print(f"Time {t}: {len(mu_x[t])}")

    print("Number of condition subpaths of nu_y")
    for t in range(T):
        print(f"Time {t}: {len(nu_y[t])}")

    # Conditional Measure to Time Quantization
    # Quantization of history sub-paths
    # non-Markovian: quantization of tuple value e.g. q2v_x[2][33] ---> (2,3,4);
    # Markovian: quantization of integer value e.g. q2v_x[2][33] ---> (4);

    q2v_x = [list(mu_x[t].keys()) for t in range(T)]
    v2q_x = [{k: v for v, k in enumerate(q2v_x[t])} for t in range(T)]

    q2v_y = [list(nu_y[t].keys()) for t in range(T)]
    v2q_y = [{k: v for v, k in enumerate(q2v_y[t])} for t in range(T)]

    return q2v, v2q, mu_x, nu_y, q2v_x, v2q_x, q2v_y, v2q_y


def nested(mu_x, nu_y, v2q_x, v2q_y, q2v, markovian=False):
    T = len(mu_x)
    square_cost_matrix = (q2v[None, :] - q2v[None, :].T) ** 2

    V = [np.zeros([len(v2q_x[t]), len(v2q_y[t])]) for t in range(T)]
    print("Nested backward induction .......")
    for t in range(T - 1, -1, -1):
        tqdm_bar = tqdm(mu_x[t].items())
        for k1, v1 in tqdm_bar:
            tqdm_bar.set_description(f"Timestep {t}")
            for k2, v2 in nu_y[t].items():
                # list of probability of conditional distribution mu_x
                w1 = list(v1.values())
                w1 = np.array(w1) / sum(w1)
                # list of probability of conditional distribution nu_y
                w2 = list(v2.values())
                w2 = np.array(w2) / sum(w2)
                # list of quantized values of conditional distribution mu_x (nu_y)
                q1 = list(v1.keys())
                q2 = list(v2.keys())
                # square cost of the values indexed by quantized values: |q2v[q1] - q2v[q2]|^2
                cost = square_cost_matrix[np.ix_(q1, q2)]

                # At T-1: add V[T] = 0, otherwise add the V[t+1] already computed
                if t < T - 1:
                    if (
                        markovian
                    ):  # If markovian, for condition path (k1,q), only the last value q matters, and V[t+1] is indexed by the time re-quantization of q
                        q1s = [v2q_x[t + 1][q] for q in v1.keys()]
                        q2s = [v2q_y[t + 1][q] for q in v2.keys()]
                    else:  # If non-markovian, for condition path (k1,q), the V[t+1] is indexed by the time re-quantization of tuple (k1,q)
                        q1s = [v2q_x[t + 1][k1 + (q,)] for q in v1.keys()]
                        q2s = [v2q_y[t + 1][k2 + (q,)] for q in v2.keys()]
                    cost += V[t + 1][np.ix_(q1s, q2s)]
                try:
                    V[t][v2q_x[t][k1], v2q_y[t][k2]] = ot.emd2(
                        w1, w2, cost
                    )  # solve the OT problem with cost |x_t-y_t|^2 + V_{t+1}(x_{1:t},y_{1:t})
                except:
                    print(k1, k2)
                    print(v2q_x[t][k1], v2q_y[t][k2])
                    print(V[t].shape)
                    V[1.2]

    AW_2square = V[0][0, 0]
    return AW_2square, V


def plot_V(q2v, q2v_x, q2v_y, V, t, markovian=True, L=None, M=None):
    if markovian:
        x = np.array(q2v[q2v_x[t]])
        y = np.array(q2v[q2v_y[t]])
    else:
        x = np.array(q2v[[x[-1] for x in q2v_x[t]]])
        y = np.array(q2v[[x[-1] for x in q2v_y[t]]])
    z = V[t]

    ix = random.sample(range(len(x)), min(100, len(x)))
    iy = random.sample(range(len(y)), min(100, len(y)))
    x = x[ix]
    y = y[iy]
    z = z[np.ix_(ix, iy)]

    X, Y = np.meshgrid(x, y)  # Create meshgrid
    Z = z.T

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(X, Y, Z, c=Z, cmap="viridis", marker="o", label="Numerical V")

    if L is not None and M is not None:
        Z_2 = (X - Y) ** 2 + (
            L[t, t - 1] / L[t - 1, t - 1] * X - M[t, t - 1] / M[t, t] * Y
        ) ** 2  # Example Z matrix
        ax1.scatter(X, Y, Z_2, c=Z_2, cmap="autumn", alpha=0.3, label="Theoretical V")

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.scatter(X, Y, Z - Z_2, c=Z - Z_2, cmap="autumn", alpha=0.3)

        ax2.set_xlabel("X Axis")
        ax2.set_ylabel("Y Axis")
        ax2.set_zlabel("Z Axis")
        ax2.set_title("Error")

    # Labels
    ax1.legend()
    ax1.set_xlabel("X Axis")
    ax1.set_ylabel("Y Axis")
    ax1.set_zlabel("Z Axis")
    ax1.set_title("V(x_t,y_t)")

    return fig, ax1
