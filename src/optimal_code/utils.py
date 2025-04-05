from collections import defaultdict
import random

import matplotlib.pyplot as plt
import numpy as np
import ot
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


###########################
# NEW UTILITIES
###########################


# Path generation
def Lmatrix2paths(L, n_sample, normalize=False, seed=0, verbose=False):
    r"""
    Lower triangular matrix L to covariance matrix A and generated paths
    """
    A0 = L @ L.T  # A = LL^T
    L = L / np.sqrt(np.trace(A0)) if normalize else L
    A = L @ L.T

    # np.linalg.cholesky(A) -  L (sanity check)

    if verbose:
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


# from path to adapted path on the grid
def path2adaptedpath(samples, delta_n):
    r"""
    Project paths to adapted grids
    """
    grid_func = lambda x: np.floor(x / delta_n + 0.5) * delta_n
    adapted_samples = grid_func(samples)
    return adapted_samples


def sort_qpath(path):
    T = path.shape[-1] - 1
    sorting_keys = [path[:, i] for i in range(T, -1, -1)]
    return path[np.lexsort(tuple(sorting_keys))]


def qpath2mu_x(qpath, markovian=False):
    r"""
    Quantized Path to Conditional Measure
    non-Markovian:
    mu_x[0] = {(3,): {1: 1, 2: 5}}
    Markovian:
    mu_x[0] = {3: {1: 1, 2: 5}}
    """
    T = qpath.shape[-1] - 1
    mu_x = [defaultdict(dict) for t in range(T)]
    for path in qpath:
        for t in range(T):
            if markovian:
                pre_path = path[t]
            else:
                pre_path = tuple([int(x) for x in path[: t + 1]])  #
            next_val = int(path[t + 1])
            if pre_path not in mu_x[t] or next_val not in mu_x[t][pre_path]:
                mu_x[t][pre_path][next_val] = 1
            else:
                mu_x[t][pre_path][next_val] += 1
    return mu_x


def list_repr_mu_x(mu_x, q2v, quantized_value=False):
    r"""
    represent mu_x[t] with
    mu_x_c[t][i]: xq_{1:t} quantized conditional path up to time t
    mu_x_v[t][i]: a list of values x_{t+1} follows x_{1:t}
    mu_x_w[t][i]: a list of weights mu_{x_{1:t}}(x_{t+1})

    e.g. if we have 4 paths (same paths count twice)
    quantized <---> real-valued
    (1, 2, 3) <---> (10, 20, 30)
    (1, 2, 4) <---> (10, 20, 40)
    (1, 2, 4) <---> (10, 20, 40)
    (2, 3, 5) <---> (20, 30, 50)
    Then we have:
    mu_x_c[t=1] = [(1,2), (2,3)]
    mu_x_v[t=1][i=1] = [30, 40]
    mu_x_w[t=1][i=1] = [1/3, 2/3]

    if quantized_value:
     mu_x_v[t=1][i=1] = [3, 4]

    """

    # mu_x_c[t][i] = xq_{1:t}
    mu_x_c = [list(mu_x_t.keys()) for mu_x_t in mu_x]
    # mu_x_c[t] = number of x_{1:t}
    mu_x_cn = [len(mu_x_t) for mu_x_t in mu_x]
    # mu_x_v[t][i] = a list of values of x_{t+1}
    if quantized_value:
        mu_x_v = [
            [np.array([k for k in d.keys()]) for d in mu_x_t.values()]
            for mu_x_t in mu_x
        ]
    else:
        mu_x_v = [
            [np.array([q2v[k] for k in d.keys()]) for d in mu_x_t.values()]
            for mu_x_t in mu_x
        ]
    # mu_x_w[t][i] = a list of weights of x_{t+1}
    mu_x_w0 = [[list(d.values()) for d in mu_x_t.values()] for mu_x_t in mu_x]
    mu_x_w = [[np.array(l) / sum(l) for l in mu_x_w0_t] for mu_x_w0_t in mu_x_w0]

    # Size of x_{t+1} after x_t
    mu_x_n = [[len(d) for d in mu_x_t.values()] for mu_x_t in mu_x]
    # mu_x_cumn[t] = a list of starting index of (x_{1:t},x_{t+1}) for different x_{t+1}
    mu_x_cumn = [np.cumsum([0] + mu_x_n_t) for mu_x_n_t in mu_x_n]

    return mu_x_c, mu_x_cn, mu_x_v, mu_x_w, mu_x_cumn


def l_cost(vx, vy, power=2):
    return (vx[:, None] - vy[None, :]) ** power


# Fast OT solver
def solve_ot(cx, vx, wx, ix, jx, cy, vy, wy, iy, jy, Vtplus, power):
    cost = l_cost(vx, vy, power)
    if Vtplus is not None:  # t < T-1
        cost += Vtplus[ix:jx, iy:jy]
    if len(vx) == 1 or len(vy) == 1:
        res = np.dot(np.dot(wx, cost), wy)  # in this case we has closed solution
    else:
        res = ot.lp.emd2(
            wx, wy, cost
        )  # more efficient for memory and speed emd2 thant emd <- then cost mult
    return res


# Solver not paralell
def nested2(
    mu_x_cn, mu_x_v, mu_x_w, mu_x_cumn, nu_y_cn, nu_y_v, nu_y_w, nu_y_cumn, power=2
):

    T = len(mu_x_cn)
    V = [np.zeros([mu_x_cn[t], nu_y_cn[t]]) for t in range(T)]  # V_t(x_{1:t},y_{1:t})
    for t in range(T - 1, -1, -1):
        x_bar = tqdm(range(mu_x_cn[t]))
        x_bar.set_description(f"Timestep {t}")
        for cx, vx, wx, ix, jx in zip(
            x_bar, mu_x_v[t], mu_x_w[t], mu_x_cumn[t][:-1], mu_x_cumn[t][1:]
        ):
            for cy, vy, wy, iy, jy in zip(
                range(nu_y_cn[t]),
                nu_y_v[t],
                nu_y_w[t],
                nu_y_cumn[t][:-1],
                nu_y_cumn[t][1:],
            ):
                Vtplus = V[t + 1] if t < T - 1 else None
                V[t][cx, cy] = solve_ot(
                    cx, vx, wx, ix, jx, cy, vy, wy, iy, jy, Vtplus, power
                )

    AW_2square = V[0][0, 0]
    return AW_2square


# Bench mark distance (it is the same as in benchmark_value_gaussian for the 1D case!)
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


###########################
# OLD (Not Recommended for Use)
###########################


def quantization(adaptedX, adaptedY, markovian=False, verbose=True):
    T = len(adaptedX) - 1

    # Global quantization for X union Y samples on grid
    q2v = np.unique(np.concatenate([adaptedX, adaptedY], axis=0))
    v2q = {k: v for v, k in enumerate(q2v)}  # Value to Quantization

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
    if verbose:
        print("Quantization ......")
        print("Number of distinct values in global quantization: ", len(q2v))

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


# Slow solver
def nested(mu_x, nu_y, v2q_x, v2q_y, q2v, markovian=False, verbose=True):
    T = len(mu_x)
    square_cost_matrix = (q2v[None, :] - q2v[None, :].T) ** 2

    V = [np.zeros([len(v2q_x[t]), len(v2q_y[t])]) for t in range(T)]
    if verbose:
        print("Nested backward induction .......")
    for t in range(T - 1, -1, -1):
        tqdm_bar = tqdm(mu_x[t].items()) if verbose else mu_x[t].items()
        for k1, v1 in tqdm_bar:
            if verbose:
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
                if len(q1) == 1 and len(q2) == 1 and t == T - 1:
                    V[t][v2q_x[t][k1], v2q_y[t][k2]] = square_cost_matrix[
                        np.ix_(q1, q2)
                    ]
                else:
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

                    # solve the OT problem with cost |x_t-y_t|^2 + V_{t+1}(x_{1:t},y_{1:t})
                    # V[t][v2q_x[t][k1], v2q_y[t][k2]] = ot.emd2(w1, w2, cost)
                    V[t][v2q_x[t][k1], v2q_y[t][k2]] = np.sum(
                        cost * ot.lp.emd(w1, w2, cost)
                    )

    AW_2square = V[0][0, 0]
    return AW_2square, V


###########################
# Plotting Utility
###########################


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
