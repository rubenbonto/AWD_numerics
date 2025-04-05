import concurrent.futures
import concurrent
import numpy as np
from tqdm import tqdm

from optimal_code.utils import solve_ot


def chunk_process(arg):
    """
    Processes a chunk of data for optimal transport computation.

    Parameters:
    - arg (tuple): Contains x_arg, y_arg, Vtplus, power.

    Returns:
    - Vt (np.ndarray): Updated cost matrix.
    """
    x_arg, y_arg, Vtplus, power = arg
    x_arg[0] = tqdm(x_arg[0])
    Vt = np.zeros([len(x_arg[0]), len(y_arg[0])])

    for cx, vx, wx, ix, jx in zip(*x_arg):
        for cy, vy, wy, iy, jy in zip(*y_arg):
            Vt[cx, cy] = solve_ot(cx, vx, wx, ix, jx, cy, vy, wy, iy, jy, Vtplus, power)

    return Vt


def nested2_parallel(
    mu_x_cn,
    mu_x_v,
    mu_x_w,
    mu_x_cumn,
    nu_y_cn,
    nu_y_v,
    nu_y_w,
    nu_y_cumn,
    n_processes=6,
    power=2,
):
    """
    Parallel computation of nested optimal transport.

    Parameters:
    - mu_x_cn, nu_y_cn (list): Number of conditions at each time step for measures X and Y.
    - mu_x_v, nu_y_v (list): Values of conditions for X and Y.
    - mu_x_w, nu_y_w (list): Weights associated with conditions for X and Y.
    - mu_x_cumn, nu_y_cumn (list): Cumulative indices for conditions.
    - n_processes (int): Number of parallel processes to use.
    - power (int): Exponent for cost function (typically 2 for squared distance).

    Returns:
    - float: Adapted Wasserstein squared distance.
    """
    T = len(mu_x_cn)
    V = [np.zeros([mu_x_cn[t], nu_y_cn[t]]) for t in range(T)]

    for t in range(T - 1, -1, -1):
        n_processes = n_processes if t > 0 else 1
        chunks = np.array_split(range(mu_x_cn[t]), n_processes)
        args = []

        for chunk in chunks:
            x_arg = [
                range(len(chunk)),
                [mu_x_v[t][i] for i in chunk],
                [mu_x_w[t][i] for i in chunk],
                [mu_x_cumn[t][:-1][i] for i in chunk],
                [mu_x_cumn[t][1:][i] for i in chunk],
            ]
            y_arg = [
                range(nu_y_cn[t]),
                nu_y_v[t],
                nu_y_w[t],
                nu_y_cumn[t][:-1],
                nu_y_cumn[t][1:],
            ]
            Vtplus = V[t + 1] if t < T - 1 else None
            args.append((x_arg, y_arg, Vtplus, power))

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_processes
        ) as executor:
            Vts = executor.map(chunk_process, args)

        for chunk, Vt in zip(chunks, Vts):
            V[t][chunk] = Vt

    AW_2square = V[0][0, 0]
    return AW_2square
