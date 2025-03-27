import numpy as np
from collections import defaultdict
import ot
from tqdm import tqdm
import concurrent.futures


# --- Cost function for multi-dimensional points ---
def l_cost_multidim(vx, vy, power=2):
    r"""
    Compute the cost matrix between two sets of d-dimensional points.

    Parameters:
        vx : np.ndarray of shape (n, d)
        vy : np.ndarray of shape (m, d)
        power : float, default 2
            The cost is computed as ||vx[i] - vy[j]||^power.

    Returns:
        cost : np.ndarray of shape (n, m)
    """
    diff = vx[:, None, :] - vy[None, :, :]
    cost = np.linalg.norm(diff, axis=2) ** power
    return cost


def solve_ot_multidim(cx, vx, wx, ix, jx, cy, vy, wy, iy, jy, Vtplus, power):
    r"""
    Solve the optimal transport problem between two discrete measures
    with cost given by l_cost. Uses a closed-form solution when one of the supports
    is a singleton.
    """
    cost = l_cost_multidim(vx, vy, power)
    if Vtplus is not None:  # for t < T-1
        cost += Vtplus[ix:jx, iy:jy]
    if len(vx) == 1 or len(vy) == 1:
        res = np.dot(np.dot(wx, cost), wy)
    else:
        res = np.sum(cost * ot.lp.emd(wx, wy, cost))
    return res


def chunk_process_multidim(arg):
    x_arg, y_arg, Vtplus, power = arg
    x_arg[0] = tqdm(x_arg[0])
    Vt = np.zeros([len(x_arg[0]), len(y_arg[0])])
    for cx, vx, wx, ix, jx in zip(*x_arg):
        for cy, vy, wy, iy, jy in zip(*y_arg):
            Vt[cx, cy] = solve_ot_multidim(
                cx, vx, wx, ix, jx, cy, vy, wy, iy, jy, Vtplus, power
            )
    return Vt


def nested2_parallel_multidim(
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
    r"""
    Compute the nested (adapted) OT cost using dynamic programming.
    """
    T = len(mu_x_cn)
    V = [np.zeros([mu_x_cn[t], nu_y_cn[t]]) for t in range(T)]
    for t in range(T - 1, -1, -1):
        n_proc = n_processes if t > 0 else 1
        chunks = np.array_split(range(mu_x_cn[t]), n_proc)
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
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_proc) as executor:
            Vts = executor.map(chunk_process_multidim, args)
        for chunk, Vt_chunk in zip(chunks, Vts):
            V[t][chunk] = Vt_chunk
    AW_2square = V[0][0, 0]
    return AW_2square
