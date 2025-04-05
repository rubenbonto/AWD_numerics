import numpy as np
from collections import defaultdict
import ot
from tqdm import tqdm
import concurrent.futures

from optimal_code.utils_multidim import *


##########################
# Multi dim Markovian utilities
##########################


def qpath2mu_x_markovian_multidim(qpath):
    """
    Wrapper for markovian case.
    """
    return qpath2mu_x_multidim(qpath, markovian=True)


def list_repr_mu_x_markovian_multidim(mu_x, q2v, quantized_value=False):
    r"""
    Represent the markovian conditional measures in list format.

    Returns:
      mu_x_c   : list of conditioning states at each time step.
      mu_x_cn  : list of counts (number of conditioning states) per time step.
      mu_x_v   : list of arrays of next-state values (converted via q2v if needed).
      mu_x_w   : list of corresponding weight arrays (normalized counts).
      mu_x_cumn: an array of cumulative indices (not used in markovian solver).
      v2q_x    : list of mappings from each state to its index.
      mu_x_idx : for t<T-1, a list of next-state index arrays (lookup into v2q_x[t+1]).
    """
    # Conditioning states at each time step.
    mu_x_c = [list(mu_x_t.keys()) for mu_x_t in mu_x]
    mu_x_cn = [len(mu_x_t) for mu_x_t in mu_x]
    # Build a mapping for each time: state -> index.
    v2q_x = [{state: i for i, state in enumerate(mu_x_t.keys())} for mu_x_t in mu_x]

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

    # For t<T-1, build next-state index arrays via v2q_x at time t+1.
    T = len(mu_x)
    mu_x_idx = []
    for t in range(T):
        if t < T - 1:
            idx_list = []
            for d in mu_x[t].values():
                keys = list(d.keys())
                idx = np.array([v2q_x[t + 1][k] for k in keys])
                idx_list.append(idx)
            mu_x_idx.append(idx_list)
        else:
            mu_x_idx.append(None)

    # For compatibility; cumulative indices are not used here.
    mu_x_cumn = [np.arange(0, count + 1) for count in mu_x_cn]

    return mu_x_c, mu_x_cn, mu_x_v, mu_x_w, mu_x_cumn, v2q_x, mu_x_idx


##########################
# Multi dim Markovian Solver
##########################


def solve_ot_markov_multidim(vx, wx, x_next_idx, vy, wy, y_next_idx, Vtplus, power):
    r"""
    Solve the OT problem for one pair of conditional measures.

    vx, vy : arrays of support points (each of shape (n, d) or (m, d)).
    wx, wy : arrays of corresponding weights.
    x_next_idx, y_next_idx : arrays of indices into Vtplus (for future cost) or None.
    Vtplus : future cost matrix (if t < T-1) or None.
    power : exponent in cost (typically 2).
    """
    # Here we use a simple cost: the absolute difference raised to power.
    # For multi-dimensional data, you could use a norm (if needed).
    cost = (vx[:, None] - vy[None, :]) ** power
    # If there is a future cost, add it in using the next-state indices.
    if Vtplus is not None and x_next_idx is not None and y_next_idx is not None:
        cost += Vtplus[np.ix_(x_next_idx, y_next_idx)]
    # Use closed-form solution if one support is a singleton.
    if len(vx) == 1 or len(vy) == 1:
        res = np.dot(wx, np.dot(cost, wy))
    else:
        res = ot.lp.emd2(wx, wy, cost)
    return res


def chunk_process_markov_multidim(arg):
    """
    Process a chunk of states for the markovian nested solver.
    """
    x_arg, y_arg, Vtplus, power = arg
    indices_x, mu_x_v_list, mu_x_w_list, mu_x_next_idx_list = x_arg
    indices_y, nu_y_v_list, nu_y_w_list, nu_y_next_idx_list = y_arg
    Vt = np.zeros((len(indices_x), len(indices_y)))
    for i, (vx, wx, x_next_idx) in enumerate(
        zip(mu_x_v_list, mu_x_w_list, mu_x_next_idx_list)
    ):
        for j, (vy, wy, y_next_idx) in enumerate(
            zip(nu_y_v_list, nu_y_w_list, nu_y_next_idx_list)
        ):
            Vt[i, j] = solve_ot_markov_multidim(
                vx, wx, x_next_idx, vy, wy, y_next_idx, Vtplus, power
            )
    return Vt


def nested2_parallel_markovian_multidim(
    mu_x_cn,
    mu_x_v,
    mu_x_w,
    mu_x_idx,
    nu_y_cn,
    nu_y_v,
    nu_y_w,
    nu_y_idx,
    n_processes=6,
    power=2,
):
    r"""
    Parallel nested backward induction solver for the markovian adapted OT problem.

    Instead of using cumulative slicing, we use the next-state index arrays.
    """
    T = len(mu_x_cn)
    V = [np.zeros((mu_x_cn[t], nu_y_cn[t])) for t in range(T)]
    for t in range(T - 1, -1, -1):
        n_proc = n_processes if t > 0 else 1
        chunks = np.array_split(range(mu_x_cn[t]), n_proc)
        args = []
        for chunk in chunks:
            # Build x_arg for the chunk
            x_indices = list(chunk)
            x_v = [mu_x_v[t][i] for i in chunk]
            x_w = [mu_x_w[t][i] for i in chunk]
            if t < T - 1:
                x_next_idx = [mu_x_idx[t][i] for i in chunk]
            else:
                x_next_idx = [None for _ in chunk]
            x_arg = [x_indices, x_v, x_w, x_next_idx]

            # Build y_arg similarly
            y_indices = list(range(nu_y_cn[t]))
            y_v = [nu_y_v[t][i] for i in range(nu_y_cn[t])]
            y_w = [nu_y_w[t][i] for i in range(nu_y_cn[t])]
            if t < T - 1:
                y_next_idx = [nu_y_idx[t][i] for i in range(nu_y_cn[t])]
            else:
                y_next_idx = [None for _ in range(nu_y_cn[t])]
            y_arg = [y_indices, y_v, y_w, y_next_idx]

            Vtplus = V[t + 1] if t < T - 1 else None
            args.append((x_arg, y_arg, Vtplus, power))
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_proc) as executor:
            Vts = list(executor.map(chunk_process_markov_multidim, args))
        for chunk, Vt_chunk in zip(chunks, Vts):
            V[t][chunk] = Vt_chunk
    AW_2square = V[0][0, 0]
    return AW_2square
