import concurrent.futures

import concurrent
import numpy as np
from tqdm import tqdm

from optimal_code.utils import solve_ot


def chunk_process(arg):
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
    T = len(mu_x_cn)
    V = [np.zeros([mu_x_cn[t], nu_y_cn[t]]) for t in range(T)]  # V_t(x_{1:t},y_{1:t})
    for t in range(T - 1, -1, -1):
        n_processes = n_processes if t > 0 else 1  # HERE WE NEED TO CHANGE BACK TO t>1
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
            arg = (x_arg, y_arg, Vtplus, power)
            args.append(arg)

        # for arg, chunk in zip(args, chunks):
        #     res = chunk_process(arg)
        #     V[t][chunk] = res
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_processes
        ) as executor:
            Vts = executor.map(chunk_process, args)

        for chunk, Vt in zip(chunks, Vts):
            V[t][chunk] = Vt

    AW_2square = V[0][0, 0]
    return AW_2square
