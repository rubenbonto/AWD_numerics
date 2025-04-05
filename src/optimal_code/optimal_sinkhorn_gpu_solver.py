#######################
# GPU Solver
##########################


import torch
import time
from torch.special import logsumexp as lse


# --- Initialisation GPU ---
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
_ = torch.randn(10, 10, device="cuda") @ torch.randn(10, 10, device="cuda")

#######################
# Sinkhorn Iteration
##########################


@torch.jit.script
def sinkhorn_iteration_batched_jit(
    distance_matrix,
    p1,
    p2,
    stopping_criterion: float = 1e-6,
    lambda_reg: float = 5.0,
    max_iterations: int = 1000,
):
    # distance_matrix: (B, n, n)
    # p1, p2: (B, n)
    logK = -lambda_reg * distance_matrix
    B, n1, n2 = logK.size()
    u = torch.zeros((B, n1), dtype=distance_matrix.dtype, device=distance_matrix.device)
    v = torch.zeros((B, n2), dtype=distance_matrix.dtype, device=distance_matrix.device)
    log_p1 = torch.log(p1)
    log_p2 = torch.log(p2)
    for it in range(max_iterations):
        u_prev = u
        u = log_p1 - lse(logK + v.unsqueeze(1), dim=2)
        v = log_p2 - lse(logK.transpose(1, 2) + u.unsqueeze(1), dim=2)
        if torch.sum(torch.abs(u - u_prev)) < stopping_criterion * B:
            break
    return torch.sum(
        torch.exp(u.unsqueeze(2) + v.unsqueeze(1) + logK) * distance_matrix, dim=(1, 2)
    )


try:
    sinkhorn_iteration_batched_jit = torch.compile(
        sinkhorn_iteration_batched_jit, mode="max-autotune"
    )
except Exception:
    pass

#######################
# Solve Time Step (Vectorized GPU)
##########################


def solve_time_step_vectorized_gpu(
    x_v,
    x_w,
    x_len,
    y_v,
    y_w,
    y_len,
    V_extra=None,
    lambda_reg=5.0,
    stopping_criterion=1e-6,
    max_iterations=1000,
):
    """
    Solveur GPU vectorisé pour un pas de temps.

    V_extra (optionnel) est un tensor de forme (n_x, n_y, max_x_len, max_y_len)
    qui ajoute un coût supplémentaire provenant de V[t+1].
    """
    device = x_v.device
    n_x, Lx = x_v.shape
    n_y, Ly = y_v.shape
    R = torch.zeros((n_x, n_y), device=device)

    trivial_mask = ((x_len == 1).unsqueeze(1)) | ((y_len == 1).unsqueeze(0))
    nontrivial_mask = ~trivial_mask

    if trivial_mask.sum() > 0:
        triv_i, triv_j = trivial_mask.nonzero(as_tuple=True)
        cost_trivial = (x_v[triv_i, 0] - y_v[triv_j, 0]) ** 2
        if V_extra is not None:
            cost_trivial = cost_trivial + V_extra[triv_i, triv_j, 0, 0]
        R[triv_i, triv_j] = cost_trivial

    nontrivial_idx = nontrivial_mask.nonzero(as_tuple=False)
    if nontrivial_idx.size(0) > 0:
        B = nontrivial_idx.size(0)
        X_batch = x_v[nontrivial_idx[:, 0]]  # (B, Lx)
        WX_batch = x_w[nontrivial_idx[:, 0]]
        Y_batch = y_v[nontrivial_idx[:, 1]]  # (B, Ly)
        WY_batch = y_w[nontrivial_idx[:, 1]]
        X_len = x_len[nontrivial_idx[:, 0]].unsqueeze(1)  # (B, 1)
        Y_len = y_len[nontrivial_idx[:, 1]].unsqueeze(1)  # (B, 1)

        idx_x = torch.arange(Lx, device=device).unsqueeze(0).expand(B, Lx)
        idx_y = torch.arange(Ly, device=device).unsqueeze(0).expand(B, Ly)
        mask_x = idx_x < X_len  # (B, Lx)
        mask_y = idx_y < Y_len  # (B, Ly)

        X_sq = X_batch**2
        Y_sq = Y_batch**2
        cost_matrix = (
            X_sq.unsqueeze(2)
            + Y_sq.unsqueeze(1)
            - 2 * (X_batch.unsqueeze(2) * Y_batch.unsqueeze(1))
        )
        valid_mask = mask_x.unsqueeze(2) & mask_y.unsqueeze(1)
        cost_matrix = torch.where(
            valid_mask, cost_matrix, torch.full_like(cost_matrix, 1e6)
        )

        if V_extra is not None:
            extra_tensor = V_extra[nontrivial_idx[:, 0], nontrivial_idx[:, 1], :Lx, :Ly]
            extra_batch = (
                extra_tensor * (mask_x.unsqueeze(2) & mask_y.unsqueeze(1)).float()
            )
            cost_matrix = cost_matrix + extra_batch

        WX_valid = WX_batch * mask_x.float()
        WY_valid = WY_batch * mask_y.float()
        p1 = WX_valid / (WX_valid.sum(dim=1, keepdim=True) + 1e-10)
        p2 = WY_valid / (WY_valid.sum(dim=1, keepdim=True) + 1e-10)

        cost_values = sinkhorn_iteration_batched_jit(
            cost_matrix,
            p1,
            p2,
            stopping_criterion=stopping_criterion,
            lambda_reg=lambda_reg,
            max_iterations=max_iterations,
        )
        R[nontrivial_idx[:, 0], nontrivial_idx[:, 1]] = cost_values
    return R


def nested2_parallel_gpu_fully_batched_all_vectorized(
    recursion_data, lambda_reg=5.0, stopping_criterion=1e-6, max_iterations=1000
):
    """
    Itère sur tous les pas de temps en mode vectorisé sur GPU.
    Chaque pas de temps exécute en parallèle toutes les opérations indépendantes (calculs Sinkhorn, etc.)
    sur le GPU.
    """
    T = len(recursion_data)
    V = [None] * T

    last_x = recursion_data[T - 1][0]
    last_y = recursion_data[T - 1][3]
    V[T - 1] = torch.zeros(last_x.size(0), last_y.size(0), device=last_x.device)

    for t in range(T - 1, -1, -1):
        print(f"Processing time step {t} ...")
        x_v, x_w, x_len, y_v, y_w, y_len = recursion_data[t]

        if t < T - 1:
            n_x, n_y = x_v.size(0), y_v.size(0)
            # Conversion unique des indices de découpage pour éviter des appels scalaires dans la boucle.
            x_starts = torch.cat(
                [torch.tensor([0], device=x_len.device), x_len.cumsum(0)[:-1]]
            ).tolist()
            x_ends = x_len.cumsum(0).tolist()
            y_starts = torch.cat(
                [torch.tensor([0], device=y_len.device), y_len.cumsum(0)[:-1]]
            ).tolist()
            y_ends = y_len.cumsum(0).tolist()
            max_nx, max_ny = int(x_len.max().item()), int(y_len.max().item())
            V_extra = torch.zeros((n_x, n_y, max_nx, max_ny), device=x_v.device)

            V_next_full = V[t + 1]
            for i in range(n_x):
                xi0, xi1 = x_starts[i], x_ends[i]
                for j in range(n_y):
                    yj0, yj1 = y_starts[j], y_ends[j]
                    V_extra[i, j, : xi1 - xi0, : yj1 - yj0] = V_next_full[
                        xi0:xi1, yj0:yj1
                    ]
        else:
            V_extra = None

        R = solve_time_step_vectorized_gpu(
            x_v,
            x_w,
            x_len,
            y_v,
            y_w,
            y_len,
            V_extra=V_extra,
            lambda_reg=lambda_reg,
            stopping_criterion=stopping_criterion,
            max_iterations=max_iterations,
        )
        V[t] = R

    return V[0][0, 0]


# ================================
# Additional Global Sinkhorn for batched global OT solver (A lot Slower)
# ================================


def to_gpu_tensor(x, device=torch.device("cuda")):
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=torch.float32, device=device)


def process_time_step_all_ot_global(
    x_v_list,
    x_w_list,
    x_start_list,
    x_end_list,
    y_v_list,
    y_w_list,
    y_start_list,
    y_end_list,
    Vtplus,
    lambda_reg=5.0,
    stopping_criterion=1e-6,
    max_iterations=1000,
):
    """
    For a given time step, process all OT problems in one full batched call.
    """
    # print("V_extra")
    # print(Vtplus)
    n_x = len(x_v_list)
    n_y = len(y_v_list)
    R = np.zeros((n_x, n_y), dtype=np.float32)
    device = torch.device("cuda")

    trivial_time = 0.0
    nontrivial_list = []

    t_start = time.perf_counter()
    for i in range(n_x):
        n1 = x_end_list[i] - x_start_list[i]
        for j in range(n_y):
            n2 = y_end_list[j] - y_start_list[j]
            t0 = time.perf_counter()
            if n1 == 1 or n2 == 1:
                vx = to_gpu_tensor(x_v_list[i], device)
                vy = to_gpu_tensor(y_v_list[j], device)
                wx = to_gpu_tensor(x_w_list[i], device)
                wy = to_gpu_tensor(y_w_list[j], device)
                cost = (vx.unsqueeze(1) - vy.unsqueeze(0)) ** 2
                if Vtplus is not None:
                    V_sub = to_gpu_tensor(
                        Vtplus[
                            x_start_list[i] : x_end_list[i],
                            y_start_list[j] : y_end_list[j],
                        ],
                        device,
                    )
                    cost = cost + V_sub
                res = torch.dot(wx, torch.mv(cost, wy))
                R[i, j] = res.item()
                trivial_time += time.perf_counter() - t0
            else:
                if Vtplus is not None:
                    V_sub = Vtplus[
                        x_start_list[i] : x_end_list[i], y_start_list[j] : y_end_list[j]
                    ]
                else:
                    V_sub = None
                nontrivial_list.append(
                    (
                        i,
                        j,
                        x_v_list[i],
                        x_w_list[i],
                        y_v_list[j],
                        y_w_list[j],
                        V_sub,
                        n1,
                        n2,
                    )
                )

    t_batch_start = time.perf_counter()
    if len(nontrivial_list) > 0:
        max_n1 = max(item[7] for item in nontrivial_list)
        max_n2 = max(item[8] for item in nontrivial_list)
        B = len(nontrivial_list)
        x_batch = torch.zeros((B, max_n1), dtype=torch.float32, device=device)
        wx_batch = torch.zeros((B, max_n1), dtype=torch.float32, device=device)
        y_batch = torch.zeros((B, max_n2), dtype=torch.float32, device=device)
        wy_batch = torch.zeros((B, max_n2), dtype=torch.float32, device=device)
        if Vtplus is not None:
            V_batch = torch.zeros(
                (B, max_n1, max_n2), dtype=torch.float32, device=device
            )
        else:
            V_batch = None
        for idx, (i, j, x_vec, wx_vec, y_vec, wy_vec, V_sub, n1, n2) in enumerate(
            nontrivial_list
        ):
            x_batch[idx, :n1] = torch.tensor(x_vec, dtype=torch.float32, device=device)
            wx_batch[idx, :n1] = torch.tensor(
                wx_vec, dtype=torch.float32, device=device
            )
            y_batch[idx, :n2] = torch.tensor(y_vec, dtype=torch.float32, device=device)
            wy_batch[idx, :n2] = torch.tensor(
                wy_vec, dtype=torch.float32, device=device
            )
            if V_batch is not None and V_sub is not None:
                V_batch[idx, :n1, :n2] = torch.tensor(
                    V_sub, dtype=torch.float32, device=device
                )
        t_batch_padding = time.perf_counter() - t_batch_start

        p1 = wx_batch / (wx_batch.sum(dim=1, keepdim=True) + 1e-10)
        p2 = wy_batch / (wy_batch.sum(dim=1, keepdim=True) + 1e-10)
        cost_matrix = (x_batch.unsqueeze(2) - y_batch.unsqueeze(1)) ** 2
        # print("first step cost")
        # print(cost_matrix)
        # print("nontrivial_list:",nontrivial_list)
        # print("V_batch:", V_batch.shape)
        if V_batch is not None:
            cost_matrix = cost_matrix + V_batch
        t_sinkhorn_start = time.perf_counter()
        cost_values = sinkhorn_iteration_batched_jit(
            cost_matrix,
            p1,
            p2,
            stopping_criterion=stopping_criterion,
            lambda_reg=lambda_reg,
            max_iterations=max_iterations,
        )
        t_sinkhorn = time.perf_counter() - t_sinkhorn_start
        cost_values = cost_values.cpu().numpy()
        for idx, (i, j, *_) in enumerate(nontrivial_list):
            R[i, j] = cost_values[idx]
    else:
        t_batch_padding = 0.0
        t_sinkhorn = 0.0

    total_time = time.perf_counter() - t_start
    return R


def nested2_parallel_gpu_fully_batched_all(
    mu_x_cn,
    mu_x_v,
    mu_x_w,
    mu_x_cumn,
    nu_y_cn,
    nu_y_v,
    nu_y_w,
    nu_y_cumn,
    lambda_reg=5.0,
    stopping_criterion=1e-6,
    max_iterations=1000,
):
    """
    Process the adapted OT problem recursively across time steps using the fully batched GPU solver.
    """
    T = len(mu_x_cn)
    V = [np.zeros((mu_x_cn[t], nu_y_cn[t]), dtype=np.float32) for t in range(T)]
    total_recursion_start = time.perf_counter()
    for t in range(T - 1, -1, -1):
        print(f"Processing time step {t} ...")
        t_step_start = time.perf_counter()
        R = process_time_step_all_ot_global(
            mu_x_v[t],
            mu_x_w[t],
            mu_x_cumn[t][:-1],
            mu_x_cumn[t][1:],
            nu_y_v[t],
            nu_y_w[t],
            nu_y_cumn[t][:-1],
            nu_y_cumn[t][1:],
            V[t + 1] if t < T - 1 else None,
            lambda_reg=lambda_reg,
            stopping_criterion=stopping_criterion,
            max_iterations=max_iterations,
        )

        # print(f"Time step {t} result (first 5 entries):", R[:5])  #### to remove
        V[t] = R
        print(f"Time step {t} total time: {time.perf_counter() - t_step_start:.4f}s")
    AW_2square = V[0][0, 0]
    print(f"Total recursion time: {time.perf_counter() - total_recursion_start:.4f}s")
    return AW_2square
