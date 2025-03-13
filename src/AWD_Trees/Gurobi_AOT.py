import numpy as np
from gurobipy import *

"""
This module implements the solution to discrete adapted optimal transport (AOT) problems
following Lemma 3.1 from:

Eckstein, S., & Pammer, G. (2023). "Computational methods for adapted optimal transport."
   - arXiv:2203.05005 [math.PR]. Available at: https://arxiv.org/abs/2203.05005

Unlike nested optimal transport methods, this implementation does NOT solve the problem via
backward induction but instead formulates and solves a large LP problem directly.
"""


def gurobi_bm(
    margs,
    f,
    p_dist=2,
    radial_cost=0,
    f_id=0,
    minmax="min",
    r_opti=0,
    outputflag=1,
    causal=0,
    anticausal=0,
):
    """
    Solves discrete adapted optimal transport (AOT) using Gurobi.

    Parameters:
    - margs (list): Two discrete probability measures represented as [N, n] arrays with weights.
    - f (function): Cost function taking two inputs x, y.
    - p_dist (int): Lp norm used for radial cost computation.
    - radial_cost (int): If 1, computes costs based on Lp norm distance.
    - f_id (int): If nonzero with radial_cost enabled, treats f as the identity function.
    - minmax (str): 'min' for minimization, any other value for maximization.
    - r_opti (int): If 1, returns the optimal transport plan.
    - outputflag (int): If 0, suppresses Gurobi output.
    - causal (int): If 1, enforces causal constraints.
    - anticausal (int): If 1, enforces anticausal constraints.

    Returns:
    - float: Optimal transport cost.
    - list (optional): Optimal transport plan if r_opti=1.
    """
    m1, m2 = margs
    xl_1, xl_2 = np.array(m1[0]), np.array(m2[0])
    pl_1, pl_2 = m1[1], m2[1]
    n1, n_dim = xl_1.shape
    n2 = len(xl_2)

    if len(xl_1.shape) == 1:
        xl_1 = xl_1.reshape(-1, 1)
    if len(xl_2.shape) == 1:
        xl_2 = xl_2.reshape(-1, 1)

    # Compute cost matrix
    if radial_cost == 0:
        cost_mat = np.array(
            [[f(xl_1[i, :], xl_2[j, :]) for j in range(n2)] for i in range(n1)]
        )
    else:
        cost_mat = np.linalg.norm(
            xl_1[:, None, :] - xl_2[None, :, :], axis=-1, ord=p_dist
        )
        if f_id == 0:
            cost_mat = f(cost_mat)

    # Initialize Gurobi model
    m = Model("Primal")
    if outputflag == 0:
        m.setParam("OutputFlag", 0)

    pi_var = m.addVars(n1, n2, lb=0, ub=1, name="pi_var")

    # Marginal constraints
    m.addConstrs((pi_var.sum(i, "*") == pl_1[i] for i in range(n1)), name="first_marg")
    m.addConstrs((pi_var.sum("*", i) == pl_2[i] for i in range(n2)), name="second_marg")

    # Causal constraints
    if causal:
        for t in range(1, n_dim):
            x_t_arr, ind_inv = np.unique(xl_1[:, :t], axis=0, return_inverse=True)
            for ind_t in range(len(x_t_arr)):
                pos_h = np.where(ind_inv == ind_t)[0]
                y_t_arr, ind_inv_y = np.unique(xl_2[:, :t], axis=0, return_inverse=True)
                for ind_t_y in range(len(y_t_arr)):
                    pos_h_y = np.where(ind_inv_y == ind_t_y)[0]
                    x_tp_arr, ind_inv_p = np.unique(
                        xl_1[pos_h, : t + 1], axis=0, return_inverse=True
                    )
                    for ind_xp in range(len(x_tp_arr)):
                        pos_xtp_real = pos_h[np.where(ind_inv_p == ind_xp)[0]]
                        pi_sum_left = quicksum(
                            pi_var[i_x, i_y] for i_x in pos_xtp_real for i_y in pos_h_y
                        )
                        pi_sum_right = quicksum(
                            pi_var[i_x, i_y] for i_x in pos_h for i_y in pos_h_y
                        )
                        mu_sum_left = sum(pl_1[i_x] for i_x in pos_h)
                        mu_sum_right = sum(pl_1[i_x] for i_x in pos_xtp_real)
                        m.addConstr(
                            pi_sum_left * mu_sum_left == pi_sum_right * mu_sum_right
                        )

    # Anticausal constraints
    if anticausal:
        for t in range(1, n_dim):
            x_t_arr, ind_inv = np.unique(xl_2[:, :t], axis=0, return_inverse=True)
            for ind_t in range(len(x_t_arr)):
                pos_h = np.where(ind_inv == ind_t)[0]
                y_t_arr, ind_inv_y = np.unique(xl_1[:, :t], axis=0, return_inverse=True)
                for ind_t_y in range(len(y_t_arr)):
                    pos_h_y = np.where(ind_inv_y == ind_t_y)[0]
                    x_tp_arr, ind_inv_p = np.unique(
                        xl_2[pos_h, : t + 1], axis=0, return_inverse=True
                    )
                    for ind_xp in range(len(x_tp_arr)):
                        pos_xtp_real = pos_h[np.where(ind_inv_p == ind_xp)[0]]
                        pi_sum_left = quicksum(
                            pi_var[i_y, i_x] for i_x in pos_xtp_real for i_y in pos_h_y
                        )
                        pi_sum_right = quicksum(
                            pi_var[i_y, i_x] for i_x in pos_h for i_y in pos_h_y
                        )
                        mu_sum_left = sum(pl_2[i_x] for i_x in pos_h)
                        mu_sum_right = sum(pl_2[i_x] for i_x in pos_xtp_real)
                        m.addConstr(
                            pi_sum_left * mu_sum_left == pi_sum_right * mu_sum_right
                        )

    # Objective function
    obj = quicksum(cost_mat[i, j] * pi_var[i, j] for i in range(n1) for j in range(n2))
    m.setObjective(obj, GRB.MINIMIZE if minmax == "min" else GRB.MAXIMIZE)

    # Solve model
    m.optimize()
    objective_val = m.ObjVal

    if r_opti == 0:
        return objective_val
    else:
        return objective_val, [[pi_var[i, j].x for j in range(n2)] for i in range(n1)]
