import numpy as np

from gurobipy import *
# FUNCTION TO DIRECTLY SOLVE CAUSAL AND BICAUSAL OT VIA LINEAR PROGRAMMING
def gurobi_bm(margs, f, p_dist=2, radial_cost=0, f_id=0, minmax='min', r_opti=0, outputflag=1, causal=0, anticausal=0):
    """
    :param margs: list with 2 entries, each entry being a discrete probability measure on R^n, where x_list is an [N, n] array
    :param f: function that takes two inputs, x, y, where the inputs are of the form as in the representation of the
    points in margs. Returns a single value
    :param p_dist: if radial cost is used, then this describes the Lp norm which is used.
    :param radial_cost: If 1, then f takes an arbitrary number of inputs but treats them element-wise. Each element
    which will be \|x-y\|_{p_dist} for some x, y. This allows for a faster computation of the cost matrix.
    :param f_id: if non-zero and raidal_cost nonzero, then f will be treated as the identity function.
    :param minmax: if 'min', then we minimize objective, else, we maximize
    :param r_opti: if 0, does not return optimizer. if 1, it does
    :return: optimal value (and optimizer) of the OT problem
    """
    # get relevant data from input:
    m1 = margs[0]
    m2 = margs[1]
    xl_1 = np.array(m1[0])
    xl_2 = np.array(m2[0])
    pl_1 = m1[1]
    pl_2 = m2[1]
    n1, n_dim = xl_1.shape
    n2 = len(xl_2)

    if len(xl_1.shape) == 1:
        xl_1 = xl_1.reshape(-1, 1)
    if len(xl_2.shape) == 1:
        xl_2 = xl_2.reshape(-1, 1)

    # build cost matrix:
    # print('Building cost matrix...')
    if radial_cost == 0:
        cost_mat = np.zeros([n1, n2])
        for i in range(n1):
            for j in range(n2):
                cost_mat[i, j] = f(xl_1[i, :], xl_2[j, :])
    else:
        cost_mat = np.linalg.norm(xl_1[:, None, :] - xl_2[None, :, :], axis=-1, ord=p_dist)
        if f_id == 0:
            cost_mat = f(cost_mat)

    # initialize model
    # print('Initializing model...')
    m = Model('Primal')
    if outputflag == 0:
        m.setParam('OutputFlag', 0)
    pi_var = m.addVars(n1, n2, lb=0, ub=1, name='pi_var')

    # add marginal constraints
    # print('Adding constraints...')
    m.addConstrs((pi_var.sum(i, '*') == pl_1[i] for i in range(n1)), name='first_marg')
    m.addConstrs((pi_var.sum('*', i) == pl_2[i] for i in range(n2)), name='second_marg')

    # add causal constraint: (Note: doesn't seem very efficient, but not sure how else to do)
    causal_count = 0
    if causal == 1:
        for t in range(1, n_dim):
            x_t_arr, ind_inv = np.unique(xl_1[:, :t], axis=0, return_inverse=True)
            for ind_t in range(len(x_t_arr)):
                pos_h = np.where(ind_inv == ind_t)[0]
                y_t_arr, ind_inv_y = np.unique(xl_2[:, :t], axis=0, return_inverse=True)
                for ind_t_y in range(len(y_t_arr)):
                    pos_h_y = np.where(ind_inv_y == ind_t_y)[0]
                    x_tp_arr, ind_inv_p = np.unique(xl_1[pos_h, :t+1], axis=0, return_inverse=True)
                    for ind_xp in range(len(x_tp_arr)):
                        pos_xtp = np.where(ind_inv_p == ind_xp)[0]
                        pos_xtp_real = pos_h[pos_xtp]
                        pi_sum_left = 0
                        for i_x in pos_xtp_real:
                            for i_y in pos_h_y:
                                pi_sum_left += pi_var[i_x, i_y]
                        pi_sum_right = 0
                        for i_x in pos_h:
                            for i_y in pos_h_y:
                                pi_sum_right += pi_var[i_x, i_y]
                        mu_sum_left = 0
                        for i_x in pos_h:
                            mu_sum_left += pl_1[i_x]
                        mu_sum_right = 0
                        for i_x in pos_xtp_real:
                            mu_sum_right += pl_1[i_x]

                        causal_count += 1
                        m.addConstr(pi_sum_left * mu_sum_left == pi_sum_right * mu_sum_right, name='causal_'+
                                                           str(t)+'_'+str(ind_t)+'_'+str(ind_t_y)+'_'+str(ind_xp))

    if anticausal == 1:
        for t in range(1, n_dim):
            x_t_arr, ind_inv = np.unique(xl_2[:, :t], axis=0, return_inverse=True)
            for ind_t in range(len(x_t_arr)):
                pos_h = np.where(ind_inv == ind_t)[0]

                y_t_arr, ind_inv_y = np.unique(xl_1[:, :t], axis=0, return_inverse=True)
                for ind_t_y in range(len(y_t_arr)):
                    pos_h_y = np.where(ind_inv_y == ind_t_y)[0]

                    x_tp_arr, ind_inv_p = np.unique(xl_2[pos_h, :t+1], axis=0, return_inverse=True)
                    # TODO: note that we have to concatenate pos_h and pos_p to get real index! (done, but good to keep in mind)

                    for ind_xp in range(len(x_tp_arr)):
                        pos_xtp = np.where(ind_inv_p == ind_xp)[0]
                        pos_xtp_real = pos_h[pos_xtp]

                        pi_sum_left = 0
                        for i_x in pos_xtp_real:
                            for i_y in pos_h_y:
                                pi_sum_left += pi_var[i_y, i_x]

                        pi_sum_right = 0
                        for i_x in pos_h:
                            for i_y in pos_h_y:
                                pi_sum_right += pi_var[i_y, i_x]

                        mu_sum_left = 0
                        for i_x in pos_h:
                            mu_sum_left += pl_2[i_x]

                        mu_sum_right = 0
                        for i_x in pos_xtp_real:
                            mu_sum_right += pl_2[i_x]

                        m.addConstr(pi_sum_left * mu_sum_left == pi_sum_right * mu_sum_right, name='anticausal_'+str(t)+'_'+str(ind_t)+'_'+str(ind_t_y)+'_'+str(ind_xp))

    # Specify objective function
    if minmax == 'min':
        obj = quicksum([cost_mat[i, j] * pi_var[i, j] for i in range(n1) for j in range(n2)])
        m.setObjective(obj, GRB.MINIMIZE)
    else:
        obj = quicksum([cost_mat[i, j] * pi_var[i, j] for i in range(n1) for j in range(n2)])
        m.setObjective(obj, GRB.MAXIMIZE)

    # solve model
    m.optimize()
    objective_val = m.ObjVal

    if r_opti == 0:
        return objective_val
    else:
        return objective_val, [[pi_var[i, j].x for j in range(n2)] for i in range(n1)]