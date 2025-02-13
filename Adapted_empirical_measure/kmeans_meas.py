import numpy as np
from sklearn.cluster import KMeans
from itertools import product
from collections import defaultdict

def empirical_k_means_measure(data, use_klist=0, klist=(), tol_decimals=6, use_weights=0, heuristic=0):
    # data is [k, T_h] array
    # klist is list with T_h entries, each being an integer lower than k; number of barycenters for each time step
    (k, T_h) = data.shape
    if not use_klist:
        klist = (np.ones(T_h) * int(np.round(np.sqrt(k)))).astype(int)

    label_list = []
    support_list = []
    out_x = np.zeros([0, T_h])
    out_w = []

    # cluster points at each time point
    # print('Clustering...')
    if heuristic:
        for t in range(T_h):
            data_t = data[:, t]
            inds_sort_t = np.argsort(data_t)
            datas_t = data_t[inds_sort_t]
            n_av = int(np.round(k/klist[t]))
            lmax = int(np.floor(n_av * klist[t]))
            all_but_end = np.reshape(datas_t[:lmax], (-1, n_av))
            mean_all_but = np.mean(all_but_end, axis=1, keepdims=1)
            cx = mean_all_but
            mean_all_but = np.tile(mean_all_but, (1, n_av))
            mean_all_but = np.reshape(mean_all_but, (-1, 1))
            mean_rest = np.mean(datas_t[lmax:])
            if lmax < k:
                mean_vec = np.concatenate([np.squeeze(mean_all_but), np.array([mean_rest])])
                cx = np.concatenate([cx, np.array([mean_rest])])
            else:
                mean_vec = np.squeeze(mean_all_but)
            lx = np.zeros(k, dtype=int)
            for i in range(k):
                for j in range(len(cx)):
                    if mean_vec[inds_sort_t[i]] == cx[j]:
                        lx[i] = j
                        continue
            label_list.append(lx)
            support_list.append(cx)

    else:
        for t in range(T_h):
            # print('t = ' + str(t))
            data_t = data[:, t:t+1]
            kmx = KMeans(n_clusters=klist[t]).fit(data_t)
            cx = kmx.cluster_centers_
            cx = np.round(cx, decimals=tol_decimals)
            lx = kmx.labels_
            label_list.append(lx)
            support_list.append(cx)

    if use_weights == 0:  # weight all cluster centers equally? ... Convenient but theoretically flawed I think
        out = np.zeros([k, T_h])
        for t in range(T_h):
            out[:, t] = support_list[t][label_list[t]][:, 0]
        return out

    # build output measure
    for i in range(k):
        cur_path = np.zeros(T_h)
        for t in range(T_h):
            cur_path[t] = support_list[t][label_list[t][i]]

        # check whether the path already exists
        path_is_here = 0
        for j in range(len(out_w)):
            if np.all(out_x[j, :] == cur_path):
                out_w[j] += 1 / k
                path_is_here = 1
                break
        if not path_is_here:
            out_x = np.append(out_x, np.expand_dims(cur_path, axis=0), axis=0)
            out_w.append(1 / k)

    return out_x, out_w
