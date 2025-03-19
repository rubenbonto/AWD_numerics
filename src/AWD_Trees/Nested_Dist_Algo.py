import numpy as np
from collections import deque
from awd_trees.Discrete_OT_Solver_algo import *
from trees.Tree_AWD_utilities import *

"""
This module implements Algorithm 1 from:

Pichler, A., & Weinhardt, M. (2021). "Nested Sinkhorn Divergence To Compute The Nested Distance."
   - arXiv:2102.05413 [math.OC]. Available at: https://arxiv.org/abs/2102.05413

It computes the adapted Wasserstein distance between trees using a nested loop structure.
It supports three different inner optimal transport solvers:
- POT library (Earth Mover's Distance - EMD)
- Linear programming (LP)
- Sinkhorn algorithm (Entropic regularization)
"""
from tqdm import tqdm


def nested_optimal_transport_loop(
    tree1_root, tree2_root, max_depth, method, lambda_reg, power
):
    """
    Computes the nested optimal transport plan between two trees.

    Parameters:
    - tree1_root (TreeNode): Root of the first tree.
    - tree2_root (TreeNode): Root of the second tree.
    - max_depth (int): Maximum depth to compute.
    - method (str): Solver method: "Sinkhorn", "solver_lp", or "solver_pot".
    - lambda_reg (float): Regularization parameter for Sinkhorn (only used if method="Sinkhorn").

    Returns:
    - float: Computed nested distance.
    - dict: Dictionary of probability matrices for each step.
    """
    if method == "Sinkhorn" and lambda_reg <= 0:
        raise ValueError("Lambda must be positive when using Sinkhorn iteration.")
    elif method not in (
        "solver_lp",
        "solver_pot",
        "Sinkhorn",
        "solver_sinkhorn",
        "solver_lp_pot",
        "solver_pot_sinkhorn",
        "solver_jax",
    ):
        raise ValueError(
            "Method must be one of 'Sinkhorn', 'solver_lp', 'solver_jax', 'solver_lp_pot', 'solver_pot_sinkhorn' or 'solver_pot'."
        )

    probability_matrices = {}
    full_distance_matrix = compute_distance_matrix_at_depth(
        tree1_root, tree2_root, max_depth, power
    )

    for depth in range(max_depth - 1, -1, -1):
        paths_tree1 = get_nodes_at_depth(tree1_root, depth)
        paths_tree2 = get_nodes_at_depth(tree2_root, depth)

        nodes_tree1 = [find_node_by_path(tree1_root, path) for path in paths_tree1]
        nodes_tree2 = [find_node_by_path(tree2_root, path) for path in paths_tree2]

        children_count_tree1 = [len(node.children) for node in nodes_tree1 if node]
        children_count_tree2 = [len(node.children) for node in nodes_tree2 if node]

        updated_distance_matrix = np.zeros((len(paths_tree1), len(paths_tree2)))

        tqdm_bar = tqdm(
            enumerate(paths_tree1), total=len(paths_tree1), desc=f"Depth {depth}"
        )
        for i, path1 in tqdm_bar:
            for j, path2 in enumerate(paths_tree2):
                step_name = (depth, path1[-1], path2[-1])

                start_row, end_row = sum(children_count_tree1[:i]), sum(
                    children_count_tree1[: i + 1]
                )
                start_col, end_col = sum(children_count_tree2[:j]), sum(
                    children_count_tree2[: j + 1]
                )
                sub_matrix = full_distance_matrix[start_row:end_row, start_col:end_col]

                pi_ratios, pi_tilde_ratios = compute_marginal_probabilities_for_subset(
                    path1, path2, tree1_root, tree2_root
                )

                if method == "Sinkhorn":
                    probability_matrix = Sinkhorn_iteration(
                        sub_matrix,
                        pi_ratios,
                        pi_tilde_ratios,
                        stopping_criterion=1e-4,
                        lambda_reg=lambda_reg,
                    )
                elif method == "solver_lp_pot":
                    probability_matrix = solver_lp_pot(
                        sub_matrix, pi_ratios, pi_tilde_ratios
                    )
                elif method == "solver_lp":
                    probability_matrix = solver_lp(
                        sub_matrix, pi_ratios, pi_tilde_ratios
                    )
                elif method == "solver_jax":
                    probability_matrix = solver_jax(
                        sub_matrix, pi_ratios, pi_tilde_ratios, epsilon=(1 / lambda_reg)
                    )
                elif method == "solver_pot_sinkhorn":
                    probability_matrix = solver_pot_sinkhorn(
                        sub_matrix, pi_ratios, pi_tilde_ratios, epsilon=(1 / lambda_reg)
                    )
                elif method == "solver_pot_1D":
                    probability_matrix = solver_pot_1D(
                        sub_matrix, pi_ratios, pi_tilde_ratios
                    )
                else:
                    probability_matrix = solver_pot(
                        sub_matrix, pi_ratios, pi_tilde_ratios
                    )

                cost = np.sum(probability_matrix * sub_matrix)
                probability_matrices[step_name] = probability_matrix
                updated_distance_matrix[i, j] = cost

        full_distance_matrix = updated_distance_matrix

    return full_distance_matrix[0][0], probability_matrices


# DO NOT USE FOR BIG PROBLEMS
def compute_final_probability_matrix(
    probability_matrices, tree1_root, tree2_root, max_depth
):
    """
    Combines probability matrices along all paths to compute the final probability matrix.

    Parameters:
    - probability_matrices (dict): Dictionary of probability matrices for each step.
    - tree1_root (TreeNode): Root of the first tree.
    - tree2_root (TreeNode): Root of the second tree.
    - max_depth (int): Maximum depth considered.

    Returns:
    - np.ndarray: Final probability matrix representing nested distance.
    """
    paths_tree1 = get_paths_to_leaves(tree1_root, max_depth)
    paths_tree2 = get_paths_to_leaves(tree2_root, max_depth)
    final_prob_matrix = np.zeros((len(paths_tree1), len(paths_tree2)))

    for i, path1 in enumerate(paths_tree1):
        for j, path2 in enumerate(paths_tree2):
            probability = 1.0
            for depth in range(max_depth):
                if depth >= len(path1) or depth >= len(path2):
                    break
                step_name = (depth, path1[depth], path2[depth])
                prob_matrix = probability_matrices.get(step_name, None)
                if prob_matrix is None or prob_matrix.size == 0:
                    probability = 0
                    break
                next_node1 = path1[depth + 1] if depth + 1 < len(path1) else None
                next_node2 = path2[depth + 1] if depth + 1 < len(path2) else None
                successors_node1 = [
                    child[-1]
                    for child in get_paths_to_leaves(tree1_root, depth + 1)
                    if child[:-1] == path1[: depth + 1]
                ]
                successors_node2 = [
                    child[-1]
                    for child in get_paths_to_leaves(tree2_root, depth + 1)
                    if child[:-1] == path2[: depth + 1]
                ]
                try:
                    index1 = successors_node1.index(next_node1)
                    index2 = successors_node2.index(next_node2)
                except ValueError:
                    probability = 0
                    break
                probability *= prob_matrix[index1, index2]
            final_prob_matrix[i, j] = probability
    return final_prob_matrix


def compute_nested_distance(
    tree1_root,
    tree2_root,
    max_depth,
    return_matrix=False,
    method="solver_lp",
    lambda_reg=0,
    power=1,
):
    """
    Computes the nested Wasserstein distance between two trees.

    Parameters:
    - tree1_root (TreeNode): Root of the first tree.
    - tree2_root (TreeNode): Root of the second tree.
    - max_depth (int): Maximum depth to compute.
    - return_matrix (bool): If True, returns the final probability matrix.
    - method (str): Solver method: "Sinkhorn", "solver_lp", or "solver_pot".
    - lambda_reg (float): Regularization parameter for Sinkhorn.

    Returns:
    - float: Computed nested distance.
    - np.ndarray (optional): Final probability matrix if return_matrix is True.
    """
    distance, probability_matrices = nested_optimal_transport_loop(
        tree1_root, tree2_root, max_depth, method, lambda_reg, power
    )

    if return_matrix:
        final_prob_matrix = compute_final_probability_matrix(
            probability_matrices, tree1_root, tree2_root, max_depth
        )
        return distance, final_prob_matrix
    return distance


# ---------------- Parallel Nested OT Using Chunked Outer Loop ----------------

import os
import sys
import ray
import numpy as np
from tqdm.notebook import tqdm  # Use notebook-friendly tqdm

# Ensure work directory is loaded properly for Ray use.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

ray.shutdown()
ray.init(
    ignore_reinit_error=True,
    include_dashboard=False,
    runtime_env={"working_dir": project_root},
)

from trees.Tree_AWD_utilities import *
from trees.Tree_Node import *
from awd_trees.Discrete_OT_Solver_algo import *


def process_chunk_seq(
    chunk_indices,
    paths_tree1,
    paths_tree2,
    children_count_tree1,
    children_count_tree2,
    full_distance_matrix,
    tree1_root,
    tree2_root,
):
    results = []
    for i in chunk_indices:
        row_result = []
        path1 = paths_tree1[i]
        for j, path2 in enumerate(paths_tree2):
            start_row = sum(children_count_tree1[:i])
            end_row = sum(children_count_tree1[: i + 1])
            start_col = sum(children_count_tree2[:j])
            end_col = sum(children_count_tree2[: j + 1])
            sub_matrix = full_distance_matrix[start_row:end_row, start_col:end_col]
            pi_ratios, pi_tilde_ratios = compute_marginal_probabilities_for_subset(
                path1, path2, tree1_root, tree2_root
            )
            if len(pi_ratios) == 1 and len(pi_tilde_ratios) == 1:
                probability_matrix = np.array([[1]])
                cost = np.sum(probability_matrix * sub_matrix)
            elif len(pi_ratios) == 1:
                probability_matrix = np.array([pi_tilde_ratios])
                cost = np.sum(probability_matrix * sub_matrix)
            elif len(pi_tilde_ratios) == 1:
                probability_matrix = np.array([[p] for p in pi_ratios])
                cost = np.sum(probability_matrix * sub_matrix)
            else:
                cost = solver_lp_pot_distance(sub_matrix, pi_ratios, pi_tilde_ratios)
            row_result.append((j, cost))
        results.append((i, row_result))
    return results


# Ray remote version for parallel processing (without inner tqdm)
@ray.remote
def process_chunk_remote(
    chunk_indices,
    paths_tree1,
    paths_tree2,
    children_count_tree1,
    children_count_tree2,
    full_distance_matrix,  # A Ray object reference (read-only)
    tree1_root,  # A Ray object reference (read-only)
    tree2_root,  # A Ray object reference (read-only)
):
    results = []
    # Removed inner tqdm to avoid messy notebook output.
    for i in chunk_indices:
        row_result = []
        path1 = paths_tree1[i]
        for j, path2 in enumerate(paths_tree2):
            start_row = sum(children_count_tree1[:i])
            end_row = sum(children_count_tree1[: i + 1])
            start_col = sum(children_count_tree2[:j])
            end_col = sum(children_count_tree2[: j + 1])
            sub_matrix = full_distance_matrix[start_row:end_row, start_col:end_col]
            pi_ratios, pi_tilde_ratios = compute_marginal_probabilities_for_subset(
                path1, path2, tree1_root, tree2_root
            )
            if len(pi_ratios) == 1 and len(pi_tilde_ratios) == 1:
                probability_matrix = np.array([[1]])
                cost = np.sum(probability_matrix * sub_matrix)
            elif len(pi_ratios) == 1:
                probability_matrix = np.array([pi_tilde_ratios])
                cost = np.sum(probability_matrix * sub_matrix)
            elif len(pi_tilde_ratios) == 1:
                probability_matrix = np.array([[p] for p in pi_ratios])
                cost = np.sum(probability_matrix * sub_matrix)
            else:
                cost = solver_lp_pot_distance(sub_matrix, pi_ratios, pi_tilde_ratios)
            row_result.append((j, cost))
        results.append((i, row_result))
    return results


def nested_optimal_transport_loop_parallel(
    tree1_root, tree2_root, max_depth, power, num_chunks
):
    full_distance_matrix = compute_distance_matrix_at_depth(
        tree1_root, tree2_root, max_depth, power
    )
    ray_tree1_root = ray.put(tree1_root)
    ray_tree2_root = ray.put(tree2_root)

    for depth in range(max_depth - 1, -1, -1):
        print("Depth:", depth)
        paths_tree1 = get_nodes_at_depth(tree1_root, depth)
        paths_tree2 = get_nodes_at_depth(tree2_root, depth)
        nodes_tree1 = [find_node_by_path(tree1_root, path) for path in paths_tree1]
        nodes_tree2 = [find_node_by_path(tree2_root, path) for path in paths_tree2]
        children_count_tree1 = [len(node.children) for node in nodes_tree1 if node]
        children_count_tree2 = [len(node.children) for node in nodes_tree2 if node]

        updated_distance_matrix = np.zeros((len(paths_tree1), len(paths_tree2)))
        indices = list(range(len(paths_tree1)))
        parallel_threshold = 400

        if len(indices) < parallel_threshold:
            chunk_results = process_chunk_seq(
                indices,
                paths_tree1,
                paths_tree2,
                children_count_tree1,
                children_count_tree2,
                full_distance_matrix,
                tree1_root,
                tree2_root,
            )
        else:
            # Use the provided number of chunks.
            split_chunks = np.array_split(indices, num_chunks)
            ray_full_distance_matrix = ray.put(full_distance_matrix)
            tasks = [
                process_chunk_remote.remote(
                    list(chunk),
                    paths_tree1,
                    paths_tree2,
                    children_count_tree1,
                    children_count_tree2,
                    ray_full_distance_matrix,
                    ray_tree1_root,
                    ray_tree2_root,
                )
                for chunk in split_chunks
            ]
            chunk_results = []
            remaining = tasks.copy()
            # Outer progress bar to track the remote tasks.
            pbar = tqdm(total=len(remaining), desc=f"Parallel Depth {depth}")
            while remaining:
                done, remaining = ray.wait(remaining, num_returns=1, timeout=1)
                for task in done:
                    chunk_results.extend(ray.get(task))
                    pbar.update(1)
            pbar.close()

        for i, row_result in chunk_results:
            for j, cost in row_result:
                updated_distance_matrix[i, j] = cost

        full_distance_matrix = updated_distance_matrix

    return full_distance_matrix[0, 0]


def compute_nested_distance_parallel(
    tree1_root, tree2_root, max_depth, power=1, num_chunks=6
):
    distance = nested_optimal_transport_loop_parallel(
        tree1_root, tree2_root, max_depth, power, num_chunks
    )
    return distance


# --------------- For you framework----------------------------

import os
import sys
import ray
import numpy as np
from tqdm.notebook import tqdm  # Use notebook-friendly tqdm
import ot  # POT library for OT computations

# Ensure the project root (e.g., "src") is in PYTHONPATH so that your modules are found.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Initialize Ray with the working directory shipped.
ray.shutdown()
ray.init(
    ignore_reinit_error=True,
    include_dashboard=False,
    runtime_env={"working_dir": project_root},
)

# Import your modules.
from trees.Tree_AWD_utilities import *  # if needed for tree-adapted code
from trees.Tree_Node import *
from awd_trees.Discrete_OT_Solver_algo import *


# -----------------------------------------------------------------------------
# Remote function to compute the OT cost for a list of condition pairs.
# This function is agnostic to whether the keys are Markovian (single values)
# or non-Markovian (tuples with full history).
@ray.remote
def compute_pair_costs_remote(pair_list):
    results = []
    for item in pair_list:
        # Unpack parameters for one pair:
        # k1, v1: key and next-step distribution from mu_x[t]
        # k2, v2: key and next-step distribution from nu_y[t]
        # t: current time index
        # markovian: flag
        # square_cost_matrix: global squared cost matrix
        # V_next: V[t+1] matrix (or None if t==T-1)
        # v2q_x_next, v2q_y_next: re-quantization mappings for time t+1 (if available)
        # v2q_x_t, v2q_y_t: current mappings for time t (to index V[t])
        (
            k1,
            v1,
            k2,
            v2,
            t,
            markovian,
            square_cost_matrix,
            V_next,
            v2q_x_next,
            v2q_y_next,
            v2q_x_t,
            v2q_y_t,
        ) = item

        # Normalize the conditional distributions.
        w1 = np.array(list(v1.values()), dtype=float)
        w1 /= w1.sum()
        w2 = np.array(list(v2.values()), dtype=float)
        w2 /= w2.sum()

        # Extract the quantized next values.
        q1 = list(v1.keys())
        q2 = list(v2.keys())
        cost = square_cost_matrix[np.ix_(q1, q2)]

        # If we're not at the final time step, add the future cost V[t+1].
        if V_next is not None:
            if markovian:
                # In the Markovian case, the condition is determined solely by the next value.
                q1s = [v2q_x_next[q] for q in v1.keys()]
                q2s = [v2q_y_next[q] for q in v2.keys()]
            else:
                # In the non-adapted (full history) case, extend the current condition keys.
                q1s = [v2q_x_next[k1 + (q,)] for q in v1.keys()]
                q2s = [v2q_y_next[k2 + (q,)] for q in v2.keys()]
            cost = cost + V_next[np.ix_(q1s, q2s)]
        try:
            distance = ot.emd2(w1, w2, cost)
        except Exception as e:
            print("Error processing pair", k1, k2, e)
            distance = np.inf

        # Map the condition keys for time t to indices in V[t].
        i_idx = v2q_x_t[k1]
        j_idx = v2q_y_t[k2]
        results.append((i_idx, j_idx, distance))
    return results


def nested_parallel_generic(
    mu_x, nu_y, v2q_x, v2q_y, q2v, markovian=False, num_chunks=6
):
    """
    Computes the nested distance using parallel backward induction.

    Parameters:
      mu_x, nu_y: Lists (over time t) of dictionaries mapping condition keys to dictionaries
                  of next-step counts.
      v2q_x, v2q_y: Lists (over time t) of dictionaries mapping condition keys to indices.
      q2v: Global array of quantized values.
      markovian: If True, keys are assumed to be simple (e.g., integers). Otherwise,
                 keys are tuples representing full history.
      num_chunks: Number of chunks to partition the pair computations at each time step.

    Returns:
      AW_2square: The nested distance (squared), taken as V[0][0,0].
      V: List of value matrices computed at each time step.
    """
    T = len(mu_x)
    square_cost_matrix = (q2v[None, :] - q2v[None, :].T) ** 2
    V = [np.zeros((len(v2q_x[t]), len(v2q_y[t]))) for t in range(T)]
    print("Parallel nested backward induction ...")

    for t in range(T - 1, -1, -1):
        print(f"Time step {t}")
        pair_list = []
        for k1, v1 in mu_x[t].items():
            for k2, v2 in nu_y[t].items():
                if t < T - 1:
                    V_next = V[t + 1]
                    v2q_x_next = v2q_x[t + 1]
                    v2q_y_next = v2q_y[t + 1]
                else:
                    V_next = None
                    v2q_x_next = None
                    v2q_y_next = None
                pair_list.append(
                    (
                        k1,
                        v1,
                        k2,
                        v2,
                        t,
                        markovian,
                        square_cost_matrix,
                        V_next,
                        v2q_x_next,
                        v2q_y_next,
                        v2q_x[t],
                        v2q_y[t],
                    )
                )
        if len(pair_list) <= num_chunks:
            chunks = [pair_list]
        else:
            chunk_size = (len(pair_list) + num_chunks - 1) // num_chunks
            chunks = [
                pair_list[i * chunk_size : (i + 1) * chunk_size]
                for i in range(num_chunks)
            ]

        # Launch remote tasks.
        tasks = [compute_pair_costs_remote.remote(chunk) for chunk in chunks]
        results = []
        for res in tqdm(ray.get(tasks), desc=f"Time step {t} chunks", leave=False):
            results.extend(res)

        # Update V[t] using the computed distances.
        for i_idx, j_idx, dist in results:
            V[t][i_idx, j_idx] = dist

    AW_2square = V[0][0, 0]
    return AW_2square


def compute_nested_distance_parallel_generic(
    mu_x, nu_y, v2q_x, v2q_y, q2v, markovian, num_chunks=6
):
    return nested_parallel_generic(
        mu_x, nu_y, v2q_x, v2q_y, q2v, markovian=markovian, num_chunks=num_chunks
    )
