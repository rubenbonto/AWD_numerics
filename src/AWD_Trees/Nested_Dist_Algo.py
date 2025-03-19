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

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np


def process_chunk(
    chunk_indices,
    depth,
    paths_tree1,
    paths_tree2,
    children_count_tree1,
    children_count_tree2,
    full_distance_matrix,
    tree1_root,
    tree2_root,
    power,
):
    """
    Processes a chunk of outer-loop indices (i values) sequentially.
    For each i in chunk_indices, iterates over all j in paths_tree2.
    Returns a list of results for each i in the chunk.
    Each result is a tuple: (i, list of (j, cost, step_name, probability_matrix))
    """
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
            cost = 0
            # Check for trivial (deterministic) cases:
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


def nested_optimal_transport_loop_parallel(tree1_root, tree2_root, max_depth, power):
    """
    Parallelized version of nested_optimal_transport_loop that splits the outer loop (over paths_tree1)
    into a fixed number of chunks if the number of indices is high.
    For small numbers (e.g., less than 50 indices), processing is done sequentially.
    """
    full_distance_matrix = compute_distance_matrix_at_depth(
        tree1_root, tree2_root, max_depth, power
    )

    # Loop over depths (backward induction)
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
        parallel_threshold = 200  # threshold below which we process sequentially

        if len(indices) < parallel_threshold:
            # Process sequentially (no overhead of task creation)
            chunk_results = process_chunk(
                indices,
                depth,
                paths_tree1,
                paths_tree2,
                children_count_tree1,
                children_count_tree2,
                full_distance_matrix,
                tree1_root,
                tree2_root,
                power,
            )
            for i, row_result in chunk_results:
                for j, cost in row_result:
                    updated_distance_matrix[i, j] = cost
        else:
            # Partition the outer indices into chunks (e.g., 12 chunks)
            num_chunks = 6
            chunks = np.array_split(indices, num_chunks)
            with ProcessPoolExecutor(max_workers=num_chunks) as executor:
                futures = []
                for chunk in chunks:
                    futures.append(
                        executor.submit(
                            process_chunk,
                            list(chunk),
                            depth,
                            paths_tree1,
                            paths_tree2,
                            children_count_tree1,
                            children_count_tree2,
                            full_distance_matrix,
                            tree1_root,
                            tree2_root,
                            power,
                        )
                    )
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Parallel Depth {depth}",
                ):
                    chunk_results = future.result()
                    for i, row_result in chunk_results:
                        for j, cost in row_result:
                            updated_distance_matrix[i, j] = cost

        full_distance_matrix = (
            updated_distance_matrix  # update for the next depth level
        )

    return full_distance_matrix[0][0]


def compute_nested_distance_parallel(
    tree1_root, tree2_root, max_depth, return_matrix=False, power=1
):
    """
    Computes the nested Wasserstein distance between two trees using only lp.emd.
    """
    distance = nested_optimal_transport_loop_parallel(
        tree1_root, tree2_root, max_depth, power
    )
    return distance
