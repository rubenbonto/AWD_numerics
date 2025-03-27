from trees.multi_dimension.Multidimension_trees import *
from tqdm import tqdm

import ot


def solver_lp_pot(distance_matrix_subset, pi_ratios, pi_tilde_ratios, reg=1e-2):
    """
    Solve for the optimal transport plan using the POT library's (fast!) EMD solver.

    Parameters:
    - distance_matrix_subset (np.ndarray): A 2D cost matrix.
    - pi_ratios (np.ndarray): 1D source distribution (row marginals).
    - pi_tilde_ratios (np.ndarray): 1D target distribution (column marginals).

    Returns:
    - np.ndarray: The optimal transport plan (probability matrix).
    """
    pi_ratios = np.array(pi_ratios, dtype=np.float64)
    pi_tilde_ratios = np.array(pi_tilde_ratios, dtype=np.float64)

    return ot.lp.emd(pi_ratios, pi_tilde_ratios, distance_matrix_subset)


def multidim_nested_optimal_transport_loop(tree1_root, tree2_root, max_depth, power):
    """
    Computes the nested optimal transport plan between two trees.

    Parameters:
    - tree1_root (TreeNode): Root of the first tree.
    - tree2_root (TreeNode): Root of the second tree.
    - max_depth (int): Maximum depth to compute.
    - power (int): The power parameter for the cost.

    Returns:
    - float: Computed nested distance.
    """
    full_distance_matrix = multidim_compute_distance_matrix_at_depth(
        tree1_root, tree2_root, max_depth, power
    )

    # Outer loop: iterate over depths
    for depth in range(max_depth - 1, -1, -1):
        paths_tree1 = multidim_get_nodes_at_depth(tree1_root, depth)
        paths_tree2 = multidim_get_nodes_at_depth(tree2_root, depth)

        nodes_tree1 = [
            multidim_find_node_by_path(tree1_root, path) for path in paths_tree1
        ]
        nodes_tree2 = [
            multidim_find_node_by_path(tree2_root, path) for path in paths_tree2
        ]

        children_count_tree1 = [len(node.children) for node in nodes_tree1 if node]
        children_count_tree2 = [len(node.children) for node in nodes_tree2 if node]

        updated_distance_matrix = np.zeros((len(paths_tree1), len(paths_tree2)))

        # Progress bar per depth
        with tqdm(
            total=len(paths_tree1) * len(paths_tree2),
            desc=f"Depth {depth}",
            unit="pair",
        ) as pbar:
            # Inner loop: iterate over pairs of paths
            for i, path1 in enumerate(paths_tree1):
                for j, path2 in enumerate(paths_tree2):
                    step_name = (depth, to_hashable(path1[-1]), to_hashable(path2[-1]))
                    start_row, end_row = sum(children_count_tree1[:i]), sum(
                        children_count_tree1[: i + 1]
                    )
                    start_col, end_col = sum(children_count_tree2[:j]), sum(
                        children_count_tree2[: j + 1]
                    )
                    sub_matrix = full_distance_matrix[
                        start_row:end_row, start_col:end_col
                    ]

                    pi_ratios, pi_tilde_ratios = (
                        multidim_compute_marginal_probabilities_for_subset(
                            path1, path2, tree1_root, tree2_root
                        )
                    )

                    probability_matrix = solver_lp_pot(
                        sub_matrix, pi_ratios, pi_tilde_ratios
                    )

                    cost = np.sum(probability_matrix * sub_matrix)
                    updated_distance_matrix[i, j] = cost

                    pbar.update(1)  # Update progress bar per iteration

        full_distance_matrix = updated_distance_matrix

    return full_distance_matrix[0][0]


def multidim_compute_nested_distance(
    tree1_root,
    tree2_root,
    max_depth,
    power=1,
):
    """
    Computes the nested Wasserstein distance between two trees.

    Parameters:
    - tree1_root (TreeNode): Root of the first tree.
    - tree2_root (TreeNode): Root of the second tree.
    - max_depth (int): Maximum depth to compute.
    - power (int): The power parameter for the cost.

    Returns:
    - float: Computed nested distance.
    """
    distance = multidim_nested_optimal_transport_loop(
        tree1_root, tree2_root, max_depth, power
    )

    return distance


import numpy as np
import concurrent.futures
from tqdm import tqdm
import ot


# Helper function that computes the cost for a given pair (i, j)
def compute_cost(args):
    (
        i,
        j,
        path1,
        path2,
        tree1_root,
        tree2_root,
        full_distance_matrix,
        start_row,
        end_row,
        start_col,
        end_col,
    ) = args
    sub_matrix = full_distance_matrix[start_row:end_row, start_col:end_col]
    # Compute marginal probabilities for the subset defined by path1 and path2
    pi_ratios, pi_tilde_ratios = multidim_compute_marginal_probabilities_for_subset(
        path1, path2, tree1_root, tree2_root
    )
    probability_matrix = solver_lp_pot(sub_matrix, pi_ratios, pi_tilde_ratios)
    cost = np.sum(probability_matrix * sub_matrix)
    return i, j, cost


def multidim_nested_optimal_transport_loop_parallel(
    tree1_root, tree2_root, max_depth, power, n_processes
):
    """
    Computes the nested optimal transport plan between two trees in parallel using
    concurrent.futures.ProcessPoolExecutor for the inner loops.

    Parameters:
    - tree1_root (TreeNode): Root of the first tree.
    - tree2_root (TreeNode): Root of the second tree.
    - max_depth (int): Maximum depth to compute.
    - power (int): The power parameter for the cost.
    - n_processes (int): Number of processes to use.

    Returns:
    - float: Computed nested distance.
    """
    # Compute the full distance matrix at the deepest level
    full_distance_matrix = multidim_compute_distance_matrix_at_depth(
        tree1_root, tree2_root, max_depth, power
    )

    # Loop over depths from max_depth-1 down to 0
    for depth in range(max_depth - 1, -1, -1):
        # Get the node paths and corresponding nodes at the current depth
        paths_tree1 = multidim_get_nodes_at_depth(tree1_root, depth)
        paths_tree2 = multidim_get_nodes_at_depth(tree2_root, depth)
        nodes_tree1 = [
            multidim_find_node_by_path(tree1_root, path) for path in paths_tree1
        ]
        nodes_tree2 = [
            multidim_find_node_by_path(tree2_root, path) for path in paths_tree2
        ]

        # Get the number of children for each node (filtering out any None nodes)
        children_count_tree1 = [len(node.children) for node in nodes_tree1 if node]
        children_count_tree2 = [len(node.children) for node in nodes_tree2 if node]

        # Compute cumulative indices to slice the full distance matrix
        cum_children_tree1 = [0] + list(np.cumsum(children_count_tree1))
        cum_children_tree2 = [0] + list(np.cumsum(children_count_tree2))

        updated_distance_matrix = np.zeros((len(paths_tree1), len(paths_tree2)))

        # Prepare tasks for the inner loop over all pairs of paths
        tasks = []
        for i, path1 in enumerate(paths_tree1):
            for j, path2 in enumerate(paths_tree2):
                start_row = cum_children_tree1[i]
                end_row = cum_children_tree1[i + 1]
                start_col = cum_children_tree2[j]
                end_col = cum_children_tree2[j + 1]
                tasks.append(
                    (
                        i,
                        j,
                        path1,
                        path2,
                        tree1_root,
                        tree2_root,
                        full_distance_matrix,
                        start_row,
                        end_row,
                        start_col,
                        end_col,
                    )
                )

        # Use ProcessPoolExecutor to parallelize the inner loop computations
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_processes
        ) as executor:
            # Submit tasks to the executor
            futures = {
                executor.submit(compute_cost, task): (task[0], task[1])
                for task in tasks
            }
            # Progress bar to track inner loop tasks
            with tqdm(total=len(tasks), desc=f"Depth {depth}", unit="pair") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    i, j, cost = future.result()
                    updated_distance_matrix[i, j] = cost
                    pbar.update(1)

        # Update the full distance matrix for the next (shallower) depth iteration
        full_distance_matrix = updated_distance_matrix

    return full_distance_matrix[0][0]
