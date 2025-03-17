from Multi_dimension.Multidimension_trees import *
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
