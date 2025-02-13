import numpy as np
from collections import deque
from Discrete_OT_Solver.LP_solver import *
from Discrete_OT_Solver.Sinkhorn import Sinkhorn_iteration

from Tree_AWD_utilities import *


def nested_optimal_transport_loop(tree1_root, tree2_root, max_depth, method, lambda_reg):
    """
    Sets up the nested loop structure for computing nested distance and initializes storage for probability matrices.

    Parameters:
    - tree1_root (TreeNode): Root of the first tree.
    - tree2_root (TreeNode): Root of the second tree.
    - max_depth (int): Maximum depth to compute.
    - method (str): The method to use for solving the optimal transport problem. 
                    Must be one of: "Sinkhorn", "solver_lp", or "solver_pot".
    - lambda_reg (float): Regularization parameter for Sinkhorn (required only if method=="Sinkhorn").

    Returns:
    - float: Computed nested distance.
    - dict: Dictionary of probability matrices for each step.
    """
    if method == "Sinkhorn":
        if lambda_reg <= 0:
            raise ValueError("Lambda must be positive when using Sinkhorn iteration.")
    elif method not in ("solver_lp", "solver_pot"):
        raise ValueError("Method must be one of 'Sinkhorn', 'solver_lp', or 'solver_pot'.")

    probability_matrices = {}

    # Initialize the full distance matrix at max_depth
    full_distance_matrix = compute_distance_matrix_at_depth(tree1_root, tree2_root, max_depth)

    # Iterate from max_depth-1 down to 0
    for depth in range(max_depth - 1, -1, -1):
        # Retrieve paths to nodes at the current depth
        paths_tree1 = get_nodes_at_depth(tree1_root, depth)  # List of paths
        paths_tree2 = get_nodes_at_depth(tree2_root, depth)

        # Convert paths to TreeNode objects
        nodes_tree1 = [find_node_by_path(tree1_root, path) for path in paths_tree1]
        nodes_tree2 = [find_node_by_path(tree2_root, path) for path in paths_tree2]

        # Count the number of children for each node
        children_count_tree1 = [len(node.children) for node in nodes_tree1 if node]
        children_count_tree2 = [len(node.children) for node in nodes_tree2 if node]

        # Initialize updated distance matrix for the current depth
        updated_distance_matrix = np.zeros((len(paths_tree1), len(paths_tree2)))

        for i, path1 in enumerate(paths_tree1):
            for j, path2 in enumerate(paths_tree2):
                step_name = (depth, path1[-1], path2[-1])
                
                # Calculate indices for submatrix extraction
                start_row = sum(children_count_tree1[:i])
                end_row = start_row + children_count_tree1[i]
                start_col = sum(children_count_tree2[:j])
                end_col = start_col + children_count_tree2[j]

                sub_matrix = full_distance_matrix[start_row:end_row, start_col:end_col]

                pi_ratios, pi_tilde_ratios = compute_marginal_probabilities_for_subset(
                    path1, path2, tree1_root, tree2_root
                )
                
                # Determine the transport plan using the selected method
                if method == "Sinkhorn":
                    probability_matrix = Sinkhorn_iteration(
                        sub_matrix,
                        pi_ratios,
                        pi_tilde_ratios,
                        stopping_criterion=1e-4,
                        lambda_reg=lambda_reg
                    )
                elif method == "solver_lp":
                    probability_matrix = solver_lp(sub_matrix, pi_ratios, pi_tilde_ratios)
                elif method == "solver_pot":
                    probability_matrix = solver_pot(sub_matrix, pi_ratios, pi_tilde_ratios)
                else:
                    # This branch should never be reached due to the earlier check.
                    raise ValueError("Method must be one of 'Sinkhorn', 'solver_lp', or 'solver_pot'.")

                cost = np.sum(probability_matrix * sub_matrix)
                probability_matrices[step_name] = probability_matrix
                updated_distance_matrix[i, j] = cost

        # Update the full distance matrix for the next iteration
        full_distance_matrix = updated_distance_matrix

    return full_distance_matrix[0][0], probability_matrices


def compute_final_probability_matrix(probability_matrices, tree1_root, tree2_root, max_depth):
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
    # Get all paths to leaves for both trees
    paths_tree1 = get_paths_to_leaves(tree1_root, max_depth)
    paths_tree2 = get_paths_to_leaves(tree2_root, max_depth)

    # Initialize the final probability matrix
    final_prob_matrix = np.zeros((len(paths_tree1), len(paths_tree2)))

    # Iterate over all leaf node pairs
    for i, path1 in enumerate(paths_tree1):
        for j, path2 in enumerate(paths_tree2):

            probability = 1.0

            # Traverse each depth level
            for depth in range(max_depth):
                node1 = path1[depth]
                node2 = path2[depth]

                # Retrieve the corresponding probability matrix
                step_name = (depth, node1, node2)
                prob_matrix = probability_matrices.get(step_name, None)
                if not isinstance(prob_matrix, np.ndarray) or prob_matrix.size == 0:
                    probability = 0
                    break


                if depth + 1 >= len(path1) or depth + 1 >= len(path2):
                    continue  # Skip iteration if depth exceeds available indices

                next_node1 = path1[depth + 1]
                next_node2 = path2[depth + 1]


                # Precompute paths to avoid redundant calls
                precomputed_paths_tree1 = {d: get_paths_to_leaves(tree1_root, d) for d in range(1, max_depth + 1)}
                precomputed_paths_tree2 = {d: get_paths_to_leaves(tree2_root, d) for d in range(1, max_depth + 1)}

                successors_node1 = [child[-1] for child in precomputed_paths_tree1.get(depth + 1, []) if child[:-1] == path1[:depth + 1]]
                successors_node2 = [child[-1] for child in precomputed_paths_tree2.get(depth + 1, []) if child[:-1] == path2[:depth + 1]]


                # Get the positions of the next nodes
                try:
                    index1 = successors_node1.index(next_node1)
                    index2 = successors_node2.index(next_node2)
                except ValueError:
                    probability = 0
                    break

                # Ensure indices are within bounds before accessing the matrix
                if index1 < prob_matrix.shape[0] and index2 < prob_matrix.shape[1]:
                    probability *= prob_matrix[index1, index2]
                else:
                    #print(f"Warning: Indices out of bounds! Step: {step_name}, prob_matrix shape: {prob_matrix.shape}, index1: {index1}, index2: {index2}")
                    probability = 0
                    break


            # Assign the computed probability to the final matrix
            final_prob_matrix[i, j] = probability

    return final_prob_matrix


def compute_nested_distance(tree1_root, tree2_root, max_depth, return_matrix=False, method="solver_lp", lambda_reg=0):
    """
    Computes the nested distance between two trees using specified algorithms.

    Parameters:
    - tree1_root (TreeNode): Root of the first tree.
    - tree2_root (TreeNode): Root of the second tree.
    - max_depth (int): Maximum depth to compute.
    - return_matrix (bool): If True, returns the final probability matrix alongside the distance.
    - use_sinkhorn (bool): If True, uses Sinkhorn iterations instead of linear programming.
    - lambda_reg (float): Regularization parameter for Sinkhorn.

    Returns:
    - float: Computed nested distance.
    - np.ndarray (optional): Final probability matrix if return_matrix is True.
    """
    distance, probability_matrices = nested_optimal_transport_loop(
        tree1_root, tree2_root, max_depth, method, lambda_reg
    )
    
    if return_matrix:
        final_prob_matrix = compute_final_probability_matrix(
            probability_matrices, tree1_root, tree2_root, max_depth
        )
        return distance, final_prob_matrix
    else:
        return distance