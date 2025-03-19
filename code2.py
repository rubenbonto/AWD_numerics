# ---------------- Tree and Helper Functions ----------------


class TreeNode:
    def __init__(self, value):
        """
        Initializes a TreeNode with a given value.

        Parameters:
        - value (int or float): The value of the node.
        """
        self.value = value
        self.children = []

    def add_child(self, child_node, probability):
        """
        Adds a child node with an associated transition probability.

        Parameters:
        - child_node (TreeNode): The child node to be added.
        - probability (float): The probability of transitioning to this child.
        """
        # It might be wise to write a check for sum proba = 1.
        self.children.append((child_node, probability))


import numpy as np


def get_sample_paths(tree_root):
    """
    Extracts all possible paths from the root to the leaves of the tree
    along with their associated probabilities.
    """
    paths = []
    probabilities = []

    def traverse(node, current_path, current_prob):
        new_path = current_path + [node.value]
        if not node.children:
            paths.append(new_path)
            probabilities.append(current_prob)
            return
        for child, prob in node.children:
            traverse(child, new_path, current_prob * prob)

    traverse(tree_root, [], 1.0)
    paths_array = np.array(paths)
    probabilities_array = np.array(probabilities)
    return [paths_array, probabilities_array]


def display_tree_data(paths_weights, tree_name):
    print(f"\n{tree_name} (Path and Weight Format):")
    print("Paths:")
    print(paths_weights[0])
    print("Weights:")
    print(paths_weights[1])


def get_depth(tree_root):
    if tree_root is None:
        return -1
    if not tree_root.children:
        return 0
    first_child, _ = tree_root.children[0]
    return 1 + get_depth(first_child)


from collections import deque


def get_nodes_at_depth(tree_root, depth):
    """Collect all nodes at a specific depth from the root."""
    nodes = []

    def traverse(node, path, current_depth):
        if current_depth == depth:
            nodes.append(path + [node.value])
            return
        for child, _ in node.children:
            traverse(child, path + [node.value], current_depth + 1)

    traverse(tree_root, [], 0)
    return nodes


def pad_paths(paths, pad_value=0):
    """Pad all paths to the same length."""
    max_length = max(len(path) for path in paths)
    padded = np.array([path + [pad_value] * (max_length - len(path)) for path in paths])
    return padded


def compute_distance_matrix_at_depth(tree_1_root, tree_2_root, depth, power):
    """
    Compute the distance matrix using vectorized NumPy operations.
    """
    nodes_at_depth_tree1 = get_nodes_at_depth(tree_1_root, depth)
    nodes_at_depth_tree2 = get_nodes_at_depth(tree_2_root, depth)
    arr1 = pad_paths(nodes_at_depth_tree1)  # shape: (m, L)
    arr2 = pad_paths(nodes_at_depth_tree2)  # shape: (n, L)
    diff = np.abs(arr1[:, None, :] - arr2[None, :, :]) ** power
    distance_matrix = diff.sum(axis=2)
    return distance_matrix


def get_paths_to_leaves(tree_root, max_depth):
    """Generate all paths from the root to each leaf node up a specified depth."""
    paths = []

    def traverse(node, path, depth):
        if depth == max_depth or not node.children:
            paths.append(path + [node.value])
            return
        for child, _ in node.children:
            traverse(child, path + [node.value], depth + 1)

    traverse(tree_root, [], 0)
    return paths


def get_node_from_path(tree_root, path):
    """Given a root node and a path (list of values), return the node at the end of the path."""
    current_node = tree_root
    for value in path[1:]:
        current_node = next(
            child for child, _ in current_node.children if child.value == value
        )
    return current_node


def find_node_by_path(node, path):
    """Traverses the tree following the given path and returns the final node."""
    current_node = node
    for value in path[1:]:
        matching_children = [
            child for child, _ in current_node.children if child.value == value
        ]
        if not matching_children:
            print(
                f"Invalid path: No child with value {value} under node with value {current_node.value}. Path so far: {path}"
            )
            return None
        if len(matching_children) > 1:
            print(
                f"Warning: Multiple children with value {value} found under node with value {current_node.value}. Path: {path}"
            )
        current_node = matching_children[0]
    return current_node


def compute_marginal_probabilities_for_subset(
    node1_path, node2_path, tree_1_root, tree_2_root
):
    """Compute marginal probabilities for the direct successors of node1 and node2."""
    node1 = get_node_from_path(tree_1_root, node1_path)
    node2 = get_node_from_path(tree_2_root, node2_path)
    successors_node1 = [
        (node1_path + [child.value], prob) for child, prob in node1.children
    ]
    successors_node2 = [
        (node2_path + [child.value], prob) for child, prob in node2.children
    ]
    pi_ratios = [prob for _, prob in successors_node1]
    pi_tilde_ratios = [prob for _, prob in successors_node2]
    return pi_ratios, pi_tilde_ratios


def build_tree_from_paths(sample_paths, weights):
    """
    Builds a weighted tree from sample paths.
    """
    start_value = sample_paths[0][0]
    for path in sample_paths:
        if path[0] != start_value:
            raise ValueError(
                "All sample paths must have the same value at time step 0."
            )
    total_weight = sum(weights)
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(
            "The sum of weights must equal 1. Got sum(weights) = {}".format(
                total_weight
            )
        )
    tree_dict = {"value": start_value, "children": {}}
    for path, path_weight in zip(sample_paths, weights):
        current = tree_dict
        for value in path[1:]:
            if value not in current["children"]:
                current["children"][value] = {
                    "node": {"value": value, "children": {}},
                    "weight": 0.0,
                }
            current["children"][value]["weight"] += path_weight
            current = current["children"][value]["node"]

    def convert_tree_dict(node_dict):
        node = TreeNode(node_dict["value"])
        children = node_dict["children"]
        if children:
            total = sum(child_info["weight"] for child_info in children.values())
            for child_val, child_info in children.items():
                child_node = convert_tree_dict(child_info["node"])
                probability = child_info["weight"] / total if total > 0 else 0
                node.add_child(child_node, probability)
        return node

    return convert_tree_dict(tree_dict)


def adapted_wasserstein_squared_1d(a, A, b, B):
    """
    Computes the Adapted Wasserstein squared distance AW_2^2 for a one-dimensional process.
    """
    L = np.linalg.cholesky(A)
    M = np.linalg.cholesky(B)
    mean_diff = np.sum((a - b) ** 2)
    trace_sum = np.trace(A) + np.trace(B)
    l1_diag = np.sum(np.abs(np.diag(L.T @ M)))
    return mean_diff + trace_sum - 2 * l1_diag


def uniform_empirical_grid_measure(data, delta_n=None, use_weights=False):
    """
    Computes an empirical measure approximation of sample paths using grid quantization.
    """
    num_path, t = data.shape
    if delta_n is None:
        delta_n = 1 / (num_path ** (1 / t))
    grid_func = lambda x: np.floor(x / delta_n + 0.5) * delta_n
    quantized_data = grid_func(data)
    quantized_data[:, 0] = data[:, 0]
    if not use_weights:
        return quantized_data
    else:
        unique_paths, indices, counts = np.unique(
            quantized_data, axis=0, return_inverse=True, return_counts=True
        )
        weights = counts / num_path
        return unique_paths, weights


# ---------------- Optimal Transport Solver (lp.emd) ----------------

from scipy.optimize import linprog
import ot
from scipy.special import logsumexp


def solver_lp_pot(distance_matrix_subset, pi_ratios, pi_tilde_ratios, reg=1e-2):
    """
    Solve for the optimal transport plan using the POT library's lp.emd solver.
    """
    pi_ratios = np.array(pi_ratios, dtype=np.float64)
    pi_tilde_ratios = np.array(pi_tilde_ratios, dtype=np.float64)
    return ot.lp.emd(pi_ratios, pi_tilde_ratios, distance_matrix_subset)


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
            probability_matrix = solver_lp_pot(sub_matrix, pi_ratios, pi_tilde_ratios)
            cost = np.sum(probability_matrix * sub_matrix)
            step_name = (depth, path1[-1], path2[-1])
            row_result.append((j, cost, step_name, probability_matrix))
        results.append((i, row_result))
    return results


def nested_optimal_transport_loop_parallel(tree1_root, tree2_root, max_depth, power):
    """
    Parallelized version of nested_optimal_transport_loop that splits the outer loop (over paths_tree1)
    into a fixed number of chunks (e.g., one per core). Each process computes its block sequentially.
    """
    probability_matrices = {}
    full_distance_matrix = compute_distance_matrix_at_depth(
        tree1_root, tree2_root, max_depth, power
    )

    for depth in range(max_depth - 1, -1, -1):
        print("Depth:", depth)
        paths_tree1 = get_nodes_at_depth(tree1_root, depth)
        paths_tree2 = get_nodes_at_depth(tree2_root, depth)
        nodes_tree1 = [find_node_by_path(tree1_root, path) for path in paths_tree1]
        nodes_tree2 = [find_node_by_path(tree2_root, path) for path in paths_tree2]
        children_count_tree1 = [len(node.children) for node in nodes_tree1 if node]
        children_count_tree2 = [len(node.children) for node in nodes_tree2 if node]

        updated_distance_matrix = np.zeros((len(paths_tree1), len(paths_tree2)))

        # Partition the outer loop indices into a fixed number of chunks (e.g., 6)
        num_chunks = 12
        indices = list(range(len(paths_tree1)))
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
                    for j, cost, step_name, prob_matrix in row_result:
                        probability_matrices[step_name] = prob_matrix
                        updated_distance_matrix[i, j] = cost

        full_distance_matrix = (
            updated_distance_matrix  # Update for the next depth level
        )

    return full_distance_matrix[0][0], probability_matrices


def compute_nested_distance_parallel(
    tree1_root, tree2_root, max_depth, return_matrix=False, power=1
):
    """
    Computes the nested Wasserstein distance between two trees using only lp.emd.
    """
    distance, probability_matrices = nested_optimal_transport_loop_parallel(
        tree1_root, tree2_root, max_depth, power
    )
    return distance


# ---------------- Main Execution ----------------

if __name__ == "__main__":
    import time

    # Normalization flag
    normalize = False

    # Define factor matrices
    L0 = np.array([[1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 0], [1, 2, 3, 4]])
    A0 = L0 @ L0.T
    L = L0 / np.sqrt(np.trace(A0)) if normalize else L0
    A = L @ L.T

    M0 = np.array([[1, 0, 0, 0], [2, 1, 0, 0], [3, 2, 1, 0], [4, 3, 2, 1]])
    B0 = M0 @ M0.T
    M = M0 / np.sqrt(np.trace(B0)) if normalize else M0
    B = M @ M.T

    # Parameters
    d = 1
    T = 4
    dim = d * T
    n_sample_plot = 2000

    # Generate noise samples
    noise1 = np.random.normal(size=(n_sample_plot, dim))
    noise2 = np.random.normal(size=(n_sample_plot, dim))

    # Apply transformations
    X_increments = (noise1 @ L.T).reshape(n_sample_plot, T, d)
    Y_increments = (noise2 @ M.T).reshape(n_sample_plot, T, d)

    # Prepend zeros along the time axis
    X_paths = np.concatenate([np.zeros((n_sample_plot, 1, d)), X_increments], axis=1)
    Y_paths = np.concatenate([np.zeros((n_sample_plot, 1, d)), Y_increments], axis=1)

    # Adapt empirical measures
    X, Y = np.squeeze(X_paths, axis=-1), np.squeeze(Y_paths, axis=-1)
    adapted_X, adapted_weights_X = uniform_empirical_grid_measure(X, use_weights=True)
    adapted_Y, adapted_weights_Y = uniform_empirical_grid_measure(Y, use_weights=True)

    # Build trees
    adapted_tree_1 = build_tree_from_paths(adapted_X, adapted_weights_X)
    adapted_tree_2 = build_tree_from_paths(adapted_Y, adapted_weights_Y)
    print("Building trees done.")

    # Compute nested distance
    max_depth_val = get_depth(adapted_tree_1)
    start_time = time.time()
    distance_pot = compute_nested_distance_parallel(
        adapted_tree_1, adapted_tree_2, max_depth_val, return_matrix=False, power=2
    )
    elapsed_time_pot = time.time() - start_time

    print("Nested distance single dim:", distance_pot)
    print("Computation time: {:.4f} seconds".format(elapsed_time_pot))
