import numpy as np
from collections import deque
from trees.Tree_Node import TreeNode


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

    Parameters:
      tree_1_root: The root of the first tree.
      tree_2_root: The root of the second tree.
      depth: The depth at which to extract nodes.
      power: The power to which differences are raised.

    Returns:
      A NumPy array with the computed distance matrix.
    """
    # Assume get_nodes_at_depth is defined elsewhere
    nodes_at_depth_tree1 = get_nodes_at_depth(tree_1_root, depth)
    nodes_at_depth_tree2 = get_nodes_at_depth(tree_2_root, depth)

    # Pad all paths to the same length for each tree
    arr1 = pad_paths(nodes_at_depth_tree1)  # shape: (m, L)
    arr2 = pad_paths(nodes_at_depth_tree2)  # shape: (n, L)

    # Use broadcasting to compute absolute differences
    # Expand dimensions so that:
    #   arr1 becomes shape (m, 1, L)
    #   arr2 becomes shape (1, n, L)
    # Their difference results in shape (m, n, L)
    diff = np.abs(arr1[:, None, :] - arr2[None, :, :]) ** power

    # Sum over the last axis (L) to get the final distance matrix of shape (m, n)
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
