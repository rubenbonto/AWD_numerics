import numpy as np
from collections import deque
from Trees.Tree_Node import TreeNode

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


def compute_distance_matrix_at_depth(tree_1_root, tree_2_root, depth):
    """Compute the distance matrix up to a specific depth."""
    nodes_at_depth_tree1 = get_nodes_at_depth(tree_1_root, depth)
    nodes_at_depth_tree2 = get_nodes_at_depth(tree_2_root, depth)

    num_nodes_tree1 = len(nodes_at_depth_tree1)
    num_nodes_tree2 = len(nodes_at_depth_tree2)
    distance_matrix = np.zeros((num_nodes_tree1, num_nodes_tree2))

    for i, path1 in enumerate(nodes_at_depth_tree1):
        for j, path2 in enumerate(nodes_at_depth_tree2):
            max_len = max(len(path1), len(path2))
            padded_path1 = path1 + [0] * (max_len - len(path1))
            padded_path2 = path2 + [0] * (max_len - len(path2))
            distance = sum(abs(a - b) for a, b in zip(padded_path1, padded_path2))
            distance_matrix[i, j] = distance

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
        current_node = next(child for child, _ in current_node.children if child.value == value)
    return current_node


def find_node_by_path(node, path):
    """Traverses the tree following the given path and returns the final node."""
    current_node = node
    for value in path[1:]:
        matching_children = [child for child, _ in current_node.children if child.value == value]
        if not matching_children:
            print(f"Invalid path: No child with value {value} under node with value {current_node.value}. Path so far: {path}")
            return None
        if len(matching_children) > 1:
            print(f"Warning: Multiple children with value {value} found under node with value {current_node.value}. Path: {path}")
        current_node = matching_children[0]
    return current_node


def compute_marginal_probabilities_for_subset(node1_path, node2_path, tree_1_root, tree_2_root):
    """Compute marginal probabilities for the direct successors of node1 and node2."""
    node1 = get_node_from_path(tree_1_root, node1_path)
    node2 = get_node_from_path(tree_2_root, node2_path)

    successors_node1 = [(node1_path + [child.value], prob) for child, prob in node1.children]
    successors_node2 = [(node2_path + [child.value], prob) for child, prob in node2.children]

    pi_ratios = [prob for _, prob in successors_node1]
    pi_tilde_ratios = [prob for _, prob in successors_node2]

    return pi_ratios, pi_tilde_ratios
