import numpy as np


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child_node, probability):
        self.children.append((child_node, probability))


# A helper to convert a value to a hashable type.
def to_hashable(x):
    if isinstance(x, np.ndarray):
        return tuple(x.tolist())
    return x


def multidim_pad_paths(paths, pad_value=None):
    max_length = max(len(path) for path in paths)
    if pad_value is None:
        first_val = paths[0][0]
        if isinstance(first_val, (np.ndarray, list)):
            first_arr = np.array(first_val)
            pad_value = np.zeros(first_arr.shape)
        else:
            pad_value = 0
    padded = np.array([path + [pad_value] * (max_length - len(path)) for path in paths])
    return padded


def multidim_build_tree_from_paths(sample_paths, weights):
    """
    Builds a weighted tree from sample paths. Each path is assumed to be a list/array
    where each element can be a scalar (d=1) or a vector (d>1).
    """
    start_value = sample_paths[0][0]
    start_key = to_hashable(start_value)
    for path in sample_paths:
        if to_hashable(path[0]) != start_key:
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
            key = to_hashable(value)
            if key not in current["children"]:
                current["children"][key] = {
                    "node": {"value": value, "children": {}},
                    "weight": 0.0,
                }
            current["children"][key]["weight"] += path_weight
            current = current["children"][key]["node"]

    def convert_tree_dict(node_dict):
        node = TreeNode(node_dict["value"])
        children = node_dict["children"]
        if children:
            total = sum(child_info["weight"] for child_info in children.values())
            for child_key, child_info in children.items():
                child_node = convert_tree_dict(child_info["node"])
                probability = child_info["weight"] / total if total > 0 else 0
                node.add_child(child_node, probability)
        return node

    return convert_tree_dict(tree_dict)


def multidim_find_node_by_path(node, path):
    current_node = node
    for value in path[1:]:
        key = to_hashable(value)
        matching_children = [
            child
            for child, _ in current_node.children
            if to_hashable(child.value) == key
        ]
        if not matching_children:
            print(
                f"Invalid path: No child with value {value} under node with value {current_node.value}."
            )
            return None
        if len(matching_children) > 1:
            print(
                f"Warning: Multiple children with value {value} found under node with value {current_node.value}."
            )
        current_node = matching_children[0]
    return current_node


def multidim_compute_distance_matrix_at_depth(tree_1_root, tree_2_root, depth, power):
    nodes_at_depth_tree1 = multidim_get_nodes_at_depth(tree_1_root, depth)
    nodes_at_depth_tree2 = multidim_get_nodes_at_depth(tree_2_root, depth)
    arr1 = multidim_pad_paths(nodes_at_depth_tree1)
    arr2 = multidim_pad_paths(nodes_at_depth_tree2)

    if arr1.ndim == 2:
        diff = np.abs(arr1[:, None, :] - arr2[None, :, :]) ** power
        distance_matrix = diff.sum(axis=2)
    elif arr1.ndim == 3:
        diff = arr1[:, None, :, :] - arr2[None, :, :, :]
        diff_norm = np.linalg.norm(diff, axis=3) ** power
        distance_matrix = diff_norm.sum(axis=2)
    return distance_matrix


def value_equal(v1, v2):
    return to_hashable(v1) == to_hashable(v2)


def multidim_get_nodes_at_depth(tree_root, depth):
    """Collect all nodes at a specific depth from the root.
    Returns each path as a list of node values (which can be scalars or vectors)."""
    nodes = []

    def traverse(node, path, current_depth):
        if current_depth == depth:
            nodes.append(path + [node.value])
            return
        for child, _ in node.children:
            traverse(child, path + [node.value], current_depth + 1)

    traverse(tree_root, [], 0)
    return nodes


def multidim_get_paths_to_leaves(tree_root, max_depth):
    """Generate all paths from the root to each leaf node up to a specified depth.
    Each path is returned as a list of node values."""
    paths = []

    def traverse(node, path, depth):
        if depth == max_depth or not node.children:
            paths.append(path + [node.value])
            return
        for child, _ in node.children:
            traverse(child, path + [node.value], depth + 1)

    traverse(tree_root, [], 0)
    return paths


def multidim_get_node_from_path(tree_root, path):
    """Given a root node and a path (list of values), return the node at the end of the path.
    Uses value_equal() to compare node values, ensuring consistency for both d=1 and d>1.
    """
    current_node = tree_root
    for value in path[1:]:
        current_node = next(
            child
            for child, _ in current_node.children
            if value_equal(child.value, value)
        )
    return current_node


def multidim_compute_marginal_probabilities_for_subset(
    node1_path, node2_path, tree_1_root, tree_2_root
):
    """Compute marginal probabilities for the direct successors of node1 and node2.
    It extracts the nodes at the end of the paths and then builds their successor paths.
    """
    node1 = multidim_get_node_from_path(tree_1_root, node1_path)
    node2 = multidim_get_node_from_path(tree_2_root, node2_path)

    successors_node1 = [
        (node1_path + [child.value], prob) for child, prob in node1.children
    ]
    successors_node2 = [
        (node2_path + [child.value], prob) for child, prob in node2.children
    ]

    pi_ratios = [prob for _, prob in successors_node1]
    pi_tilde_ratios = [prob for _, prob in successors_node2]

    return pi_ratios, pi_tilde_ratios


def multidim_get_sample_paths(tree_root):
    """
    Extracts all possible paths from the root to the leaves of the tree
    along with their associated probabilities.

    Parameters:
    - tree_root (TreeNode): The root node of the tree.

    Returns:
    - tuple: A tuple containing:
        - paths_array (np.ndarray): 2D array where each row is a path.
        - probabilities_array (np.ndarray): 1D array of path probabilities.
    """
    paths = []
    probabilities = []

    def traverse(node, current_path, current_prob):
        """
        Recursively traverses the tree to collect paths and probabilities.

        Parameters:
        - node (TreeNode): The current node being traversed.
        - current_path (list): The path taken to reach the current node.
        - current_prob (float): The cumulative probability of the current path.
        """
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


def multidim_get_depth(tree_root):
    """
    Calculates the depth (height) of the tree, starting at 0 for the root node.

    Parameters:
    - tree_root (TreeNode): The root node of the tree.

    Returns:
    - int: The depth of the tree.
    """
    if tree_root is None:
        return -1

    if not tree_root.children:
        return 0

    first_child, _ = tree_root.children[0]
    return 1 + multidim_get_depth(first_child)
