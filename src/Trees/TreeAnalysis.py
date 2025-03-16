import numpy as np
from trees.Tree_Node import *


def get_sample_paths(tree_root):
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

    # Initialize traversal from the root
    traverse(tree_root, [], 1.0)

    # Convert lists to NumPy arrays
    paths_array = np.array(paths)
    probabilities_array = np.array(probabilities)

    return [paths_array, probabilities_array]


def display_tree_data(paths_weights, tree_name):
    """
    Displays the paths and their associated probabilities for a given tree.

    Parameters:
    - paths_weights (tuple): A tuple containing paths and probabilities.
    - tree_name (str): The name of the tree for display purposes.
    """
    print(f"\n{tree_name} (Path and Weight Format):")
    print("Paths:")
    print(paths_weights[0])
    print("Weights:")
    print(paths_weights[1])


def get_depth(tree_root):
    """
    Calculates the depth (height) of the tree, starting at 0 for the root node.

    Parameters:
    - tree_root (TreeNode): The root node of the tree.

    Returns:
    - int: The depth of the tree.
    """
    if tree_root is None:
        return -1  # Assuming an empty tree has a depth of -1 (convenient for us)

    # If the node has no children, its depth is 0
    if not tree_root.children:
        return 0

    # Since all paths have the same depth, we can check just one path
    first_child, _ = tree_root.children[0]
    return 1 + get_depth(first_child)
