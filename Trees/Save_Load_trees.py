import pickle
import os

def save_tree(tree_root, path, filename):
    """
    Saves a tree to a file using pickle.

    Parameters:
    - tree_root (TreeNode): The root node of the tree to be saved.
    - path (str): The directory path where the file will be saved.
    - filename (str): The name of the file to save the tree to.
    """
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(tree_root, f)
    print(f"Tree successfully saved to {file_path}")


def load_tree(path, filename):
    """
    Loads a tree from a pickle file.

    Parameters:
    - path (str): The directory path where the file is saved.
    - filename (str): The name of the file to load the tree from.

    Returns:
    - TreeNode: The root node of the loaded tree.
    """
    file_path = os.path.join(path, filename)
    with open(file_path, 'rb') as f:
        tree_root = pickle.load(f)
    print(f"Tree successfully loaded from {file_path}")
    return tree_root
