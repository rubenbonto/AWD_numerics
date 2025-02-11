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