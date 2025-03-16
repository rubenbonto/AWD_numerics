from trees.Tree_Node import TreeNode
from trees.Save_Load_trees import save_tree


def build_tree_1():
    """
    Constructs the first predefined tree structure.
    """
    # Root node
    tree_1_root = TreeNode(10)

    # Level 1
    node_10 = TreeNode(10)
    tree_1_root.add_child(node_10, 1.0)

    # Level 2
    node_12 = TreeNode(12)
    node_8 = TreeNode(8)
    node_10.add_child(node_12, 34 / 100)
    node_10.add_child(node_8, 66 / 100)

    # Level 3
    node_13 = TreeNode(13)
    node_10_level3 = TreeNode(10)
    node_12.add_child(node_13, 54 / 100)
    node_12.add_child(node_10_level3, 46 / 100)

    node_9 = TreeNode(9)
    node_6 = TreeNode(6)
    node_8.add_child(node_9, 24 / 100)
    node_8.add_child(node_6, 76 / 100)

    return tree_1_root


def build_tree_2():
    """
    Constructs the second predefined tree structure.
    """
    # Root node
    tree_2_root = TreeNode(10)

    # Level 1
    node_13 = TreeNode(13)
    node_7 = TreeNode(7)
    tree_2_root.add_child(node_13, 3 / 10)
    tree_2_root.add_child(node_7, 7 / 10)

    # Level 2
    node_14 = TreeNode(14)
    node_11 = TreeNode(11)
    node_13.add_child(node_14, 2 / 10)
    node_13.add_child(node_11, 8 / 10)

    node_8 = TreeNode(8)
    node_5 = TreeNode(5)
    node_7.add_child(node_8, 1 / 10)
    node_7.add_child(node_5, 9 / 10)

    # Level 3
    node_15 = TreeNode(15)
    node_14_self = TreeNode(14)
    node_13_level3 = TreeNode(13)
    node_14.add_child(node_15, 5 / 10)
    node_14.add_child(node_14_self, 1 / 10)
    node_14.add_child(node_13_level3, 4 / 10)

    node_12 = TreeNode(12)
    node_10_level3 = TreeNode(10)
    node_11.add_child(node_12, 6 / 10)
    node_11.add_child(node_10_level3, 4 / 10)

    node_9_level3 = TreeNode(9)
    node_7_level3 = TreeNode(7)
    node_8.add_child(node_9_level3, 4 / 10)
    node_8.add_child(node_7_level3, 6 / 10)

    node_6 = TreeNode(6)
    node_4 = TreeNode(4)
    node_5.add_child(node_6, 3 / 10)
    node_5.add_child(node_4, 7 / 10)

    return tree_2_root


# Build trees
tree_1_root = build_tree_1()
tree_2_root = build_tree_2()

# Save trees
save_tree(
    tree_1_root,
    "/Users/rubenbontorno/Documents/Master_Thesis/Code/AWD_numerics/src/Trees/Data_trees_exemple",
    "tree_1.pkl",
)
save_tree(
    tree_2_root,
    "/Users/rubenbontorno/Documents/Master_Thesis/Code/AWD_numerics/src/Trees/Data_trees_exemple",
    "tree_2.pkl",
)
