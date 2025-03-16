from trees.Tree_Node import TreeNode


def build_tree_from_paths(sample_paths, weights):
    """
    Builds a weighted tree from sample paths.

    Parameters:
      - sample_paths (list of lists): Each inner list represents a path (e.g. [10, 13, 14, 15]).
      - weights (list of floats): The weight corresponding to each sample path.

    Returns:
      - TreeNode: The root of the constructed tree.

    Raises:
      - ValueError: If not all sample paths have the same starting value,
                    or if the sum of weights is not equal to 1.
    """

    # Check that all sample paths start with the same value.
    start_value = sample_paths[0][0]
    for path in sample_paths:
        if path[0] != start_value:
            raise ValueError(
                "All sample paths must have the same value at time step 0."
            )

    # Check that the sum of weights equals 1.
    total_weight = sum(weights)
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(
            "The sum of weights must equal 1. Got sum(weights) = {}".format(
                total_weight
            )
        )

    # Build a nested dictionary that accumulates the weights for transitions.
    # Each node is represented as:
    #   { 'value': <node_value>, 'children': { child_value: { 'node': <child_node_dict>, 'weight': accumulated_weight } } }
    tree_dict = {"value": start_value, "children": {}}

    # Loop over each sample path along with its weight.
    for path, path_weight in zip(sample_paths, weights):
        current = tree_dict
        # Process each transition from the root (index 0) to the end.
        for value in path[1:]:
            if value not in current["children"]:
                # Create a new child node in the dictionary structure.
                current["children"][value] = {
                    "node": {"value": value, "children": {}},
                    "weight": 0.0,
                }
            # Accumulate the weight for this transition.
            current["children"][value]["weight"] += path_weight
            # Move to the child node.
            current = current["children"][value]["node"]

    # Recursively convert the nested dictionary into a TreeNode structure.
    def convert_tree_dict(node_dict):
        node = TreeNode(node_dict["value"])
        children = node_dict["children"]
        if children:
            # Total weight for the current node's children.
            total = sum(child_info["weight"] for child_info in children.values())
            for child_val, child_info in children.items():
                child_node = convert_tree_dict(child_info["node"])
                # Normalize the transition probability.
                probability = child_info["weight"] / total if total > 0 else 0
                node.add_child(child_node, probability)
        return node

    return convert_tree_dict(tree_dict)
