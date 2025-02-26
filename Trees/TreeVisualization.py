import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdicts
import numpy as np
from scipy.stats import gaussian_kde



def visualize_tree(tree_root, title="Tree Visualization"):
    """
    Visualizes the stochastic tree using NetworkX and Matplotlib.

    Parameters:
    - tree_root (TreeNode): Root node of the tree to visualize.
    - title (str): Title of the plot.
    """
    G = nx.DiGraph()
    node_labels = {}
    pos = {}
    id_counter = defaultdict(int)
    depths = {}

    def assign_depths(node, current_depth=0):
        depths[node] = current_depth
        for child, _ in node.children:
            assign_depths(child, current_depth + 1)

    assign_depths(tree_root)
    
    name_to_node = {}

    def add_edges(node, parent_name=None, x=0, y=0, dx=1.0):
        nonlocal G, pos, node_labels, id_counter

        if parent_name is None:
            node_name = f"root_{node.value:.2f}"
        else:
            id_counter[parent_name] += 1
            node_name = f"{parent_name}_{id_counter[parent_name]}_{node.value:.2f}"

        pos[node_name] = (x, y)
        node_labels[node_name] = f"{node.value:.2f}"
        G.add_node(node_name)

        if parent_name is not None:
            parent_node = name_to_node[parent_name]
            for child, prob in parent_node.children:
                if abs(child.value - node.value) < 1e-6:
                    G.add_edge(parent_name, node_name, weight=prob)
                    break

        name_to_node[node_name] = node

        if node.children:
            num_children = len(node.children)
            width = dx / num_children
            for i, (child, prob) in enumerate(node.children):
                child_x = x - dx/2 + i * width + width / 2
                child_y = y - 1
                add_edges(child, node_name, child_x, child_y, dx / 2)
    
    add_edges(tree_root)

    node_depths = {node_name: depths.get(name_to_node[node_name], 0) for node_name in G.nodes()}
    cmap = plt.get_cmap('viridis')
    max_depth = max(node_depths.values(), default=1)
    node_colors = [cmap(depth / max_depth) for depth in node_depths.values()]

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos, labels=node_labels, with_labels=True, node_size=800, node_color=node_colors, 
            font_size=8, font_weight="bold", arrows=True, arrowstyle='-|>', arrowsize=10, ax=ax)
    
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_depth))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label('Depth')

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def find_node_by_value(node, value):
    """
    Helper function to find the first node with the given value.
    """
    if abs(node.value - value) < 1e-6:
        return node
    for child, _ in node.children:
        result = find_node_by_value(child, value)
        if result:
            return result
    return None








#### for big trees!!

def gather_paths(root):
    """
    Gathers all root-to-leaf paths in the tree along with the product
    of transition probabilities for each path.

    Returns:
        A list of tuples (values_list, path_probability), where:
         - values_list is the list of node.values from root to leaf
         - path_probability is the product of the probabilities along that path
    """
    stack = [(root, [root.value], 1.0)]
    paths = []

    while stack:
        node, path_values, path_prob = stack.pop()
        # If the node has no children, it's a leaf -> store the path
        if not node.children:
            paths.append((path_values, path_prob))
        else:
            for (child, p) in node.children:
                stack.append((child,
                              path_values + [child.value],
                              path_prob * p))
    return paths




def visualize_big_tree(tree_root, fig_size=(10,6), title="Stochastic Tree"):
    """
    Plots a 'fan' or 'scenario tree' style visualization:
      - On the left: lines of state vs. stage/time for each root-to-leaf path.
      - On the right: a distribution (histogram or kernel density) of the leaf states.

    Parameters:
    - tree_root (TreeNode): The root of the tree to visualize.
    - fig_size (tuple): Figure size in inches, e.g. (12, 6).
    - title (str): Title of the entire figure.
    """

    # 1) Gather all paths (root -> leaf)
    paths = gather_paths(tree_root)
    # Each element of `paths` is (list_of_values, probability_of_path)

    # 2) Determine the maximum depth (for x-axis on the left plot)
    max_depth = max(len(p[0]) for p in paths)  # largest path length

    # 3) Prepare the figure with two subplots:
    #    - left: scenario paths
    #    - right: final distribution
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=fig_size, gridspec_kw={'width_ratios': [3, 1]}, sharey=True
    )

    fig.suptitle(title, fontsize=14)

    # 4) Plot each path in the left subplot
    for path_values, path_prob in paths:
        # x-coords = stage indices (0 to len(path_values)-1)
        x_coords = np.arange(len(path_values))
        ax_left.plot(x_coords, path_values, linewidth=1.0, alpha=0.7)

    ax_left.set_xlabel("Stage / Time")
    ax_left.set_ylabel("State")
    ax_left.set_title("States over time")

    # 5) Collect final states + their probabilities
    leaf_values = [pv[-1] for (pv, _) in paths]
    leaf_probs = [pr for (_, pr) in paths]

    # 6a) Option A: Weighted histogram on the right subplot
    #    (orientation='horizontal' so that y is the state axis)
    # ax_right.hist(leaf_values, bins=30, orientation='horizontal',
    #               weights=leaf_probs, alpha=0.6, color='blue')
    # ax_right.set_title("Distribution of Final States")

    # 6b) Option B: Kernel density estimate of final states (weighted)
    y_min = min(leaf_values)
    y_max = max(leaf_values)
    # Some padding in y-limits
    y_pad = 0.05 * (y_max - y_min if y_max != y_min else 1)
    y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 500)

    # Use gaussian_kde with weights:
    kde = gaussian_kde(leaf_values, weights=leaf_probs)
    pdf = kde(y_grid)  # evaluate density on the grid

    ax_right.plot(pdf, y_grid, color='blue')
    ax_right.fill_betweenx(y_grid, 0, pdf, color='blue', alpha=0.2)
    ax_right.set_title("Final State Distribution")
    ax_right.set_xlabel("Density")
    # Mirror the y-limits of the left axis
    ax_right.set_xlim(left=0, right=1.1 * pdf.max())

    # 7) Clean up / show
    plt.tight_layout()
    plt.show()