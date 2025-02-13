import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from Tree_Node import TreeNode

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