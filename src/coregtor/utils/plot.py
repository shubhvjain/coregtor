"""
Plots for all parts of the pipeline 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram, cophenet
from scipy.spatial.distance import squareform
from sklearn.linear_model import QuantileRegressor
from typing import Tuple, List, Optional, Dict
import networkx as nx
import hashlib
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
import matplotlib.pyplot as plt
import networkx as nx
#------------------
# Visualize modules
#------------------


def _ensure_distance_matrix(comparison_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the comparison matrix is a distance matrix.
    
    Converts similarity to distance if needed based on metadata.
    """
    is_distance = comparison_matrix.attrs.get('is_distance', False)
    
    if not is_distance:
        # Convert similarity to distance
        dist_matrix = 1 - comparison_matrix
        dist_matrix.attrs = comparison_matrix.attrs.copy()
        dist_matrix.attrs['is_distance'] = True
        return dist_matrix
    
    return comparison_matrix


def _prepare_for_scipy(distance_matrix: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Prepare distance matrix for scipy hierarchical clustering."""
    mat = distance_matrix.copy()
    np.fill_diagonal(mat.values, 0.0)
    condensed = squareform(mat.values, checks=False)
    labels = mat.index.tolist()
    return condensed, labels


def dendrogram(
    comparison_matrix: pd.DataFrame,
    method: str = "average",
    figsize: Tuple[int, int] = (15, 9),
    leaf_rotation: float = 90,
    leaf_font_size: int = 8,
    color_threshold: float = None,
    show: bool = True
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """Plot hierarchical clustering dendrogram from comparison matrix."""
    dist_matrix = _ensure_distance_matrix(comparison_matrix)
    condensed, labels = _prepare_for_scipy(dist_matrix)
    Z = linkage(condensed, method=method)
    
    fig, ax = plt.subplots(figsize=figsize)
    scipy_dendrogram(
        Z,
        labels=labels,
        leaf_rotation=leaf_rotation,
        leaf_font_size=leaf_font_size,
        color_threshold=color_threshold,
        ax=ax
    )
    
    ax.set_title(f"Hierarchical Clustering Dendrogram ({method} linkage)")
    ax.set_ylabel("Distance")
    ax.set_xlabel("Gene")
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, ax, Z


def cophenetic(
    comparison_matrix: pd.DataFrame,
    methods: List[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    quantile: float = 0.1,
    show_regression: bool = True,
    show: bool = True
) -> Tuple[plt.Figure, np.ndarray, Dict[str, float], Optional[float]]:
    """Plot cophenetic correlation analysis to assess clustering quality."""
    if methods is None:
        methods = ["average", "complete", "ward"]
    
    dist_matrix = _ensure_distance_matrix(comparison_matrix)
    condensed, labels = _prepare_for_scipy(dist_matrix)
    
    ccc_dict = {}
    linkage_matrices = {}
    
    for method in methods:
        Z = linkage(condensed, method=method)
        ccc, coph_dist = cophenet(Z, condensed)
        ccc_dict[method] = ccc
        linkage_matrices[method] = (Z, coph_dist)
    
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    
    suggested_threshold = None
    
    for ax, method in zip(axes, methods):
        Z, coph_dist = linkage_matrices[method]
        ccc = ccc_dict[method]
        
        df = pd.DataFrame({
            "original": condensed,
            "cophenetic": coph_dist
        })
        
        ax.scatter(df["cophenetic"], df["original"], alpha=0.4, s=10)
        ax.set_xlabel("Cophenetic distance")
        ax.set_ylabel("Original distance")
        ax.set_title(f"{method.capitalize()}\nCCC = {ccc:.3f}")
        ax.grid(True, alpha=0.3)
        
        if show_regression and method == methods[0]:
            X = df["cophenetic"].values.reshape(-1, 1)
            y = df["original"].values
            
            model = QuantileRegressor(quantile=quantile, alpha=0).fit(X, y)
            x_sorted = np.sort(X[:, 0])
            y_pred = model.predict(x_sorted.reshape(-1, 1))
            
            ax.plot(x_sorted, y_pred, 'r-', linewidth=2, 
                   label=f'{int(quantile*100)}th percentile')
            
            median_coph = np.median(df["cophenetic"])
            suggested_threshold = model.predict([[median_coph]])[0]
            ax.axvline(median_coph, color='orange', linestyle='--', alpha=0.5)
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    if show:
        plt.show()
    
    return fig, axes, ccc_dict, suggested_threshold


def get_color_from_hash(text):
    """Generate a consistent color from a text hash."""
    hash_obj = hashlib.md5(text.encode())
    hash_hex = hash_obj.hexdigest()
    r = int(hash_hex[0:2], 16) / 255
    g = int(hash_hex[2:4], 16) / 255
    b = int(hash_hex[4:6], 16) / 255
    return (r, g, b)

def preprocess_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Add color and label columns to edges DataFrame."""
    edges = edges.copy()
    
    # Check if edge_type and edge_source columns exist
    has_edge_type = 'edge_type' in edges.columns
    has_edge_source = 'edge_source' in edges.columns
    
    if has_edge_type and has_edge_source:
        edges['label'] = edges['edge_type'] + ' + ' + edges['edge_source']
        edges['color'] = edges.apply(
            lambda row: get_color_from_hash(f"{row['edge_type']}_{row['edge_source']}"), 
            axis=1
        )
    elif has_edge_type:
        edges['label'] = edges['edge_type']
        edges['color'] = edges['edge_type'].apply(get_color_from_hash)
    elif has_edge_source:
        edges['label'] = edges['edge_source']
        edges['color'] = edges['edge_source'].apply(get_color_from_hash)
    else:
        edges['label'] = 'default'
        edges['color'] = [(0, 0, 1)] * len(edges)  # Blue default
    
    return edges

def cluster_target_edges(cluster_nodes: list, target_n: str, edges: pd.DataFrame, 
                        title="Cluster edges", caption=None):
    """
    Plot a cluster graph with cluster nodes distributed in area and target node outside.
    
    Parameters:
    - cluster_nodes: list of nodes in the cluster
    - target_n: the target node (outside the circle)
    - edges: DataFrame with columns ['node1', 'node2', optional: 'edge_type', 'edge_source']
    - title: Title for the plot
    - caption: Optional caption text
    """
    # Preprocess edges to add color and label
    edges = preprocess_edges(edges)
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(cluster_nodes)
    G.add_node(target_n)
    
    # Separate edges
    within_cluster = []
    to_target = []
    
    for _, row in edges.iterrows():
        node1, node2 = row['node1'], row['node2']
        
        if node1 in cluster_nodes and node2 in cluster_nodes:
            within_cluster.append(row)
            G.add_edge(node1, node2)
        elif (node1 in cluster_nodes and node2 == target_n) or \
             (node2 in cluster_nodes and node1 == target_n):
            to_target.append(row)
            G.add_edge(node1, node2)
    
    # Create layout
    n_nodes = len(cluster_nodes)
    pos = {}
    
    if n_nodes <= 25:
        pos = nx.circular_layout(cluster_nodes, scale=1.5)
        target_offset = 3.0
    else:
        nodes_per_ring = 15
        n_rings = int(np.ceil(n_nodes / nodes_per_ring))
        node_idx = 0
        
        for ring in range(n_rings):
            remaining = n_nodes - node_idx
            nodes_in_ring = min(nodes_per_ring, remaining)
            radius = 0.8 + ring * 0.6
            
            for i in range(nodes_in_ring):
                if node_idx < n_nodes:
                    angle = 2 * np.pi * i / nodes_in_ring
                    pos[cluster_nodes[node_idx]] = np.array([radius * np.cos(angle), 
                                                             radius * np.sin(angle)])
                    node_idx += 1
        
        target_offset = 0.8 + (n_rings - 1) * 0.6 + 1.5
    
    pos[target_n] = np.array([target_offset, 0])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=cluster_nodes, 
                          node_color='lightblue', node_size=400, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[target_n], 
                          node_color='orange', node_size=700, ax=ax)
    
    # Draw edges
    for row in within_cluster:
        nx.draw_networkx_edges(G, pos, 
                              edgelist=[(row['node1'], row['node2'])], 
                              edge_color=[row['color']], 
                              width=1.5, alpha=0.6, style='solid', ax=ax)
    
    for row in to_target:
        nx.draw_networkx_edges(G, pos, 
                              edgelist=[(row['node1'], row['node2'])], 
                              edge_color=[row['color']], 
                              width=1.5, alpha=0.7, style='dashed', ax=ax)
    
    # Draw labels
    font_size = 8 if n_nodes > 25 else 9
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight='bold', ax=ax)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    if caption:
        plt.figtext(0.5, 0.02, caption, wrap=True, 
                   horizontalalignment='center', fontsize=10, style='italic')
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
               markersize=10, label='Cluster Nodes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
               markersize=12, label='Target Node'),
    ]
    
    # Get unique edge types for legend
    unique_edges = edges[['label', 'color']].drop_duplicates().sort_values('label')
    if len(unique_edges) > 0 and unique_edges['label'].iloc[0] != 'default':
        legend_elements.append(Line2D([0], [0], color='none', label=''))
        for _, edge_row in unique_edges.iterrows():
            legend_elements.append(
                Line2D([0], [0], color=edge_row['color'], linewidth=2.5, 
                       label=f"  {edge_row['label']}")
            )
        
    ax.legend(handles=legend_elements, loc='best', fontsize=9, framealpha=0.9)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_network(G, title="Cluster", figsize=(6,4), layout='spring', node_color=None):
    """
    Visualize NetworkX graph with nodes colored by cluster_id.
    
    Args:
        G: NetworkX graph
        title: Plot title
        figsize: Figure size
        layout: Layout algorithm ('spring', 'kamada_kawai', 'circular')
        node_color: Dict mapping hex colors to lists of nodes. Example: {"#ff0000": [node1, node2], "#00ff00": [node3]}
    """
    if G.number_of_nodes() == 0:
        print("Graph has no nodes")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Get cluster_id for each node and assign colors
    node_colors = {}
    default_colors = plt.cm.tab20
    
    for node in G.nodes():
        cluster_id = G.nodes[node].get('cluster_id', 0)
        # Use hash for consistent color per cluster
        color_idx = hash(str(cluster_id)) % 20
        node_colors[node] = default_colors(color_idx)
    
    # Override colors with custom node_color mappings if provided
    if node_color:
        for hex_color, nodes in node_color.items():
            for node in nodes:
                if node in node_colors:
                    node_colors[node] = hex_color
                else:
                    print(f"Warning: Node {node} not found in graph")
    
    # Convert node_colors dict to list in node order
    node_color_list = [node_colors[node] for node in G.nodes()]
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color='black',
        alpha=0.3,
        arrows=False,
        arrowsize=10,
        width=2,
        ax=ax
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_color_list,
        node_size=300,
        alpha=0.9,
        ax=ax
    )
    
    # Draw labels for small graphs
    if G.number_of_nodes() <= 50:
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=15)
    ax.axis('off')
    plt.tight_layout()
    
    plt.show()