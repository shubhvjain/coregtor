"""
Plots for all parts of the pipeline 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram, cophenet
from scipy.spatial.distance import squareform
from sklearn.linear_model import QuantileRegressor
from typing import Tuple, List, Optional, Dict,Any
import networkx as nx
import hashlib
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def gene_expression_heatmap(df, gene_list, 
                                 show_sample_labels=True,
                                 label_threshold=25,
                                 figsize=None,
                                 cmap='RdBu_r',
                                 center=0,
                                 title='Gene Expression Heatmap',
                                 standardize=False):
    """
    Plots a heatmap of gene expression values for selected genes.

    Given a gene expression DataFrame and a list of genes, this function displays a heatmap of expression values for those genes across all samples. It automatically suppresses y-axis (sample) labels if the gene list is large, to keep the visualization clear.

    Args:
        df (pandas.DataFrame): Gene expression matrix (rows are samples, columns are genes).
        gene_list (list): List of gene names to include on the heatmap. Must contain at least one gene.
        show_sample_labels (bool, optional): Whether to display sample (y-axis) labels. Defaults to True.
        label_threshold (int, optional): Hide sample labels if the gene list length exceeds this value. Defaults to 50.
        figsize (tuple, optional): Figure size in inches (width, height). Defaults to an automatic value based on gene/sample count.
        cmap (str, optional): Colormap for the heatmap. Defaults to 'RdBu_r'.
        center (float, optional): Value at which to center the colormap. Defaults to 0.
        title (str, optional): Title for the plot. Defaults to 'Gene Expression Heatmap'.
        standardize (bool, optional): If True, standardizes expression values (z-score per gene). Defaults to False.

    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects of the generated plot.

    Raises:
        ValueError: If gene_list is empty or none of the genes are found in the DataFrame.

    """
    
    # Validate input
    if not gene_list or len(gene_list) < 1:
        raise ValueError("gene_list must contain at least 1 gene")
    
    # Check if all genes exist in dataframe
    missing_genes = [g for g in gene_list if g not in df.columns]
    if missing_genes:
        print(f"Warning: The following genes are not in the dataframe: {missing_genes}")
        gene_list = [g for g in gene_list if g in df.columns]
        if len(gene_list) == 0:
            raise ValueError("None of the specified genes exist in the dataframe")
    
    # Subset the data
    data = df[gene_list].copy()
    
    # Standardize if requested
    if standardize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data = pd.DataFrame(
            scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        width = max(10, len(gene_list) * 0.3)
        height = max(8, len(data) * 0.1)
        figsize = (min(width, 20), min(height, 15))
    
    # Determine whether to show row (sample) labels
    show_yticklabels = show_sample_labels and len(gene_list) <= label_threshold
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        data,
        cmap=cmap,
        center=center,
        xticklabels=True,  # Always show gene names
        yticklabels=show_yticklabels,  # Conditionally show sample labels
        cbar_kws={'label': 'Expression Level'},
        ax=ax
    )
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)
    if show_yticklabels:
        plt.yticks(rotation=0)
    
    plt.title(title)
    plt.xlabel('Genes')
    plt.ylabel('Samples')
    plt.tight_layout()
    
    return fig, ax


def gene_cluster(pca_df, x_col, y_col, z_col=None, cluster_col=None, 
                   figsize=(10, 8), title='Gene Expression PCA', 
                   alpha=0.6, s=50):
    """Plot gene expression PCA data in 2D or 3D with optional cluster coloring.
    
    Creates a scatter plot of genes in reduced dimensional space. Points can be 
    colored uniformly or by cluster membership. Supports both 2D and 3D visualization.
    
    Args:
        pca_df (pd.DataFrame): DataFrame containing gene data with principal 
            components and optional cluster assignments. Should have columns 
            matching x_col, y_col, and optionally z_col and cluster_col.
        x_col (str): Column name for x-axis values (e.g., 'pc1').
        y_col (str): Column name for y-axis values (e.g., 'pc2').
        z_col (str, optional): Column name for z-axis values (e.g., 'pc3'). 
            If provided, creates a 3D plot. Defaults to None for 2D plot.
        cluster_col (str, optional): Column name for cluster assignments. If None, 
            all points will be plotted in the same color. Defaults to None.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (10, 8).
        title (str, optional): Plot title. Defaults to 'Gene Expression PCA'.
        alpha (float, optional): Point transparency (0-1). Defaults to 0.6.
        s (int, optional): Point size. Defaults to 50.
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects.
    
    Example:
        >>> pca_result = compute_gene_pca(expr_data, n_components=3)
        >>> 
        >>> # 2D plot without clustering
        >>> fig, ax = plot_gene_data(pca_result, 'pc1', 'pc2')
        >>> plt.show()
        >>> 
        >>> # 3D plot without clustering
        >>> fig, ax = plot_gene_data(pca_result, 'pc1', 'pc2', z_col='pc3')
        >>> plt.show()
        >>> 
        >>> # 3D plot with cluster coloring
        >>> pca_result['cluster'] = kmeans.fit_predict(pca_result[['pc1', 'pc2', 'pc3']])
        >>> fig, ax = plot_gene_data(pca_result, 'pc1', 'pc2', z_col='pc3', 
        ...                          cluster_col='cluster')
        >>> plt.show()
    """
    # Determine if 3D plot is needed
    is_3d = z_col is not None
    
    # Create figure and axes
    fig = plt.figure(figsize=figsize)
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    if cluster_col is None:
        # Plot all points in same color
        if is_3d:
            ax.scatter(pca_df[x_col], pca_df[y_col], pca_df[z_col],
                      alpha=alpha, s=s, c='steelblue', edgecolors='k', linewidth=0.5)
        else:
            ax.scatter(pca_df[x_col], pca_df[y_col], 
                      alpha=alpha, s=s, c='steelblue', edgecolors='k', linewidth=0.5)
    else:
        # Plot points colored by cluster
        unique_clusters = pca_df[cluster_col].unique()
        n_clusters = len(unique_clusters)
        
        # Use colormap for clusters
        cmap = plt.cm.get_cmap('tab10', n_clusters)
        
        for i, cluster in enumerate(sorted(unique_clusters)):
            cluster_mask = pca_df[cluster_col] == cluster
            
            if is_3d:
                ax.scatter(pca_df.loc[cluster_mask, x_col], 
                          pca_df.loc[cluster_mask, y_col],
                          pca_df.loc[cluster_mask, z_col],
                          alpha=alpha, s=s, c=[cmap(i)], 
                          label=f'Cluster {cluster}',
                          edgecolors='k', linewidth=0.5)
            else:
                ax.scatter(pca_df.loc[cluster_mask, x_col], 
                          pca_df.loc[cluster_mask, y_col],
                          alpha=alpha, s=s, c=[cmap(i)], 
                          label=f'Cluster {cluster}',
                          edgecolors='k', linewidth=0.5)
        
        ax.legend(loc='best', frameon=True, fancybox=True)
    
    # Set labels
    ax.set_xlabel(x_col.upper(), fontsize=12)
    ax.set_ylabel(y_col.upper(), fontsize=12)
    if is_3d:
        ax.set_zlabel(z_col.upper(), fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid (2D only, 3D has default grid)
    if not is_3d:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return fig, ax



def similarity_matrix_2d_embedding(similarity_matrix: np.ndarray, 
                                       config: Dict[str, Any] = None, 
                                       target_names: List[str] = None) -> plt.Figure:
    """
    Creates 2D embedding (MDS) of a single similarity matrix to visualize target clustering.
    
    Args:
        similarity_matrix: np.ndarray (n_targets, n_targets) similarity/distance matrix
        config: Plotting options (colors, size, method)
        target_names: Optional list of target names for labels
    
    Returns:
        matplotlib Figure with 2D scatter plot
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS
    import numpy as np
    
    # Validate input
    if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("Expected square similarity matrix")
    
    # Default config
    plot_config = config or {}
    method = plot_config.get("method", "mds")
    n_components = plot_config.get("n_components", 2)
    
    # Convert to distance matrix (smaller = more similar)
    distance_matrix = 1 - similarity_matrix  # Assumes similarity in [0,1]
    
    # Create 2D embedding
    if method == "mds":
        embedding = MDS(n_components=n_components, dissimilarity="precomputed", 
                       random_state=42, max_iter=300)
        coords = embedding.fit_transform(distance_matrix)
    else:
        raise ValueError(f"Method {method} not supported. Use 'mds'")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Main scatter plot
    scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                        c=similarity_matrix.max(axis=1),  # Color by avg similarity to others
                        s=plot_config.get("point_size", 200),
                        cmap='viridis', alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Average Similarity to Other Targets", fontsize=12)
    
    # Labels (target names or indices)
    if target_names:
        for i, name in enumerate(target_names[:20]):  # Limit for readability
            ax.annotate(name[:8], (coords[i, 0], coords[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
    else:
        for i in range(min(20, len(coords))):
            ax.annotate(str(i), (coords[i, 0], coords[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Styling
    ax.set_xlabel("MDS Dimension 1", fontsize=12)
    ax.set_ylabel("MDS Dimension 2", fontsize=12)
    ax.set_title("Target Similarity 2D Embedding\n(Closer points = more similar contexts)", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def add_caption(fig: plt.Figure, target: str, created_on: str = None) -> plt.Figure:
    """
    Adds standardized caption to bottom of figure with extra spacing.
    """
    import matplotlib.pyplot as plt
    from datetime import datetime
    import pytz
    
    # Auto-generate timestamp if not provided
    if created_on is None:
        tz = pytz.timezone('Europe/Berlin')  # CET/CEST
        created_on = datetime.now(tz).strftime("%Y-%m-%d %H:%M CET")
    
    caption = f"Results for target {target}. Created on: {created_on}"
    
    # Add caption with EXTRA BOTTOM SPACING (0.01 → 0.005)
    fig.text(0.5, 0.005, caption, 
             ha='center', va='bottom', fontsize=10, 
             style='italic', color='gray', alpha=0.8,
             wrap=True, transform=fig.transFigure)
    
    # Ensure white background
    fig.patch.set_facecolor('white')
    
    return fig

def similarity_matrix_heatmap(similarity_matrix: np.ndarray, 
                                  config: Dict[str, Any] = None, 
                                  target_names: List[str] = None) -> plt.Figure:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    plot_config = config or {}
    percentile = plot_config.get("percentile", 75)  # NEW: Configurable percentile
    n = similarity_matrix.shape[0]
    
    # Convert to numpy if pandas
    if hasattr(similarity_matrix, 'values'):
        matrix_data = similarity_matrix.values
        if target_names is None and hasattr(similarity_matrix, 'index'):
            target_names = similarity_matrix.index.tolist()
    else:
        matrix_data = similarity_matrix
    
    # NEW: 75th percentile selection by average similarity
    avg_sim = matrix_data.mean(axis=1)
    threshold = np.percentile(avg_sim, percentile)
    high_sim_indices = np.where(avg_sim >= threshold)[0]
    
    matrix_subset = matrix_data[high_sim_indices][:, high_sim_indices]
    labels_subset = [target_names[i] for i in high_sim_indices] if target_names else high_sim_indices
    n_subset = len(labels_subset)
    
    title_suffix = f"(Top {percentile}th percentile similarity, n={n_subset}/{n})"
    
    # Create figure
    figsize = (max(8, min(15, n_subset/6)), max(8, min(15, n_subset/6)))
    fig, ax = plt.subplots(figsize=figsize)
    
    # Heatmap
    mask = np.triu(np.ones_like(matrix_subset, dtype=bool))
    sns.heatmap(matrix_subset, mask=mask, ax=ax, cmap='viridis', 
                cbar_kws={'label': 'Similarity', 'shrink': 0.8},
                square=True, linewidths=0)
    
    # Labels with dynamic font size
    if labels_subset and len(labels_subset) > 0:
        short_labels = [str(l)[:8] for l in labels_subset]  # Slightly longer labels
        label_fontsize = 7 if n_subset > 25 else 8
        
        ax.set_xticks(np.arange(n_subset))
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=label_fontsize)
        ax.set_yticks(np.arange(n_subset))
        ax.set_yticklabels(short_labels, rotation=0, fontsize=label_fontsize)
    
    ax.set_title(f"Similarity Matrix Heatmap {title_suffix}", 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Targets (High Similarity)", fontsize=12)
    ax.set_ylabel("Targets (High Similarity)", fontsize=12)
    
    plt.tight_layout()
    return fig


def dendrogram1(similarity_matrix: np.ndarray, 
                   config: Dict[str, Any] = None, 
                   target_names: List[str] = None) -> plt.Figure:
    """
    Creates hierarchical clustering dendrogram from similarity matrix.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    import numpy as np
    import pandas as pd
    
    plot_config = config or {}
    method = plot_config.get('method', 'average')
    figsize = plot_config.get('figsize', (15, 9))
    leaf_rotation = plot_config.get('leaf_rotation', 90)
    leaf_font_size = plot_config.get('leaf_font_size', 8)
    color_threshold = plot_config.get('color_threshold')
    
    # Handle pandas DataFrame
    if hasattr(similarity_matrix, 'values'):
        matrix_data = similarity_matrix.values
        if target_names is None and hasattr(similarity_matrix, 'index'):
            target_names = similarity_matrix.index.tolist()
    else:
        matrix_data = similarity_matrix
    
    # Convert to distance matrix
    dist_matrix = 1 - matrix_data
    
    # Ensure symmetric condensed distance matrix for scipy
    n = dist_matrix.shape[0]
    condensed_dist = dist_matrix[np.triu_indices(n, k=1)]
    
    # Create linkage
    Z = linkage(condensed_dist, method=method)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot dendrogram
    dendrogram(Z, ax=ax,
               labels=target_names,
               leaf_rotation=leaf_rotation,
               leaf_font_size=leaf_font_size,
               color_threshold=color_threshold,
               above_threshold_color='gray')
    
    ax.set_title(f"Hierarchical Clustering Dendrogram ({method} linkage)", 
                fontsize=14, fontweight='bold')
    ax.set_ylabel("Distance", fontsize=12)
    ax.set_xlabel("Genes", fontsize=12)
    
    plt.tight_layout()
    return fig
