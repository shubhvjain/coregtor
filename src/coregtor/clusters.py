"""
Co-regulator identification through clustering of gene context similarity matrices.

Provides a unified interface for multiple clustering methods with consistent output format.
"""
import secrets
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.cluster.hierarchy import linkage, fcluster, inconsistent
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_samples, calinski_harabasz_score, davies_bouldin_score
import igraph as ig
from sklearn.cluster import HDBSCAN
from coregtor.utils.error import CoRegTorError

# --- UTILITIES ---


def sim_to_dist(sim):
    """Convert similarity matrix [0,1] to distance matrix [1,0]."""
    arr = np.array(sim, dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    dist = 1.0 - arr
    np.fill_diagonal(dist, 0)
    return dist


def to_clusters(membership, labels):
    """Map cluster IDs to a set of gene-label tuples."""
    clusters = defaultdict(list)
    for idx, comm_id in enumerate(membership):
        clusters[comm_id].append(labels[idx])
    return set(tuple(sorted(c)) for c in clusters.values())

# --- HIERARCHIAL CLUSTERING ---


def auto_threshold(method, z, labels):
    """Heuristic methods to find optimal cut-off for hierarchical clustering."""
    if method == "inconsistency":
        R = inconsistent(z)
        inc_values = R[:, 3]
        threshold = float(np.mean(inc_values) + np.std(inc_values))
        flat_labels = fcluster(z, t=threshold, criterion='inconsistent', R=R)
        note = f"inco-t-{round(threshold, 4)}"

    elif method == "elbow":
        merge_distances = z[:, 2]
        acceleration = np.diff(merge_distances, 2)
        idx = np.argmax(acceleration) + 2
        cut = merge_distances[idx]
        flat_labels = fcluster(z, t=cut, criterion='distance')
        note = f"elbow-t-{round(cut, 4)}"

    else:
        raise ValueError(f"Unknown auto_threshold method: {method}")

    return to_clusters(flat_labels, labels), note


def hierarchical_clustering(distance, target_gene, options=None):
    """Main entry point for Agglomerative Hierarchical Clustering."""
    labels = list(distance.index)
    n_items = len(labels)

    if n_items == 0:
        return set(), "empty"
    if n_items <= 3:
        return {tuple(labels)}, "small-set"

    if options is None:
        options = {"linkage_method": "average",
                   "auto_threshold": "inconsistency"}

    condensed_dist = squareform(distance, checks=False)
    z = linkage(condensed_dist, method=options.get(
        'linkage_method', 'average'))

    # Logic branching for different thresholding strategies
    if 'n_cluster_size' in options:
        n = options['n_cluster_size']
        thresholds = np.sort(np.unique(z[:, 2]))
        clusters = {tuple(labels)}
        note = "ncs-all-in-one"
        for t in thresholds:
            flat_labels = fcluster(z, t=t, criterion='distance')
            if any(len(c) >= n for c in to_clusters(flat_labels, labels)):
                clusters = to_clusters(flat_labels, labels)
                note = f"ncs-first-t-{t:.4f}"
                break
    elif 'n_clusters' in options:
        k = max(1, min(options['n_clusters'], n_items))
        clusters = to_clusters(fcluster(z, t=k, criterion='maxclust'), labels)
        note = f"custom-nc-{k}"
    elif 'auto_threshold' in options:
        clusters, note = auto_threshold(options['auto_threshold'], z, labels)
    elif 'threshold' in options:
        t = options['threshold']
        clusters = to_clusters(fcluster(z, t=t, criterion='distance'), labels)
        note = f"custom-th-{t}"
    else:
        clusters = {tuple(labels)}
        note = "default-all"

    return clusters, f"hc-{note}-nc-{len(clusters)}"


# --- COMMUNITY DETECTION -------


def dist_to_net(dist_df, options={}):
    """
    Converts a distance matrix into an UNDIRECTED igraph network.
    Optimized for community detection.

    Args:
        dist_df (pd.DataFrame): Distance matrix (columns/index = node labels).
        options (dict): {
            'normalize': bool, 
            'method': 'edge_threshold_percentile' | 'edge_knn',
            'value': float/int
        }
    """
    # 1. SETUP: Extract labels and raw data
    node_labels = dist_df.columns.tolist()
    dist_mat = dist_df.to_numpy(dtype=float)
    n = len(node_labels)

    # 2. NORMALIZATION: Map distances to a 0-1 scale if requested
    if options.get('normalize', False):
        d_min, d_max = dist_mat.min(), dist_mat.max()
        # If all values are same, similarity is 0 (prevents division by zero)
        if d_max > d_min:
            # Scale distance then flip: similarity = 1 - normalized_distance
            sim_mat = 1 - ((dist_mat - d_min) / (d_max - d_min))
        else:
            sim_mat = np.zeros_like(dist_mat)
    else:
        # Standard decay: similarity decreases as distance increases
        sim_mat = 1 / (1 + dist_mat)

    # Kill the diagonal: a node should not have an edge to itself
    np.fill_diagonal(sim_mat, 0)

    # 3. EDGE SELECTION: Use a dict to ensure edges are strictly unique and undirected
    # Format: {(smaller_idx, larger_idx): weight}
    edge_dict = {}
    method = options.get('edge_creation_method', 'edge_threshold_percentile')
    val = options.get('edge_creation_value', 0.5)

    if method == 'threshold_percentile':
        # Calculate percentile based only on unique node pairs (upper triangle)
        upper_vals = sim_mat[np.triu_indices(n, k=1)]
        if len(upper_vals) > 0:
            threshold = np.percentile(upper_vals, val * 100)
            rows, cols = np.where(np.triu(sim_mat, k=1) >= threshold)
            for r, c in zip(rows, cols):
                edge_dict[(r, c)] = sim_mat[r, c]

    elif method == 'knn':
        k = int(val)
        for i in range(n):
            # Find indices of the K highest similarity values in this row
            nn_indices = np.argsort(sim_mat[i])[-k:]
            for neighbor in nn_indices:
                if sim_mat[i, neighbor] > 0:
                    # Sort indices (u < v) so A-B and B-A map to the same key
                    u, v = (i, neighbor) if i < neighbor else (neighbor, i)
                    # Keep the highest similarity if both nodes "pick" each other
                    edge_dict[(u, v)] = max(edge_dict.get(
                        (u, v), 0), sim_mat[i, neighbor])
    else:
        raise ValueError(
            f"Unknown automated edge creation method: {method}. Use 'threshold_percentile' or 'knn'.")

    # 4. ASSEMBLY: Create the igraph object
    edges = list(edge_dict.keys())
    weights = [edge_dict[e] for e in edges]

    # directed=False ensures the graph is mathematically undirected
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.vs['name'] = node_labels
    g.es['weight'] = weights

    return g


def community_detection_leiden(dist_matrix, target_gene, options=None):
    """
    Performs graph-based community detection on a gene similarity matrix.

    This function converts a dense similarity matrix into a sparse graph and 
    optimizes modularity (or CPM) using the Leiden algorithm.
    """
    # Default options for gene clustering
    if options is None:
        options = {
            "edge_creation_method": "threshold_percentile",
            "edge_creation_value": 0.5,
            "resolution": 0.5,
            "objective_function": "CPM",
            "n_iterations": 2
        }
    # print(options)
    g = dist_to_net(dist_matrix, options)

    # LEIDEN
    partition = g.community_leiden(
        resolution=options.get("resolution", 1.0),
        objective_function=options.get("objective_function", "CPM"),
        n_iterations=options.get("n_iterations", 2)
    )
    cluster_set = {tuple(sorted(g.vs[cluster]['name']))
                   for cluster in partition}

    # Generate a metadata string for the result DataFrame
    note = f"leiden"
    return cluster_set, note


# ----- HDBSCAN

def hdbscan_clustering(distance_matrix, target_gene, options=None):
    """
    Performs HDBSCAN clustering on a precomputed distance matrix.
    Highly effective for biological data as it identifies 'noise' points.
    """
    labels = list(distance_matrix.index)
    n_items = len(labels)

    if n_items == 0:
        return set(), "empty"
    if n_items <= 3:
        return {tuple(labels)}, "small-set"

    # Default options for biological gene context data
    if options is None:
        options = {
            "min_cluster_size": 3,
            "min_samples": 1,
            "cluster_selection_epsilon": 0.0
        }

    # Extract values for the model
    min_cluster_size = options.get("min_cluster_size", 2)
    min_samples = options.get("min_samples", 1)
    epsilon = options.get("cluster_selection_epsilon", 0.0)

    # Initialize and fit HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric='precomputed',
        cluster_selection_method='eom',
        copy=False
    )
    cluster_labels = clusterer.fit_predict(distance_matrix.to_numpy())

    clusters = defaultdict(list)
    singleton_counter = 0

    for idx, cluster_id in enumerate(cluster_labels):
        if cluster_id != -1:
            # Group valid cluster members normally
            clusters[cluster_id].append(labels[idx])
        else:
            # FIX: Assign each noise point to a unique standalone cluster ID
            # This forces the validator to see them as isolated singletons
            unique_noise_id = f"noise_singleton_{singleton_counter}"
            clusters[unique_noise_id].append(labels[idx])
            singleton_counter += 1

    cluster_set = {tuple(sorted(c)) for c in clusters.values()}

    # Track how many noise points were found in the note for your CSV
    note = f"hdbscan-mcs{min_cluster_size}-noise{singleton_counter}"
    return cluster_set, note


METHOD_REGISTRY = {
    'hierarchical': hierarchical_clustering,
    'community_detection': community_detection_leiden,
    'hdbscan': hdbscan_clustering
}


def get_cluster_method_list():
    return list(METHOD_REGISTRY.keys())

# --- SCORING & RESULTS ---


def silhouette_score(distance_matrix, target_gene, clusters):
    # Flatten clusters to maintain a strict 1-gene-1-label mapping
    ordered_genes = [gene for cluster in clusters for gene in cluster]
    labels = [i for i, cluster in enumerate(clusters) for _ in cluster]

    n_samples = len(ordered_genes)
    n_unique_labels = len(set(labels))

    # EDGE CASE CHECK:
    # 1. Need at least 2 clusters to have a "neighbor"
    # 2. Need at least one cluster with >1 member (n_labels must be < n_samples)
    if n_unique_labels < 2 or n_unique_labels >= n_samples:
        return pd.DataFrame({
            "gene": ordered_genes,
            "cluster": labels,
            "score": 0.0  # Default to 0 so it fails your >0 filter later
        })

    # Ensure the matrix matches our ordered labels
    reordered_matrix = distance_matrix.loc[ordered_genes, ordered_genes]

    try:
        scores = silhouette_samples(
            reordered_matrix, labels, metric="precomputed")
    except Exception:
        # Catch-all for any other weird sklearn math edge cases
        scores = np.zeros(n_samples)

    return pd.DataFrame({
        "gene": ordered_genes,
        "cluster": labels,
        "score": np.round(scores, 5)
    })


def generate_cluster_results(distance_matrix, target_gene, clusters, note, cluster_note=""):
    if not clusters:
        return pd.DataFrame()

    # 1. Calculate silhouette scores
    sil_df = silhouette_score(distance_matrix, target_gene, clusters)

    # Check if we have valid, non-NaN numeric scores
    has_scores = sil_df["score"].notna().any()

    # Dynamically append text to the note if only one cluster was found
    is_single_cluster = len(clusters) == 1
    if is_single_cluster:
        cluster_note = f"{cluster_note} (single-cluster)".strip()

    rows = []
    for idx, cluster in enumerate(clusters):
        gene_scores = sil_df[sil_df["cluster"] == idx]

        if has_scores and not is_single_cluster:
            gene_scores = gene_scores.sort_values("score", ascending=False)
            mean_score = round(gene_scores["score"].mean(), 5)
        else:
            mean_score = np.nan

        ordered_genes = gene_scores["gene"].tolist()
        n_genes = len(ordered_genes)

        # --- Calculate Internal Metrics ---
        if n_genes >= 2:
            # Extract the compact sub-matrix containing only this cluster's genes
            sub_matrix = distance_matrix.loc[ordered_genes, ordered_genes].to_numpy()
            
            # Extract the unique pairs (upper triangle above diagonal, k=1)
            upper_tri_indices = np.triu_indices(n_genes, k=1)
            internal_distances = sub_matrix[upper_tri_indices]
            
            # Density: Average distance between all unique internal pairs
            cluster_density = round(float(np.mean(internal_distances)), 5)
            
            # Diameter: Maximum distance between any internal pair
            cluster_diameter = round(float(np.max(internal_distances)), 5)
        else:
            # Fallback for a 1-gene cluster edge-case (though your filters drop these later)
            cluster_density = 0.0
            cluster_diameter = 0.0

        rows.append({
            "cluster_uid": secrets.token_hex(7),
            "target": target_gene,
            "sources": ";".join(str(g) for g in ordered_genes),
            "n_source": n_genes,
            "silhouette_score": mean_score,
            "cluster_density": cluster_density,
            "cluster_diameter": cluster_diameter,
            "note": note,
            "cluster_note": cluster_note
        })

    df = pd.DataFrame(rows)

    # 2. Adjusted Filter:
    # Keep rows if size >= 2 AND (Score is NaN OR Score is > 0)
    keep_condition = (df["n_source"] >= 2) & (
        df["silhouette_score"].isna() | (df["silhouette_score"] > 0))
    df = df[keep_condition].reset_index(drop=True)

    return df


def normalize_distance_matrix(distance_matrix):
    """
    Min-Max normalizes a distance matrix to a strict [0, 1] scale.
    Preserves the matrix structure, index, columns, and zero-diagonal.
    """
    # Work on a numpy copy to keep it blazing fast
    mat = distance_matrix.to_numpy(dtype=float)

    d_min = mat.min()
    d_max = mat.max()

    if d_max > d_min:
        norm_mat = (mat - d_min) / (d_max - d_min)
    else:
        norm_mat = np.zeros_like(mat)

    # Strictly enforce that a gene's distance to itself is exactly 0
    np.fill_diagonal(norm_mat, 0.0)

    # Reconstruct the DataFrame with original indices/columns
    return pd.DataFrame(norm_mat, index=distance_matrix.index, columns=distance_matrix.columns)


def identify_coregulators(
    distance_matrix,
    target_gene,
    method="hierarchical",
    options={},
    note=""
):
    """Identify co-regulatory modules from gene distance matrix.

    Args:
      distance_matrix: distance_matrix  DataFrame from context comparison.
      target_gene: Target gene identifier.
      method: Clustering method name. (hierarchical)
      options: Method-specific parameters dictionary.

    Returns:
      Dict containing:
        - model: Fitted clustering model or None
        - clusters_df: DataFrame of all clusters
        - best: Best cluster information dict or None
        - best_df: Best cluster as single-row DataFrame or None
        - methodology: Complete parameter string
        - validation_scores: Validation scores dict (validation_index only)
    """
    if method not in METHOD_REGISTRY:
        available = list(METHOD_REGISTRY.keys())
        raise CoRegTorError(
            f"Unknown method '{method}'. Available: {available}")

    should_normalize = options.get("normalize_distance", True)
    if should_normalize:
        distance_matrix = normalize_distance_matrix(distance_matrix)
    # print(distance_matrix)
    method_func = METHOD_REGISTRY[method]
    clusters, cluster_note = method_func(distance_matrix, target_gene, options)
    # print("=====")
    # print(clusters)
    # print("=====")
    results = generate_cluster_results(
        distance_matrix, target_gene, clusters, note, cluster_note)
    # print(results)
    return results
