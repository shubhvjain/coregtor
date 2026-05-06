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

def hierarchical_clustering(distance,target_gene, options=None):
    """Main entry point for Agglomerative Hierarchical Clustering."""
    labels = list(distance.index)
    n_items = len(labels)

    if n_items == 0: return set(), "empty"
    if n_items <= 3: return {tuple(labels)}, "small-set"

    if options is None:
        options = {"linkage_method": "average", "auto_threshold": "inconsistency"}

    condensed_dist = squareform(distance, checks=False)
    z = linkage(condensed_dist, method=options.get('linkage_method', 'average'))

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
            'method': 'edge_threshold_value' | 'edge_threshold_percentile' | 'edge_knn',
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
    method = options.get('edge_creation_method', 'threshold_value')
    val = options.get('edge_creation_value', 0.5)

    if method == 'threshold_value':
        # Only scan the upper triangle (k=1) to find unique pairs
        rows, cols = np.where(np.triu(sim_mat, k=1) >= val)
        for r, c in zip(rows, cols):
            edge_dict[(r, c)] = sim_mat[r, c]
                
    elif method == 'threshold_percentile':
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
                    edge_dict[(u, v)] = max(edge_dict.get((u, v), 0), sim_mat[i, neighbor])

    # 4. ASSEMBLY: Create the igraph object
    edges = list(edge_dict.keys())
    weights = [edge_dict[e] for e in edges]
    
    # directed=False ensures the graph is mathematically undirected
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.vs['name'] = node_labels
    g.es['weight'] = weights
    
    return g

# --- EXAMPLE USAGE ---
# g = distance_to_network(my_df, {'normalize': True, 'method': 'edge_knn', 'value': 5})
# clusters = g.community_leiden(weights='weight', objective_function='modularity')


def community_detection_leiden(dist_matrix, target_gene,options=None):
    """
    Performs graph-based community detection on a gene similarity matrix.
    
    This function converts a dense similarity matrix into a sparse graph and 
    optimizes modularity (or CPM) using the Leiden algorithm.
    """
    # Default options for gene clustering
    if options is None:
        options = {
            "edge_creation_method":"threshold_value",
            "edge_creation_value":0.5,
            "resolution": 1.0,                        
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
    cluster_set = {tuple(sorted(g.vs[cluster]['name'])) for cluster in partition}
    
    # Generate a metadata string for the result DataFrame
    note = f"leiden"
    return cluster_set, note



METHOD_REGISTRY = {
    'hierarchical': hierarchical_clustering,
    'community_detection':community_detection_leiden
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
        scores = silhouette_samples(reordered_matrix, labels, metric="precomputed")
    except Exception:
        # Catch-all for any other weird sklearn math edge cases
        scores = np.zeros(n_samples)
    
    return pd.DataFrame({
        "gene": ordered_genes, 
        "cluster": labels, 
        "score": np.round(scores, 5)
    })


def generate_cluster_results(distance_matrix, target_gene, clusters, note,cluster_note=""):
    # clusters = [list(c) for c in clusters if len(c) >= 2]
    if not clusters:
        return pd.DataFrame()

    sil_df = silhouette_score(distance_matrix, target_gene, clusters)
    has_scores = sil_df["score"].notna().any()
    
    # ch_score, db_score = np.nan, np.nan
    #if len(clusters) >= 2:
        #dist = sim_to_dist(distance_matrix)
        #features = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(dist)
        #ch_score = calinski_harabasz_score(features, labels)
        #db_score = davies_bouldin_score(features, labels)

    rows = []
    
    for idx, cluster in enumerate(clusters):
        gene_scores = sil_df[sil_df["cluster"] == idx]
        if has_scores:
            gene_scores = gene_scores.sort_values("score", ascending=False)

        ordered_genes = gene_scores["gene"].tolist()
        rows.append({
            "uid": secrets.token_hex(6),
            "target": target_gene,
            "sources": ";".join(str(g) for g in ordered_genes),
            "n_source": len(ordered_genes),
           
            "silhouette_score": round(gene_scores["score"].mean(), 5) if has_scores else np.nan,
            "sil_gene_scores": ";".join(f"{s:.5f}" for s in gene_scores["score"].tolist()) if has_scores else "",
            "sil_score_optimal": False,
            "note": note,
            "cluster_note":cluster_note
        })

    df = pd.DataFrame(rows)
    if has_scores:
        df.at[df["silhouette_score"].idxmax(), "sil_score_optimal"] = True

    # include only clusters with 2 or more sources and positive module sill score
    df = df[(df['silhouette_score']  > 0 ) & (df["n_source"] >= 2)].reset_index(drop=True)

    return df


def identify_coregulators(
    distance_matrix,
    target_gene,
    method  = "hierarchical",
    options = {},
    note = ""
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

    method_func = METHOD_REGISTRY[method]
    clusters,cluster_note = method_func(distance_matrix, target_gene, options)
    results = generate_cluster_results(distance_matrix,target_gene,clusters,note,cluster_note)
    return results

