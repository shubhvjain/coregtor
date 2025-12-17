"""
Clustering and visualization for co-regulator identification.

This module provides functions for interactive exploration of pathway similarity
and identification of putative co-regulatory modules.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from typing import Tuple, List, Optional, Dict, Callable


# ------------------
# Helper Functions
# ------------------

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


# ------------------
# Identify co-regulators - Individual methods
# ------------------

def _identify_coregulators_hierarchical(
    comparison_matrix: pd.DataFrame,
    target_gene: str,
    n_clusters: int = None,
    distance_threshold: float = None,
    linkage: str = "average",
    min_module_size: int = 2,
    **kwargs
) -> Tuple[pd.DataFrame, AgglomerativeClustering]:
    """
    Hierarchical clustering for co-regulator identification.

    Args:
        comparison_matrix: Output from compare_context()
        target_gene: Name of the target gene
        n_clusters: Number of clusters (mutually exclusive with distance_threshold)
        distance_threshold: Distance threshold (mutually exclusive with n_clusters)
        linkage: Linkage method ('average', 'complete', 'ward', 'single')
        min_module_size: Minimum genes per module

    Returns:
        Tuple of (modules_df, model)
    """
    if n_clusters is None and distance_threshold is None:
        raise ValueError(
            "Must specify either 'n_clusters' or 'distance_threshold'")
    if n_clusters is not None and distance_threshold is not None:
        raise ValueError(
            "Cannot specify both 'n_clusters' and 'distance_threshold'")

    dist_matrix = _ensure_distance_matrix(comparison_matrix)

    if len(dist_matrix) <= 1:
        gene_clusters = pd.DataFrame({
            'gene': dist_matrix.index.astype(str),
            'cluster_id': 0
        })
        
        modules_df = pd.DataFrame({
            'target_gene': [target_gene],
            'gene_cluster': [','.join(gene_clusters['gene'])],
            'n_genes': [len(gene_clusters)],
            'cluster_id': [0]
        })
        
        # Store metadata
        modules_df.attrs['method'] = 'hierarchical'
        modules_df.attrs['linkage'] = linkage
        modules_df.attrs['n_modules'] = 1
        modules_df.attrs['target_gene'] = target_gene
        modules_df.attrs['single_gene'] = True
        
        # Return dummy model for API consistency
        model = AgglomerativeClustering(n_clusters=1, metric="precomputed")
        return modules_df, model

    clustering_params = {
        "metric": "precomputed",
        "linkage": linkage
    }

    if n_clusters is not None:
        clustering_params["n_clusters"] = n_clusters
    else:
        clustering_params["n_clusters"] = None
        clustering_params["distance_threshold"] = distance_threshold

    model = AgglomerativeClustering(**clustering_params)
    labels = model.fit_predict(dist_matrix.values)

    gene_clusters = pd.DataFrame({
        'gene': dist_matrix.index.astype(str),
        'cluster_id': labels
    })

    # Filter by minimum module size
    cluster_sizes = gene_clusters['cluster_id'].value_counts()
    valid_clusters = cluster_sizes[cluster_sizes >= min_module_size].index
    gene_clusters = gene_clusters[gene_clusters['cluster_id'].isin(
        valid_clusters)].copy()

    # Renumber clusters
    unique_clusters = sorted(gene_clusters['cluster_id'].unique())
    cluster_mapping = {old: new for new, old in enumerate(unique_clusters)}
    gene_clusters['cluster_id'] = gene_clusters['cluster_id'].map(
        cluster_mapping)

    # Aggregate to cluster-level
    modules_df = gene_clusters.groupby('cluster_id').agg(
        gene_cluster=('gene', lambda x: ','.join(sorted(x))),
        n_genes=('gene', 'count')
    ).reset_index()

    modules_df.insert(0, 'target_gene', target_gene)
    modules_df = modules_df[['target_gene',
                             'gene_cluster', 'n_genes', 'cluster_id']]
    modules_df = modules_df.sort_values(
        ['n_genes', 'cluster_id'], ascending=[False, True])
    modules_df = modules_df.reset_index(drop=True)

    # Store metadata
    modules_df.attrs['method'] = 'hierarchical'
    modules_df.attrs['linkage'] = linkage
    modules_df.attrs['n_modules'] = len(modules_df)
    modules_df.attrs['target_gene'] = target_gene

    return modules_df, model


# Dispatcher dictionary
CLUSTERING_METHODS: Dict[str, Callable] = {
    "hierarchical": _identify_coregulators_hierarchical,
}


def list_clustering_methods() -> List[str]:
    """List all available clustering methods."""
    return sorted(CLUSTERING_METHODS.keys())


# ------------------
# Main interface
# ------------------

def identify_coregulators(
    comparison_matrix: pd.DataFrame,
    target_gene: str,
    method: str = "hierarchical",
    **kwargs
) -> Tuple[pd.DataFrame, any]:
    """
    Identify putative co-regulatory modules from context similarity matrix of root genes.

    Clusters genes with similar  contexts to generate hypotheses about genes that may function as co-regulators.

    Args:
        comparison_matrix: Output from compare_context()
        target_gene: Name of the target gene being regulated
        method: Clustering method ('hierarchical')
        **kwargs: Method-specific parameters.For method='hierarchical':
                - n_clusters (int): Number of clusters
                - distance_threshold (float): Distance threshold
                - linkage (str): Linkage method ('average', 'complete', 'ward', 'single')
                - min_module_size (int): Minimum genes per module

    Returns:
        Tuple of (modules_df, model)
        - modules_df: DataFrame with columns ['target_gene', 'gene_cluster', 'n_genes', 'cluster_id']
        - model: Fitted clustering model

    Raises:
        ValueError: If method is unknown
        NotImplementedError: If method is not yet implemented

    """
    if method not in CLUSTERING_METHODS:
        raise ValueError(
            f"Unknown clustering method: {method}. "
            f"Available methods: {list_clustering_methods()}"
        )

    clusterer = CLUSTERING_METHODS[method]
    return clusterer(comparison_matrix, target_gene, **kwargs)
