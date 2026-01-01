"""
Co-regulator identification through clustering of gene context similarity matrices.

Provides a unified interface for multiple clustering methods with consistent output format.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from coregtor.utils.error import CoRegTorError


def _ensure_distance_matrix(sim_matrix: pd.DataFrame) -> np.ndarray:
  """Convert similarity matrix to validated distance matrix.

  Args:
    sim_matrix: Input similarity matrix DataFrame (square).

  Returns:
    Validated distance matrix as numpy array.
  """
  is_distance = sim_matrix.attrs.get('is_distance', False)
  if not is_distance:
    dist = 1.0 - sim_matrix.values
  else:
    dist = sim_matrix.values.copy()
  np.fill_diagonal(dist, 0.0)
  return dist.astype(np.float64)


def _format_clusters_df(
    sim_matrix: pd.DataFrame, 
    cluster_labels: np.ndarray, 
    target_gene: str, 
    min_module_size: int = 2
) -> pd.DataFrame:
  """Format cluster labels into standardized clusters DataFrame.

  Args:
    sim_matrix: Input similarity matrix.
    cluster_labels: Cluster assignments array.
    target_gene: Target gene identifier.
    min_module_size: Minimum cluster size threshold.

  Returns:
    Formatted DataFrame with columns: target_gene, cluster_id, gene_cluster, n_genes.
  """
  gene_clusters = pd.DataFrame({
      'gene': sim_matrix.index.astype(str),
      'cluster_id': cluster_labels
  })
  
  cluster_sizes = gene_clusters['cluster_id'].value_counts()
  valid_clusters = cluster_sizes[cluster_sizes >= min_module_size].index
  gene_clusters = gene_clusters[gene_clusters['cluster_id'].isin(valid_clusters)]
  
  unique_clusters = sorted(gene_clusters['cluster_id'].unique())
  mapping = {old: new for new, old in enumerate(unique_clusters, 1)}
  gene_clusters['cluster_id'] = gene_clusters['cluster_id'].map(mapping)
  
  clusters_df = gene_clusters.groupby('cluster_id').agg(
      gene_cluster=('gene', lambda x: ';'.join(sorted(x))),
      n_genes=('gene', 'count')
  ).reset_index()
  
  clusters_df.insert(0, 'target_gene', target_gene)
  clusters_df = clusters_df[['target_gene', 'cluster_id', 'gene_cluster', 'n_genes']]
  return clusters_df.sort_values(['n_genes', 'cluster_id'], ascending=[False, True]).reset_index(drop=True)


def format_methodology(method: str, options: Dict[str, Any]) -> str:
  """Format method name and options into standardized methodology string.

  Args:
    method: Clustering method name.
    options: Method parameter dictionary.

  Returns:
    URL-encoded methodology string.
  """
  parts = [f"method={method}"]
  for key, value in options.items():
    parts.append(f"{key}={value}")
  return "&".join(parts)


def hierarchical_clustering(
    sim_matrix: pd.DataFrame, 
    target_gene: str, 
    method_options: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
  """Perform hierarchical clustering with fixed parameters.

  Args:
    sim_matrix: Gene similarity matrix.
    target_gene: Target gene name.
    method_options: Parameters dict containing 'n_clusters' or 'distance_threshold'.
    **kwargs: Additional parameters.

  Returns:
    Dict with model, clusters_df, best, best_df, methodology.
  """
  n_clusters = method_options.get('n_clusters')
  distance_threshold = method_options.get('distance_threshold')
  linkage = method_options.get('linkage', 'average')
  min_module_size = method_options.get('min_module_size', 2)
  
  if n_clusters is None and distance_threshold is None:
    raise CoRegTorError("Must specify 'n_clusters' or 'distance_threshold'")
  if n_clusters is not None and distance_threshold is not None:
    raise CoRegTorError("Cannot specify both 'n_clusters' and 'distance_threshold'")
  
  dist_matrix = _ensure_distance_matrix(sim_matrix)
  
  params = {'metric': 'precomputed', 'linkage': linkage}
  if n_clusters is not None:
    params['n_clusters'] = n_clusters
  else:
    params['distance_threshold'] = distance_threshold
  
  model = AgglomerativeClustering(**params)
  labels = model.fit_predict(dist_matrix)
  
  clusters_df = _format_clusters_df(sim_matrix, labels, target_gene, min_module_size)
  
  methodology = format_methodology("hierarchical_clustering", method_options)
  
  return {
      'model': model,
      'clusters_df': clusters_df,
      'best': None,
      'best_df': None,
      'methodology': methodology
  }


def validation_index(
    sim_matrix: pd.DataFrame, 
    target_gene: str, 
    method_options: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
  """Automatic cluster number selection using validation indices.

  Args:
    sim_matrix: Gene similarity matrix.
    target_gene: Target gene name.
    method_options: Parameters dict with 'index', 'k_range'.
    **kwargs: Additional parameters.

  Returns:
    Dict with model, clusters_df, best, best_df, methodology, validation_scores.
  """
  index = method_options.get('index', 'silhouette')
  linkage1 = method_options.get('linkage', 'average')
  k_range = method_options.get('k_range', (2, 20))
  min_module_size = method_options.get('min_module_size', 2)
  
  dist_matrix = _ensure_distance_matrix(sim_matrix)
  n_samples = dist_matrix.shape[0]
  
  condensed_dist = squareform(dist_matrix)
  Z = linkage(condensed_dist, method=linkage1)
  
  scores = {}
  k_max = min(k_range[1] + 1, n_samples)
  for k in range(max(2, k_range[0]), k_max):
    labels = fcluster(Z, k, criterion='maxclust')
    
    try:
      if index == 'silhouette':
        score = silhouette_score(dist_matrix, labels, metric='precomputed')
      elif index == 'davies_bouldin':
        score = davies_bouldin_score(dist_matrix, labels, metric='precomputed')
      elif index == 'calinski_harabasz':
        score = calinski_harabasz_score(dist_matrix, labels)
      else:
        raise ValueError(f"Unknown index: {index}")
    except:
      score = -1.0 if index == 'silhouette' else float('inf')
    
    scores[k] = score
  
  if not scores:
    raise CoRegTorError(f"No valid k values in range {k_range}")
  
  if index == 'silhouette':
    best_k = max(scores.keys(), key=lambda k: scores[k])
    if scores[best_k] <= 0:
      best_k = None
  elif index == 'davies_bouldin':
    best_k = min(scores.keys(), key=lambda k: scores[k])
  elif index == 'calinski_harabasz':
    best_k = max(scores.keys(), key=lambda k: scores[k])
  
  if best_k is None:
    raise CoRegTorError(f"No valid clustering found using {index}")
  
  labels = fcluster(Z, best_k, criterion='maxclust')
  clusters_df = _format_clusters_df(sim_matrix, labels, target_gene, min_module_size)
  
  best_row = clusters_df.iloc[0]
  best_score = scores[best_k]
  
  method_options_best = method_options.copy()
  method_options_best['best_k'] = best_k
  
  best_methodology = format_methodology("validation_index", method_options_best)
  
  best = {
      'items': best_row['gene_cluster'].split(';'),
      'score': best_score,
      'methodology': best_methodology
  }
  
  best_df = pd.DataFrame([{
      'target_gene': target_gene,
      'cluster_id': best_row['cluster_id'],
      'items': best_row['gene_cluster'],
      'score': best_score,
      'methodology': best_methodology
  }])
  
  methodology = format_methodology("validation_index", method_options_best)
  
  return {
      'model': None,
      'clusters_df': clusters_df,
      'best': best,
      'best_df': best_df,
      'methodology': methodology,
      'validation_scores': scores
  }


METHOD_REGISTRY = {
    'hierarchical_clustering': hierarchical_clustering,
    'validation_index': validation_index
}


def identify_coregulators(
    sim_matrix: pd.DataFrame,
    target_gene: str,
    method: str,
    method_options: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
  """Identify co-regulatory modules from gene similarity matrix.

  Args:
    sim_matrix: Similarity matrix DataFrame from context comparison.
    target_gene: Target gene identifier.
    method: Clustering method name. (hierarchical_clustering,validation_index)
    method_options: Method-specific parameters dictionary.

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
    raise CoRegTorError(f"Unknown method '{method}'. Available: {available}")
  
  method_func = METHOD_REGISTRY[method]
  return method_func(sim_matrix, target_gene, method_options, **kwargs)
