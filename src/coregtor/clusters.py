"""
Co-regulator identification through clustering of gene context similarity matrices.

Provides a unified interface for multiple clustering methods with consistent output format.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster, inconsistent

from scipy.spatial.distance import squareform
from coregtor.utils.error import CoRegTorError
import secrets

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


def _compute_auto_threshold(sim_matrix, method, linkage1):
    """Compute automatic distance threshold using specified method.
    
    Methods:
    - 'inconsistency': Statistical jumps in dendrogram [μ(coeffs) + 0.5σ]
    - 'elbow': Largest gap in last 10 merge heights 
    - 'percentile': 75th percentile of all pairwise distances (fastest)
    
    Args:
      sim_matrix: Similarity matrix DataFrame
      method: {'inconsistency', 'elbow', 'percentile'}
      linkage1: Linkage method ('average', 'complete', etc.)
    
    Returns:
      float: Optimal distance_threshold
    """

    
    dist_matrix = _ensure_distance_matrix(sim_matrix)
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method=linkage1)
    
    if method == 'inconsistency':
        """Cut where dendrogram has statistical jumps.
        inconsistent(Z) → (n-1,4) array, col3=inconsistency coeffs (>1=odd merge)
        threshold = mean(coeffs) + 0.5*std(coeffs)
        """
        incons = inconsistent(Z)
        coeffs = incons[:, 3]  # Inconsistency coefficients
        return np.mean(coeffs) + 0.5 * np.std(coeffs)
        
    elif method == 'elbow':
        """Find largest gap in merge distances (elbow plot for dendrogram).
        Z[:,-10:,2] = last 10 merge heights [0.12,0.23,...,0.89←jump!,1.02]
        diffs = np.diff(heights), threshold = heights[max_diff_idx]
        """
        merge_heights = Z[-min(10, len(Z)), 2]
        if len(merge_heights) < 2:
            return np.median(squareform(dist_matrix))
        diffs = np.diff(merge_heights)
        elbow_idx = np.argmax(diffs)
        return merge_heights[elbow_idx]
        
    elif method == 'percentile':
        """75th percentile of all pairwise distances (fast, robust).
        squareform(dist) → [0.12,0.45,0.89,...], cuts "large" distances
        """
        return np.percentile(squareform(dist_matrix), 75)
        
    else:
        raise ValueError(f"Unknown auto_threshold method: {method}. "
                        f"Use 'inconsistency', 'elbow', or 'percentile'")


def _format_clusters_df(
    sim_matrix: pd.DataFrame, 
    cluster_labels: np.ndarray, 
    target_gene: str, 
    min_module_size: int = 2
) -> pd.DataFrame:
  """
  Convert raw clustering labels into gene-level  DataFrame with quality scores.
    
  Example:
    target_gene | cluster_id | cluster_uid | gene  | score
    ESR1        | 1          | a1b2c3d4e5f6| ESR1  | 0.85
    ESR1        | 1          | a1b2c3d4e5f6| FOXA1 | 0.72
    ESR1        | 2          | x9y8z7w6v5u4| FOXA2 | 0.61
    
  Args:
    sim_matrix: Square gene-gene SIMILARITY matrix DataFrame
    cluster_labels: Array matching sim_matrix.index [0,0,1,2,1,...]
    target_gene: Name of target gene (added as column)
    min_module_size: Drop clusters with fewer genes
  
  Returns:
    Sorted DataFrame ready for downstream analysis/visualization
  """
  
  # 1. Ensure we have validated distance matrix
  dist_matrix = _ensure_distance_matrix(sim_matrix)
  gene_names = sim_matrix.index.astype(str)
  
  # 2. Calculate silhouette scores PER GENE (higher = better cluster fit)
  sil_scores = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')
  
  # 3. Build DataFrame + filter small clusters
  gene_clusters = pd.DataFrame({
      'gene': gene_names,
      'cluster_id': cluster_labels,
      'score': sil_scores
  })
  
  # Remove singletons (biological noise)
  cluster_sizes = gene_clusters['cluster_id'].value_counts()
  valid_clusters = cluster_sizes[cluster_sizes >= min_module_size].index
  gene_clusters = gene_clusters[gene_clusters['cluster_id'].isin(valid_clusters)]
  
  # 4. Clean cluster numbering (1,2,3... regardless of algorithm)
  unique_clusters = sorted(gene_clusters['cluster_id'].unique())
  mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters, 1)}
  gene_clusters['cluster_id'] = gene_clusters['cluster_id'].map(mapping)
  
  # 5. Unique cluster identifiers (for database, cross-analysis tracking)
  cluster_uids = {cid: secrets.token_hex(6) for cid in gene_clusters['cluster_id'].unique()}
  gene_clusters['cluster_uid'] = gene_clusters['cluster_id'].map(cluster_uids)
  
  # 6. Final DataFrame (target_gene column + sort order)
  df = gene_clusters[['cluster_id', 'cluster_uid', 'gene', 'score']]
  df.insert(0, 'target_gene', target_gene)
  
  # Sort: cluster_id ASC, score DESC  best genes first within each cluster
  return df.sort_values(['cluster_id', 'score'], ascending=[True, False]).reset_index(drop=True)


def get_best_cluster(clusters_df: pd.DataFrame) -> Dict[str, Any]:
  """Returns the best cluster based on the computed score.
  
  Selects cluster with highest average silhouette score.
  
  Args:
    clusters_df: DataFrame from _format_clusters_df with columns 
      ['target_gene', 'cluster_id', 'cluster_uid', 'gene', 'score']
  
  Returns:
    Dict: {'genes': list[str], 'score': float, 'cluster_uid': str, 'n_genes': int}
  """
  if clusters_df.empty:
    return {'genes': [], 'score': 0.0, 'cluster_uid': '', 'n_genes': 0}
  
  # Compute average score per cluster
  cluster_avg_scores = clusters_df.groupby('cluster_id')['score'].mean()
  best_cluster_id = cluster_avg_scores.idxmax()
  
  # Extract best cluster data
  best_rows = clusters_df[clusters_df['cluster_id'] == best_cluster_id]
  genes = best_rows['gene'].tolist()
  
  return {
      'genes': genes,
      'score': float(cluster_avg_scores[best_cluster_id]),
      'cluster_uid': best_rows['cluster_uid'].iloc[0],
      'n_genes': len(genes)
  }




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
    sim_matrix: Gene similarity matrix DataFrame (square, rows/cols = gene names).
    target_gene: Target gene name (must be in sim_matrix.index).
    method_options: Dict with:
      - 'n_clusters': int (exact number of clusters) OR 
      - 'distance_threshold': float (cut dendrogram at distance)
      - 'linkage': str ['average', 'complete', 'ward', 'single'] (default: 'average')
      - 'min_module_size': int (default: 2),
      - 'auto_threshold' : to automatically determine the distance threshold value, [inconsistency,elbow,percentile]
    **kwargs: Ignored.

  Raises:
    CoRegTorError: If neither n_clusters nor distance_threshold specified,
      or both specified together.

  Returns:
    Dict:
      - 'model': Fitted AgglomerativeClustering instance
      - 'clusters_df': pd.DataFrame with columns 
          ['target_gene', 'cluster_id', 'cluster_uid', 'gene', 'score']
          (gene-level silhouette scores, sorted by cluster_id then score DESC)
      - 'best_cluster': Dict {'genes': list[str], 'score': float, 
          'cluster_uid': str, 'n_genes': int} (highest avg silhouette cluster)
      - 'methodology': str ("method=hierarchical_clustering&n_clusters=5&...")
  """
  n_clusters = method_options.get('n_clusters',None)
  distance_threshold = method_options.get('distance_threshold',None)
  linkage = method_options.get('linkage', 'average')
  min_module_size = method_options.get('min_module_size', 2)
  
  auto_threshold = method_options.get('auto_threshold',None)
  if auto_threshold:
    if n_clusters or distance_threshold:
      raise CoRegTorError("Cannot mix auto_threshold with n_clusters/distance_threshold")
    distance_threshold = _compute_auto_threshold(sim_matrix, auto_threshold, linkage)

  if n_clusters is None and distance_threshold is None:
    raise CoRegTorError("Must specify 'n_clusters' or 'distance_threshold'")
  if n_clusters is not None and distance_threshold is not None:
    raise CoRegTorError("Cannot specify both  n_clusters and  distance_threshold")
  
  dist_matrix = _ensure_distance_matrix(sim_matrix)
  params = {'metric': 'precomputed', 'linkage': linkage}
  params['n_clusters'] = n_clusters
  params['distance_threshold'] = distance_threshold
  
  
  model = AgglomerativeClustering(**params)
  labels = model.fit_predict(dist_matrix)
  
  clusters_df = _format_clusters_df(sim_matrix, labels, target_gene, min_module_size)

  best_cluster = get_best_cluster(clusters_df)

  params.update({
       "auto_threshold" : auto_threshold,
      'min_module_size': min_module_size,
      'scoring_index': 'silhouette',
  })
  
  methodology = format_methodology("hierarchical_clustering", params)
  
  return {
      'model': model,
      'clusters_df': clusters_df,
      'best_cluster': best_cluster,
      'methodology': methodology
  }


# def validation_index(
#     sim_matrix: pd.DataFrame, 
#     target_gene: str, 
#     method_options: Dict[str, Any],
#     **kwargs
# ) -> Dict[str, Any]:
#   """Automatic cluster number selection using validation indices.

#   Args:
#     sim_matrix: Gene similarity matrix.
#     target_gene: Target gene name.
#     method_options: Parameters dict with 'index', 'k_range'.
#     **kwargs: Additional parameters.

#   Returns:
#     Dict with model, clusters_df, best, best_df, methodology, validation_scores.
#   """
#   index = method_options.get('index', 'silhouette')
#   linkage1 = method_options.get('linkage', 'average')
#   k_range = method_options.get('k_range', (2, 20))
#   min_module_size = method_options.get('min_module_size', 2)
  
#   dist_matrix = _ensure_distance_matrix(sim_matrix)
#   n_samples = dist_matrix.shape[0]
  
#   condensed_dist = squareform(dist_matrix)
#   Z = linkage(condensed_dist, method=linkage1)
  
#   scores = {}
#   k_max = min(k_range[1] + 1, n_samples)
#   for k in range(max(2, k_range[0]), k_max):
#     labels = fcluster(Z, k, criterion='maxclust')
    
#     try:
#       if index == 'silhouette':
#         score = silhouette_score(dist_matrix, labels, metric='precomputed')
#       elif index == 'davies_bouldin':
#         score = davies_bouldin_score(dist_matrix, labels, metric='precomputed')
#       elif index == 'calinski_harabasz':
#         score = calinski_harabasz_score(dist_matrix, labels)
#       else:
#         raise ValueError(f"Unknown index: {index}")
#     except:
#       score = -1.0 if index == 'silhouette' else float('inf')
    
#     scores[k] = score
  
#   if not scores:
#     raise CoRegTorError(f"No valid k values in range {k_range}")
  
#   if index == 'silhouette':
#     best_k = max(scores.keys(), key=lambda k: scores[k])
#     if scores[best_k] <= 0:
#       best_k = None
#   elif index == 'davies_bouldin':
#     best_k = min(scores.keys(), key=lambda k: scores[k])
#   elif index == 'calinski_harabasz':
#     best_k = max(scores.keys(), key=lambda k: scores[k])
  
#   if best_k is None:
#     raise CoRegTorError(f"No valid clustering found using {index}")
  
#   labels = fcluster(Z, best_k, criterion='maxclust')
#   clusters_df = _format_clusters_df(sim_matrix, labels, target_gene, min_module_size)
  
#   best_cluster = get_best_cluster(clusters_df)

  
#   method_options_best = method_options.copy()
#   method_options_best['best_k'] = best_k
  
#   best_methodology = format_methodology("validation_index", method_options_best)
  
#   best = {
#       'items': best_row['gene_cluster'].split(';'),
#       'score': best_score,
#       'methodology': best_methodology
#   }
  
#   best_df = pd.DataFrame([{
#       'target_gene': target_gene,
#       'cluster_id': best_row['cluster_id'],
#       'items': best_row['gene_cluster'],
#       'score': best_score,
#       'methodology': best_methodology
#   }])
  
#   methodology = format_methodology("validation_index", method_options_best)
  
#   return {
#       'model': None,
#       'clusters_df': clusters_df,
#       'best': best,
#       'best_df': best_df,
#       'methodology': methodology,
#       'validation_scores': scores
#   }


METHOD_REGISTRY = {
    'hierarchical_clustering': hierarchical_clustering,
    # 'validation_index': validation_index
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
