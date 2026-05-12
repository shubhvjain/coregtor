import pandas as pd
import re
from typing import Union, Dict, Any, Dict, Callable, List
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from coregtor.utils.error import CoRegTorError

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import numpy as np

# ----------------
# Create context
# ----------------


def _create_context_set_tree_paths(tree_paths: pd.DataFrame, **kwargs) -> dict:
    """
    Generate context sets from tree paths DataFrame (excluding root/leaf).

    Args:
        tree_paths: DataFrame with 'source', 'node1', 'node2', ... columns

    Returns:
        dict: {source_gene: [list of sub-paths excluding root and leaf]}
    """
    if "source" not in tree_paths.columns:
        raise CoRegTorError("DataFrame must contain a 'source' column.")

    # Identify and sort node columns (node1, node2, ...)
    node_cols = []
    for c in tree_paths.columns:
        m = re.fullmatch(r"node(\d+)", str(c))
        if m:
            node_cols.append((int(m.group(1)), c))
    if not node_cols:
        raise CoRegTorError("No 'node*' columns found.")
    node_cols = [c for _, c in sorted(node_cols, key=lambda t: t[0])]

    def extract_subpaths(group):
        """Extract sub-paths (excluding root and leaf) from a group of rows."""
        subpaths = []

        # Convert to numpy for faster iteration
        nodes_arr = group[node_cols].to_numpy(dtype=object)

        for row in nodes_arr:
            # Remove NaN values
            path = [n for n in row if pd.notna(n)]

            # Need at least 3 nodes (root, intermediate, leaf)
            if len(path) >= 3:
                sub_path = path[1:-1]  # Exclude first (root) and last (leaf)
                subpaths.append(sub_path)

        # Remove duplicates while preserving order
        unique_subpaths = []
        seen = set()
        for sp in subpaths:
            sp_tuple = tuple(sp)
            if sp_tuple not in seen:
                seen.add(sp_tuple)
                unique_subpaths.append(sp)

        return unique_subpaths

    # Group by source and apply extraction function
    result = (
        tree_paths
        .groupby('source', sort=False)
        .apply(extract_subpaths, include_groups=False)
        .to_dict()
    )

    return result


CONTEXT_SET_METHODS: Dict[str, callable] = {
    "tree_paths": _create_context_set_tree_paths
}


def create_context(
    data: Union[pd.DataFrame, Any],
    method: str = "tree_paths",
    **kwargs
) -> dict:
    """
    Generates context for all unique roots in the tree using the specified method

    By default, tree_paths are used. Given a table of all paths in a random forest, this function generates a dictionary of all possible sub paths between each root gene and the target gene at the leaf. The key is the name of the gene on the root of the path (source) and value is the list of sub paths in the table from the root to the leaf excluding the root and the leaf. 

    Args:
        data: Input data in format appropriate for the method: tree_paths: DataFrame with 'source' and 'node*' columns. 
        method: One of 'tree_paths', 'tree' (default: 'tree_paths')
        **kwargs: Method-specific arguments

    Returns:
        dict: {source_gene: [list of subpaths]}

    Raises:
        CoRegTorError: If method is unknown
    """
    if method not in CONTEXT_SET_METHODS:
        raise CoRegTorError(
            f"Unknown method: {method}. Choose from {list(CONTEXT_SET_METHODS.keys())} ")

    generator = CONTEXT_SET_METHODS[method]
    return generator(data, **kwargs)

# ------------------
# Transform context
# ------------------


def _transform_to_gene_frequency(context_set: dict, **kwargs) -> pd.DataFrame:
    """
    Transform context set into gene frequency histograms.

    Creates a histogram counting the occurrence of each unique gene across all 
    sub-paths for each source (root gene).

    Args:
        context_set: Dictionary with structure {source: [[gene1, gene2, ...], ...]}
        **kwargs: Optional parameters
            - normalize (bool): If True, normalize frequencies to proportions (default: False)
            - min_frequency (int): Minimum frequency threshold to include gene (default: 1)

    Returns:
        pd.DataFrame: Rows are sources, columns are genes, values are frequencies/proportions. The name of root genes is the index.

    """
    normalize = kwargs.get('normalize', False)
    min_frequency = kwargs.get('min_frequency', 1)

    # Collect gene frequencies for each source
    freq_data = {}

    for source, paths in context_set.items():
        # Flatten all paths for this source into single list
        all_genes = []
        for path in paths:
            all_genes.extend(path)

        # Count frequencies
        gene_counts = Counter(all_genes)

        # Apply minimum frequency filter
        if min_frequency > 1:
            gene_counts = {gene: count for gene, count in gene_counts.items()
                           if count >= min_frequency}

        freq_data[source] = gene_counts

    # Convert to DataFrame
    # Transpose so sources are rows, genes are columns
    df = pd.DataFrame(freq_data).T
    df = df.fillna(0).astype(int)  # Fill missing values with 0

    # Optional normalization
    if normalize:
        df = df.div(df.sum(axis=1), axis=0)  # Normalize each row to sum to 1

    # Store metadata about transformation type
    df.attrs['transformation_type'] = 'gene_frequency'
    df.attrs['normalized'] = normalize

    return df


CONTEXT_TRANSFORMS: Dict[str, Callable] = {
    "gene_frequency": _transform_to_gene_frequency,
}


def transform_context(
    context_set: dict,
    method: str = "gene_frequency",
    **kwargs
) -> pd.DataFrame:
    """
    Transform context sets into feature representations easier for comparison.

    Args:
        context_set: Dictionary with structure `{source: [[gene1, gene2, ...], ...]}` (output from the `create_context` method) 
        method: Transformation method to apply. Currently available:
            - "gene_frequency": Returns a gene frequency histogram
        **kwargs: Method-specific parameters passed to the transformation function

    Returns:
        pd.DataFrame: Transformed representation (format depends on method)

    Raises:
        CoRegTorError: If method is unknown
    """
    if method not in CONTEXT_TRANSFORMS:
        raise CoRegTorError(
            f"Unknown method: {method}. Choose from {list(CONTEXT_TRANSFORMS.keys())} "
        )

    transformer = CONTEXT_TRANSFORMS[method]
    return transformer(context_set, **kwargs)


# -----------------
# Compare Context
# -----------------


def sim_to_dist(sim):
    """
    convert a similarity matrix to distance matrix 
    d = 1 - s
    """
    arr = np.array(sim, dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    dist = 1.0 - arr
    np.fill_diagonal(dist, 0)
    return dist


def gf_cosine_distance(data, **kwargs):
    """
    for a given gene frequency matrix 
    (row - root nodes, cols - gene cols, cell ij is the count of gene j in context for source i), 
    compute cosine similarity and then convert them into distance
    """

    sim = cosine_similarity(data.values)
    dist = sim_to_dist(sim)
    result = pd.DataFrame(
        dist,
        index=data.index,
        columns=data.index
    )
    return result


def gf_wasserstein_distance(data, **kwargs):
    """
    Computes the pairwise Wasserstein distance (Earth Mover's Distance)
    between all rows in a gene frequency matrix.
    """
    # Initialize an empty matrix
    n_samples = data.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    # Get values as a numpy array for speed
    values = data.values

    # Compute pairwise distances
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = wasserstein_distance(values[i], values[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    result = pd.DataFrame(
        dist_matrix,
        index=data.index,
        columns=data.index
    )
    return result


def gf_euclidean_distance(data, **kwargs):
    """
    Computes the pairwise Euclidean distance between rows.
    Euclidean distance: sqrt(sum((x - y)^2))
    """
    dist = euclidean_distances(data.values)

    # ensure the diagonal is exactly 0 to handle floating point errors
    np.fill_diagonal(dist, 0)

    result = pd.DataFrame(
        dist,
        index=data.index,
        columns=data.index
    )
    return result


def gf_weighted_jaccard_distance(data, **kwargs):
    """
    Computes the pairwise Weighted Jaccard distance between rows.
    J = sum(min(xi, yi)) / sum(max(xi, yi))
    Distance = 1 - J
    """
    values = data.values
    n_samples = values.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            v1 = values[i]
            v2 = values[j]

            # Calculate sum of mins and maxes
            sum_min = np.sum(np.minimum(v1, v2))
            sum_max = np.sum(np.maximum(v1, v2))

            # Handle division by zero for empty rows
            if sum_max == 0:
                dist = 0.0
            else:
                similarity = sum_min / sum_max
                dist = 1.0 - similarity

            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    result = pd.DataFrame(
        dist_matrix,
        index=data.index,
        columns=data.index
    )
    return result


def gf_sorensen_distance(data, **kwargs):
    """
    Computes the pairwise Sørensen (Bray-Curtis) distance.
    Formula: sum(|xi - yi|) / sum(xi + yi)
    """
    # cdist is highly optimized for pairwise operations
    dist = cdist(data.values, data.values, metric='braycurtis')

    # Ensure diagonal is 0
    np.fill_diagonal(dist, 0)

    result = pd.DataFrame(
        dist,
        index=data.index,
        columns=data.index
    )
    return result


def gf_manhattan_distance(data, **kwargs):
    """
    Computes the pairwise Manhattan (L1) distance between rows.
    Formula: sum(|xi - yi|)
    """
    # 'cityblock' is the standard identifier for Manhattan distance
    dist = pairwise_distances(data.values, metric='cityblock')

    # Ensure the diagonal is 0 to handle floating point precision
    np.fill_diagonal(dist, 0)

    result = pd.DataFrame(
        dist,
        index=data.index,
        columns=data.index
    )
    return result


def gf_canberra_distance(data, **kwargs):
    """
    Computes the pairwise Canberra distance between rows.
    Formula: sum(|xi - yi| / (|xi| + |yi|))
    """
    # pairwise_distances handles the summation logic efficiently
    dist = pairwise_distances(data.values, metric='canberra')

    # Ensure diagonal is 0
    np.fill_diagonal(dist, 0)

    result = pd.DataFrame(
        dist,
        index=data.index,
        columns=data.index
    )
    return result


def gf_jensenshannon_distance(data, **kwargs):
    """
    Computes the pairwise Jensen-Shannon distance.
    JS Distance = sqrt(JSD)
    """
    # Ensure data is float for division
    vals = data.values.astype(float)

    # Calculate row sums and handle potential zeros to avoid division by zero
    row_sums = vals.sum(axis=1, keepdims=True)
    # If a row sum is 0, we treat it as 0 to avoid NaNs
    probs = np.divide(vals, row_sums, out=np.zeros_like(
        vals), where=row_sums != 0)

    # pdist returns the condensed distance matrix (sqrt of divergence)
    # metric='jensenshannon' is highly optimized in scipy
    dist_array = pdist(probs, metric='jensenshannon')

    # Convert to square format
    dist_matrix = squareform(dist_array)

    return pd.DataFrame(
        dist_matrix,
        index=data.index,
        columns=data.index
    )


def gf_chebyshev_distance(data, **kwargs):
    """
    Computes the pairwise Chebyshev distance between rows.
    Formula: max(|xi - yi|)
    """
    # 'chebyshev' identifies the L-infinity distance
    dist = pairwise_distances(data.values, metric='chebyshev')

    # Ensure the diagonal is 0 to handle floating point precision
    np.fill_diagonal(dist, 0)

    result = pd.DataFrame(
        dist,
        index=data.index,
        columns=data.index
    )
    return result


COMPARISON_METHODS = {
    "gene_frequency": {
        "cosine_distance": gf_cosine_distance,
        "euclidean_distance": gf_euclidean_distance,
        "weighted_jaccard": gf_weighted_jaccard_distance,
        "manhattan_distance": gf_manhattan_distance,
        "sorensen_distance": gf_sorensen_distance,
        "canberra_distance": gf_canberra_distance,
        "jensenshannon_distance": gf_jensenshannon_distance,
        "chebyshev_distance": gf_chebyshev_distance
    }
}


def get_distance_measures_list():
    names = list(set([m for ctype in COMPARISON_METHODS.values()
                 for m in ctype.keys()]))
    return names


def _is_compatible(transformation_type: str, method: str) -> bool:
    """
    Check if comparison method is compatible with transformation type.

    Args:
        transformation_type: Type of transformation (e.g., 'gene_frequency')
        method: Comparison method name (e.g., 'cosine')

    Returns:
        bool: True if compatible, False otherwise
    """
    if transformation_type in COMPARISON_METHODS:
        if method in COMPARISON_METHODS[transformation_type]:
            return True
    return False


def _list_compatible_methods(transformation_type: str) -> List[str]:
    """
    List all comparison methods compatible with a transformation type.

    Args:
        transformation_type: Type of transformation (e.g., 'gene_frequency')

    Returns:
        List[str]: Sorted list of compatible method names
    """
    methods = set()
    if transformation_type in COMPARISON_METHODS:
        methods.update(COMPARISON_METHODS[transformation_type].keys())
    methods.update(COMPARISON_METHODS.get("universal", {}).keys())
    return sorted(methods)


def compare_context(
    transformed_data,
    method,
    transformation_type="gene_frequency",
    **kwargs
):
    """
    Compare contexts using specified similarity/distance metric.

    Args:
        transformed_data: Output from transform_context() - DataFrame with sources as rows
        method: Similarity/distance metric name (e.g., 'cosine')
        transformation_type: Type of transformation used .If None, attempts to read from DataFrame metadata.
        **kwargs: Metric-specific parameters
            - convert_to_distance (bool): Convert similarity to distance (1 - similarity)

    Returns:
        pd.DataFrame: Symmetric pairwise similarity/distance matrix (sources x sources)

    Raises:
        CoRegTorError: If method is unknown or incompatible with transformation type

    """
    # Validate compatibility
    if not _is_compatible(transformation_type, method):
        compatible = _list_compatible_methods(transformation_type)
        raise CoRegTorError(
            f"Method '{method}' is not compatible with transformation type {transformation_type}.  Compatible methods: {compatible} "
        )

    # Get the comparison function
    if transformation_type in COMPARISON_METHODS and method in COMPARISON_METHODS[transformation_type]:
        comparator = COMPARISON_METHODS[transformation_type][method]
    else:
        raise CoRegTorError(
            f"Unknown comparison method: {method}. Available methods: {_list_compatible_methods(transformation_type)}"
        )

    # Execute comparison
    result = comparator(transformed_data, **kwargs)

    # Store transformation type in result metadata
    # result.attrs['transformation_type'] = transformation_type

    return result
