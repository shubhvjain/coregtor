import pandas as pd
import re
from typing import Union, Dict, Any, Callable, List
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances,pairwise_distances
from coregtor.util import CoRegTorError

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist, squareform, cdist
import pandas as pd
import numpy as np

# ----------------
# Create context
# ----------------

def _create_context_set_tree_paths(tree_paths: pd.DataFrame, genes: list = None, **kwargs) -> dict:
    """
    Generate context sets from tree paths DataFrame (excluding the queried gene and leaf).
    Single-pass version: scans each path once, extracting sub-paths for every
    occurrence of any gene of interest at any position.
    """
    if "source" not in tree_paths.columns:
        raise CoRegTorError("DataFrame must contain a 'source' column.")

    node_cols = []
    for c in tree_paths.columns:
        m = re.fullmatch(r"node(\d+)", str(c))
        if m:
            node_cols.append((int(m.group(1)), c))
    if not node_cols:
        raise CoRegTorError("No 'node*' columns found.")
    node_cols = [c for _, c in sorted(node_cols, key=lambda t: t[0])]

    all_cols = ["source"] + node_cols
    full_paths = tree_paths[all_cols].to_numpy(dtype=object)

    target_genes = set(genes) if genes is not None else set(tree_paths["source"].unique())

    # gene -> set of unique subpath tuples (dedupe via set, convert to list at end)
    seen = defaultdict(set)

    for row in full_paths:
        path = [g for g in row if pd.notna(g)]
        n = len(path)
        if n < 2:
            continue

        # single pass: check each position once against target_genes
        for idx in range(n - 1):  # exclude the leaf position itself as a "query" position
            gene_name = path[idx]
            if gene_name not in target_genes:
                continue
            sub_path = path[idx + 1: -1]
            if sub_path:
                seen[gene_name].add(tuple(sub_path))

    result = {gene: [list(sp) for sp in subpaths] for gene, subpaths in seen.items()}

    # ensure all requested genes appear in output, even if no matches found
    for gene_name in target_genes:
        result.setdefault(gene_name, [])

    return result

CONTEXT_SET_METHODS: Dict[str, callable] = {
    "tree_paths": _create_context_set_tree_paths
}


def create_context(
    data: Union[pd.DataFrame, Any],
    method: str = "tree_paths",
    genes: list = None,
    **kwargs
) -> dict:
    """
    Generates context sets for the given genes (or all root genes by default).

    Args:
        data: Input data in format appropriate for the method: tree_paths: DataFrame with 'source' and 'node*' columns.
        method: One of 'tree_paths' (default: 'tree_paths')
        genes: Optional list of gene names to build context for, regardless of their
               position in the path (root or intermediate node). Defaults to all root genes.
        **kwargs: Method-specific arguments

    Returns:
        dict: {gene: [list of subpaths]}

    Raises:
        CoRegTorError: If method is unknown
    """
    if method not in CONTEXT_SET_METHODS:
        raise CoRegTorError(
            f"Unknown method: {method}. Choose from {list(CONTEXT_SET_METHODS.keys())} ")
    # print(len(genes))
    generator = CONTEXT_SET_METHODS[method]
    return generator(data, genes=genes, **kwargs)

# ------------------
# Transform context
# ------------------

def _transform_to_gene_frequency(context_set: dict,normalize_row=True, normalize_col=False,**kwargs) -> pd.DataFrame:
    min_frequency = kwargs.get('min_frequency', 1)

    freq_data = {}
    for source, paths in context_set.items():
        all_genes = [g for path in paths for g in path]
        gene_counts = Counter(all_genes)
        freq_data[source] = dict(gene_counts)

    df = pd.DataFrame.from_dict(freq_data, orient='index').fillna(0)

    if min_frequency > 1:
        df = df.where(df >= min_frequency, 0)

    # column normalization (IDF-style) must happen BEFORE row normalization,
    # otherwise row proportions get distorted by the column reweighting
    if normalize_col:
        n_sources = df.shape[0]
        doc_freq = (df > 0).sum(axis=0)
        idf = np.log(n_sources / doc_freq.replace(0, np.nan)).fillna(0)
        df = df.mul(idf, axis=1)

    if normalize_row:
        row_sums = df.sum(axis=1)
        df = df.div(row_sums, axis=0).fillna(0)
    else:
        if not normalize_col:
            df = df.astype(int)  # only safe to cast to int if nothing was reweighted

    df.attrs['transformation_type'] = 'gene_frequency'
    df.attrs['normalize_row'] = normalize_row
    df.attrs['normalize_col'] = normalize_col
    return df

CONTEXT_TRANSFORMS = {
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
