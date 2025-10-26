import pandas as pd
import re
from typing import Union, Dict, Any,Dict,Callable,List
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

#----------------
# Create context
#----------------

def _create_context_set_tree_paths(tree_paths: pd.DataFrame, **kwargs) -> dict:
    """
    Generate context sets from tree paths DataFrame (excluding root/leaf).
    
    Args:
        tree_paths: DataFrame with 'source', 'node1', 'node2', ... columns
        
    Returns:
        dict: {source_gene: [list of sub-paths excluding root and leaf]}
    """
    if "source" not in tree_paths.columns:
        raise ValueError("DataFrame must contain a 'source' column.")

    # Identify and sort node columns (node1, node2, ...)
    node_cols = []
    for c in tree_paths.columns:
        m = re.fullmatch(r"node(\d+)", str(c))
        if m:
            node_cols.append((int(m.group(1)), c))
    if not node_cols:
        raise ValueError("No 'node*' columns found.")
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


def _create_context_set_tree(tree_data: Any, **kwargs) -> dict:
    """
    Generate context sets from tree structure.
    
    Args:
        tree_data: TODO - describe input format
        **kwargs: TODO - describe additional parameters
        
    Returns:
        dict: {source_gene: [list of subpaths]}
    """
    # TODO: Implement tree-based pathway set generation
    # TODO: Define expected input format for tree_data
    # TODO: Extract pathways from tree structure
    # TODO: Apply pathway extraction logic
    # TODO: Return formatted pathway sets
    
    raise NotImplementedError("Method 'tree' is not yet implemented")


# Dispatcher dictionary
CONTEXT_SET_METHODS: Dict[str, callable] = {
    "tree_paths": _create_context_set_tree_paths,
    "tree": _create_context_set_tree,
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
        ValueError: If method is unknown
    """
    if method not in CONTEXT_SET_METHODS:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Choose from {list(CONTEXT_SET_METHODS.keys())}"
        )
    
    generator = CONTEXT_SET_METHODS[method]
    return generator(data, **kwargs)

#------------------
# Transform context
#------------------

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
    df = pd.DataFrame(freq_data).T  # Transpose so sources are rows, genes are columns
    df = df.fillna(0).astype(int)  # Fill missing values with 0
    
    # Optional normalization
    if normalize:
        df = df.div(df.sum(axis=1), axis=0)  # Normalize each row to sum to 1
    
    # Store metadata about transformation type
    df.attrs['transformation_type'] = 'gene_frequency'
    df.attrs['normalized'] = normalize

    return df


# Dispatcher dictionary
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
        ValueError: If method is unknown
    """
    if method not in CONTEXT_TRANSFORMS:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Choose from {list(CONTEXT_TRANSFORMS.keys())}"
        )
    
    transformer = CONTEXT_TRANSFORMS[method]
    return transformer(context_set, **kwargs)

#-----------------
# Compare Context
#-----------------

def _compare_cosine(transformed_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Compute pairwise cosine similarity between all sources.
    
    Args:
        transformed_data: DataFrame with sources as rows, features as columns
        **kwargs: Optional parameters
            - convert_to_distance (bool): If True, convert similarity to distance (default: False)
    
    Returns:
        pd.DataFrame: Symmetric similarity/distance matrix (sources × sources)
    """
    convert_to_distance = kwargs.get('convert_to_distance', False)
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(transformed_data.values)
    
    # Convert to DataFrame with proper labels
    result = pd.DataFrame(
        similarity_matrix,
        index=transformed_data.index,
        columns=transformed_data.index
    )
    
    # Optional conversion to distance
    if convert_to_distance:
        result = 1 - result
    
    # Store metadata
    result.attrs['metric'] = 'cosine'
    result.attrs['is_distance'] = convert_to_distance
    
    return result


# Grouped dispatcher: maps transformation types to compatible comparison methods
COMPARISON_METHODS: Dict[str, Dict[str, Callable]] = {
    "gene_frequency": {
        "cosine": _compare_cosine,
    },
    # Universal metrics that work with any numeric data
    "universal": {
        "cosine": _compare_cosine,
    }
}


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
    # Check universal metrics
    if method in COMPARISON_METHODS.get("universal", {}):
        return True
    return False


def list_compatible_methods(transformation_type: str) -> List[str]:
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
    transformed_data: pd.DataFrame,
    method: str,
    transformation_type: str = None,
    **kwargs
) -> pd.DataFrame:
    """
    Compare contexts using specified similarity/distance metric.
    
    Args:
        transformed_data: Output from transform_context() - DataFrame with sources as rows
        method: Similarity/distance metric name (e.g., 'cosine')
        transformation_type: Type of transformation used (for validation).If None, attempts to read from DataFrame metadata.
        **kwargs: Metric-specific parameters
            - convert_to_distance (bool): Convert similarity to distance (1 - similarity)
    
    Returns:
        pd.DataFrame: Symmetric pairwise similarity/distance matrix (sources x sources)
    
    Raises:
        ValueError: If method is unknown or incompatible with transformation type
    
    """
    # Auto-detect transformation type from metadata if not provided
    if transformation_type is None:
        transformation_type = transformed_data.attrs.get('transformation_type', 'unknown')
    
    # Validate compatibility
    if not _is_compatible(transformation_type, method):
        compatible = list_compatible_methods(transformation_type)
        raise ValueError(
            f"Method '{method}' is not compatible with transformation type "
            f"'{transformation_type}'. Compatible methods: {compatible}"
        )
    
    # Get the comparison function
    if transformation_type in COMPARISON_METHODS and method in COMPARISON_METHODS[transformation_type]:
        comparator = COMPARISON_METHODS[transformation_type][method]
    elif method in COMPARISON_METHODS.get("universal", {}):
        comparator = COMPARISON_METHODS["universal"][method]
    else:
        raise ValueError(
            f"Unknown comparison method: {method}. "
            f"Available methods: {list_compatible_methods(transformation_type)}"
        )
    
    # Execute comparison
    result = comparator(transformed_data, **kwargs)
    
    # Store transformation type in result metadata
    result.attrs['transformation_type'] = transformation_type
    
    return result