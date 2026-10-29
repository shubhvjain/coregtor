"""
Module for Ensemble Generation (using Gene expression data)

This module provides functionalities for generating and analyzing ensemble models for gene expression data (for one target gene). 
It includes:
- Model generation and training
- Root-to-leaf path extraction and analysis
- Path-based gene interaction insights
"""

import pandas as pd
import numpy as np
from sklearn.tree import _tree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from typing import List, Dict, Any
from coregtor.util import CoRegTorError

def _expression_pre_checks(X: pd.DataFrame, Y: pd.DataFrame, min_threshold: float = 0.10) -> pd.DataFrame:
    """
    Validate target gene Y and filter uninformative sparse feature columns in X.

    Args:
        X (pd.DataFrame): Feature expression matrix (samples x TFs).
        Y (pd.DataFrame): Target gene expression matrix (samples x 1).
        min_threshold (float): Minimum fraction of non-zero values required per column.

    Returns:
        pd.DataFrame: Filtered feature matrix X containing only genes that meet the threshold.
    """
    total_samples = len(Y)
    
    # 1. Validate Target Gene Y (Strict check)
    y_pct_nonzero = (Y != 0).mean(axis=0)
    if (y_pct_nonzero < min_threshold).any():
        failed_col = y_pct_nonzero.idxmin()
        failed_pct = y_pct_nonzero.min()
        failed_count = (Y[failed_col] != 0).sum()
        min_req = total_samples * min_threshold
        
        raise CoRegTorError(
            f"Target gene column '{failed_col}' in Y has only {failed_count}/{total_samples} "
            f"({failed_pct:.1%}) non-zero samples. Need ≥{min_req:.1f} ({min_threshold:.1%}).",
            "validation"
        )

    # 2. Filter Feature Matrix X based on sparsity
    total_source_genes = X.shape[1]
    x_pct_nonzero = (X != 0).mean(axis=0)
    valid_tfs = x_pct_nonzero[x_pct_nonzero >= min_threshold].index
    num_kept = len(valid_tfs)

    if num_kept == 0:
        raise CoRegTorError(
            f"No feature genes in X met the minimum non-zero threshold of {min_threshold:.1%}.",
            "validation"
        )

    if num_kept < total_source_genes:
        print(f"Using {num_kept} out of {total_source_genes} source genes with sparsity ≥ {min_threshold:.1%}")

    return X[valid_tfs]


def create_model_input(
    raw_ge_data: pd.DataFrame, 
    target_gene: str, 
    t_factors: list = None, 
    sparsity_threshold: float = 0.10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare gene expression data for training by splitting and validating feature and target matrices.
    
    This function splits raw gene expression data into features (X) and target (Y), filters features 
    to transcription factors if provided, drops duplicate columns, and enforces sparsity validation.
    
    Args:
        raw_ge_data (pd.DataFrame): Gene expression data in samples x genes format.
        target_gene (str): Name of the target gene to predict. Must be present in raw_ge_data columns.
        t_factors (list, optional): A list of transcription factor gene names.
        sparsity_threshold (float): Minimum fraction of non-zero expression values required per column.
      
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 
            X - Validated feature matrix (samples x TFs) filtered by sparsity_threshold
            Y - Validated target matrix (samples x 1)
    """
    # Check if raw_ge_data is empty
    if raw_ge_data.empty:
        raise CoRegTorError("The gene expression data (raw_ge_data) is empty.", "validation")

    # Check if target_gene is in raw_ge_data columns
    if target_gene not in raw_ge_data.columns:
        raise CoRegTorError(f"Target gene '{target_gene}' not found in gene expression data columns.", "validation")
    
    # Drop duplicate gene columns, keeping the first occurrence
    if raw_ge_data.columns.duplicated().any():
        raw_ge_data = raw_ge_data.loc[:, ~raw_ge_data.columns.duplicated(keep="first")]
    
    # Extract the target matrix Y
    Y = raw_ge_data[[target_gene]]

    # Get initial feature columns
    if t_factors is not None and len(t_factors) > 0:
        X_cols = [gene for gene in t_factors if gene in raw_ge_data.columns and gene != target_gene]
    else:
        # Use all columns except target if no t_factors provided
        X_cols = [col for col in raw_ge_data.columns if col != target_gene]
    
    if not X_cols:
        raise CoRegTorError("No valid feature columns found after filtering.", "validation")

    X = raw_ge_data[X_cols]

    # Perform pre-checks: validates Y and filters sparse columns in X
    X = _expression_pre_checks(X, Y, min_threshold=sparsity_threshold)

    return X, Y


def _extract_feature_importance(ensemble_model, X_data):
    """
    Extract feature importance scores from a trained ensemble model.

    The feature names are taken from the supplied gene expression dataframe,
    so the dataframe must contain the same feature columns used during training.

    Args:
        ensemble_model: Trained sklearn ensemble model with ``feature_importances_``.
        X_data (pd.DataFrame): Input dataframe used for the feature names.

    Returns:
        dict: Ordered mapping of ``gene -> importance`` sorted from highest to lowest importance.
    """
    if X_data.empty:
        raise CoRegTorError("The gene expression data (gene_expression_data) is empty.", "validation")

    if not hasattr(ensemble_model, "feature_importances_"):
        raise CoRegTorError("The supplied model does not expose feature importance scores.", "validation")

    feature_names = list(X_data.columns)
    importance_scores = np.asarray(ensemble_model.feature_importances_)

    if len(feature_names) != len(importance_scores):
        raise CoRegTorError(
            "The number of feature columns does not match the model's feature importance vector.",
            "validation",
        )

    importance_pairs = sorted(zip(feature_names, importance_scores), key=lambda item: item[1], reverse=True)
    return {gene: float(score) for gene, score in importance_pairs}

def create_model(X: pd.DataFrame, Y: pd.DataFrame, method: str = "rf", options: dict = None):
    """
    Train an ensemble regression model to predict the expression of target gene Y using feature genes X.

    Currently supported models:
    - `RandomForestRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_ (`rf`)
    - `ExtraTreesRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html>`_ (`et`)

    Args:
        X (pd.DataFrame): Gene expression feature matrix (samples x TFs). Generated via `create_model_input`.
        Y (pd.DataFrame): Gene expression target matrix (samples x 1). Generated via `create_model_input`.
        method (str): Type of ensemble model ('rf' or 'et'). Default is 'rf'.
        options (dict, optional): Hyperparameters for the estimator. Overrides defaults (max_depth=5, n_estimators=1000, n_jobs=-1).
            See `RandomForestRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_ or 
            `ExtraTreesRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html>`_ for full parameter options.

    Returns:
        tuple: A tuple containing:
            - ensemble (RegressorMixin): The trained scikit-learn ensemble model.
            - importance (dict): Feature gene importances ordered from highest to lowest score.
    """
    # Safely merge custom options with default parameters
    default_options = {"max_depth": 5, "n_estimators": 1000, "n_jobs": -1}
    if options is not None:
        default_options.update(options)
    options = default_options

    if method == "rf":
        ensemble = RandomForestRegressor(**options)
    elif method == "et":
        ensemble = ExtraTreesRegressor(**options)
    else:
        raise CoRegTorError(f"Invalid method '{method}'. Must be 'rf' or 'et'.", "validation") 
    
    # Train the model and extract feature importance
    ensemble.fit(X, Y.values.ravel())
    importance = _extract_feature_importance(ensemble, X)
    
    return ensemble, importance

def tree_paths(model,X,Y):
    """
    Extract all root-to-leaf decision paths from a trained ensemble model.
    
    This is the main entry point for path extraction. It processes a trained ensemble model to extract all unique decision paths.
    
    Args:
      model : sklearn ensemble model
      X (pd.DataFrame) : Input of the model. This is required to get gene names 
      Y (pd.DataFrame) : Training output of the model. This is required to get gene name  of the target 
    
    Returns:
      pd.DataFrame :  DataFrame containing all decision tree paths with columns:
        - tree: Tree index within the ensemble (0-based)
        - source: First gene in the decision path (root decision)
        - target: Target gene being predicted (constant across all rows)
        - path_length: Number of decision nodes in the path
        - node1, node2, ...: Genes used at each decision level (excluding source)
        - Unused node columns are filled with None
    
    """    
    # Extract all paths from the ensemble
    paths_df = extract_paths_from_forest(model, X.columns, Y.columns[0])
    return paths_df


def _extract_paths_from_tree(tree, feature_names, target_col):
    """
    Extract all root-to-leaf paths from a single decision tree.
    
    This function recursively traverses a decision tree to extract all possible
    paths from root to leaf nodes. Each path represents a sequence of gene-based
    decisions that lead to a prediction.
    
    Args:
    tree (sklearn.tree._tree.Tree) :  The tree structure from a trained decision tree estimator.
    feature_names (List[str]) : List of feature (gene) names corresponding to tree feature indices.
    target_col (str) :  Name of the target gene/column being predicted.
    
    Returns:
      List[Dict[str, Any]]
        List of path dictionaries, each containing:
        - source: First gene in the path (None if path is empty)
        - target: Target gene name
        - nodes: List of genes in the intermediate path (excluding source)
        - path_length: Total number of decision nodes
    
    Notes:
    ------
    - Uses depth-first traversal to explore all paths
    - Leaf nodes are identified by TREE_UNDEFINED feature value
    - Empty paths (direct root-to-leaf) are handled gracefully
    - Paths are collected in a set to automatically deduplicate within tree
    
    """
    # Use set to automatically handle duplicate paths within the tree
    unique_paths = set()
    
    def _traverse_tree(node_id: int, current_path_features: List[str]) -> None:
        """
        Recursively traverse tree nodes to collect complete paths.
        
        Parameters:
        -----------
        node_id : int
            Current node ID being visited.
        current_path_features : List[str]
            List of feature names encountered from root to current node.
        """
        # Check if current node is a leaf (end of path)
        if tree.feature[node_id] != _tree.TREE_UNDEFINED:
            # Internal node: get feature name and continue traversal
            feature_name = feature_names[tree.feature[node_id]]
            updated_path = current_path_features + [feature_name]
            
            # Recursively traverse left and right children
            _traverse_tree(tree.children_left[node_id], updated_path)
            _traverse_tree(tree.children_right[node_id], updated_path)
        else:
            # Leaf node reached: record complete path
            source_gene = current_path_features[0] if current_path_features else None
            intermediate_nodes = current_path_features[1:] if len(current_path_features) > 1 else []
            
            # Create path tuple for set storage (immutable)
            path_tuple = (
                source_gene,
                target_col,
                tuple(intermediate_nodes),
                len(current_path_features)
            )
            unique_paths.add(path_tuple)
    
    # Start traversal from root node
    _traverse_tree(node_id=0, current_path_features=[])
    
    # Convert set of tuples back to list of dictionaries
    path_list = [
        {
            "source": source,
            "target": target,
            "nodes": list(nodes),
            "path_length": length
        }
        for (source, target, nodes, length) in unique_paths
    ]
    
    return path_list


def extract_paths_from_forest(ensemble_model, feature_names, target_col = "target"):
    """
    Extract and consolidate all paths from an ensemble of decision trees.
    
    This function processes each tree in an ensemble model, extracts all paths,
    and consolidates them into a structured DataFrame suitable for analysis.
    The resulting format enables easy statistical analysis and visualization.
    
    Args:
    ensemble_model : sklearn ensemble model Trained Random Forest or Extra Trees model containing multiple estimators.
    feature_names  (List[str]) : List of feature (gene) names used in the model.
    target_col (str): default="target",  Name of the target column/gene being predicted.
    
    Returns:
    --------
    pd.DataFrame
        Consolidated DataFrame with all paths from the ensemble:
        - tree: Tree index (0-based) within the ensemble
        - source: Root decision gene for each path
        - target: Target gene (constant across all paths)
        - path_length: Number of decision nodes in each path
        - node1, node2, ..., nodeN: Genes at each decision level
        - Unused node columns filled with None
    
    Notes:
    ------
    - Automatically determines maximum path length for column creation
    - Each tree contributes its unique paths with tree identification
    - Node columns are numbered sequentially (node1, node2, etc.)
    - Shorter paths are padded with None values for consistent structure
    
    """
    all_forest_paths = []
    
    # Process each tree in the ensemble
    for tree_index, estimator in enumerate(ensemble_model.estimators_):
        tree_structure = estimator.tree_
        
        # Extract paths from this individual tree
        tree_paths = _extract_paths_from_tree(tree_structure, feature_names, target_col)
        
        # Add tree identification to each path
        for path_dict in tree_paths:
            path_dict["tree"] = tree_index
        
        # Add to master collection
        all_forest_paths.extend(tree_paths)
    
    # Determine maximum path length for DataFrame structure
    if not all_forest_paths:
        # Handle empty case
        return pd.DataFrame(columns=["tree", "source", "target", "path_length"])
    
    max_path_length = max(path["path_length"] for path in all_forest_paths)
    
    # Convert to structured DataFrame format
    structured_records = []
    for path_dict in all_forest_paths:
        # Start with basic path information
        record = {
            "tree": path_dict["tree"],
            "source": path_dict["source"], 
            "target": path_dict["target"],
            "path_length": path_dict["path_length"]
        }
        
        # Add intermediate nodes to numbered columns
        intermediate_nodes = path_dict["nodes"]
        for node_index, gene_name in enumerate(intermediate_nodes):
            record[f"node{node_index + 1}"] = gene_name
        
        # Pad shorter paths with None for consistent column structure
        for node_index in range(len(intermediate_nodes) + 1, max_path_length):
            record[f"node{node_index}"] = None
            
        structured_records.append(record)
    
    # Create final DataFrame
    paths_df = pd.DataFrame(structured_records)
    
    return paths_df
