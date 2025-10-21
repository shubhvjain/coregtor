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
from collections import Counter, defaultdict
from sklearn.tree import _tree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from typing import List, Dict, Any,Union, Optional
from pathlib import Path
from coregtor._expression import create_model_input

PathLike = Union[str, Path]

def create_model(X,Y,model="rf",model_options=None):
    """
    Train an ensemble regression model to predict the expression of a target gene  using transcription factors in the gene expression data as input features.

    Currently 2 ensemble model are supported : `sklearn.ensemble.RandomForestRegressor` and `sklearn.ensemble.ExtraTreesRegressor`

    Args:
        X (pd.DataFrame) : Gene expression data (sample by genes) of transcription factors. This can be generated using the `create_model_input` method
        Y (pd.DataFrame) : Gene expression data for the target gene. This can be generated using the `create_model_input` method.
        model (str) : The type of ensemble based model. This must be a valid model in sklearn.ensemble module. Use `rf` (default) for random forest regressor,  `et` for extra trees regressor
        model_options (dict,optional) : Dictionary of key value pairs to specify training options. See the scikit-learn model docs for options: `RandomForestRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_ or  `ExtraTreesRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html>`_

    Returns:
       Trained sklearn ensemble model
    """

    if model == "rf":
        ensemble = RandomForestRegressor(**model_options)
    elif model == "et":
        ensemble = ExtraTreesRegressor(**model_options)
    else:
        raise ValueError(f"Invalid method '{model}'. Must be 'rf' or 'et'")
    
    # Train the model and measure time
    ensemble.fit(X, Y.values.ravel())
    #ensemble.fit(X.values, Y.values.ravel())
    return ensemble


def tree_paths(model,X,Y) -> pd.DataFrame:
    """
    Extract all root-to-leaf decision paths from a trained ensemble model.
    
    This is the main entry point for path extraction. It processes a trained
    ensemble model to extract all unique decision paths.
    
    Args:
      model : sklearn ensemble model
      ge_data (pd.DataFrame) : Gene expression data used to train the model. This is required to extract gene names.
      X (pd.DataFrame) : Input of the model. This is required to get gene names 
      Y (pd.DataFrame) : Training output of the model. This is required to get gene name  of the target 

    
    Returns:
      pd.DataFrame :  DataFrame containing all decision tree paths with columns:
        - tree: Tree index within the ensemble (0-based)
        - source: First gene in the decision path (root decision)
        - target: Target gene being predicted (constant across all rows)
        - path_length: Number of decision nodes in the path
        - node1, node2, ...: Genes used at each decision level
        - Unused node columns are filled with None
    
    """    
    # Extract all paths from the ensemble
    paths_df = extract_paths_from_forest(model, X.columns, Y.columns[0])
    return paths_df


def extract_paths_from_tree(tree, feature_names: List[str], target_col: str) -> List[Dict[str, Any]]:
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


def extract_paths_from_forest(ensemble_model, feature_names: List[str], target_col: str = "target") -> pd.DataFrame:
    """
    Extract and consolidate all paths from an ensemble of decision trees.
    
    This function processes each tree in an ensemble model, extracts all paths,
    and consolidates them into a structured DataFrame suitable for analysis.
    The resulting format enables easy statistical analysis and visualization.
    
    Parameters:
    -----------
    ensemble_model : sklearn ensemble model
        Trained Random Forest or Extra Trees model containing multiple estimators.
    feature_names : List[str]
        List of feature (gene) names used in the model.
    target_col : str, default="target"
        Name of the target column/gene being predicted.
    
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
        tree_paths = extract_paths_from_tree(tree_structure, feature_names, target_col)
        
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
