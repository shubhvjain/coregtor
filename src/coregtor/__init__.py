from coregtor.forest import create_model_input, create_model, tree_paths
from coregtor.context import create_context, transform_context, compare_context
from coregtor.clusters import identify_coregulators
from coregtor.frequent_coreg import gene_frequency_patterns
import pandas as pd
import time

def GenerateGeneSimilarity(ge_data,
                           source_gene,
                           target_gene,
                           ge_sparsity_threshold=0.10,
                           ensemble_method="rf",
                           ensemble_options=None,
                           distance_measure="canberra_distance"):
    """
    """
    if ensemble_options is None:
        ensemble_options = {"max_depth": 5, "n_estimators": 1000, "n_jobs": -1}

    errors = []
    if ge_data is None:
        errors.append("No Gene Expression Data provided. G_sample*gene")
    if source_gene is None:
        errors.append("No source gene list provided")
    if target_gene is None:
        errors.append("No target gene provided")

    if len(errors) != 0:
        raise ValueError(f"Error: {'; '.join(errors)}")

    X, Y = create_model_input(ge_data, target_gene, source_gene)
    model, fi = create_model(
        X, Y,
        method=ensemble_method,
        options=ensemble_options,
        sparsity_threshold=ge_sparsity_threshold
    )
    paths = tree_paths(model, X, Y)
    contexts = create_context(paths)
    transformed = transform_context(contexts)
    matrix = compare_context(
        transformed, method=distance_measure, transformation_type="gene_frequency")
    return model, fi, matrix


def RunCoRegTor(
        ge_data,
        source_gene,
        target_gene,
        ge_sparsity_threshold=0.10,
        ensemble_method="rf",
        ensemble_options=None,
        distance_measure="canberra_distance",
        cluster_method="community_detection",
        cluster_options=None):
    """
    Run CoRegTor to identify co-regulators for target gene(s).

    This function orchestrates a multi-stage pipeline that trains an ensemble model on gene 
    expression data, extracts decision paths, creates biological contexts from those paths, 
    and identifies co-regulators based on context similarity.

    Args:
        ge_data: Gene expression data matrix (samples by genes).
        source_gene: List of source/predictor gene names to use as features. Only sources that exists in ge_data will be considered
        target_gene: Target gene name (str)
        ge_sparsity_threshold (float): Sparsity threshold for feature selection. Default: 0.10.
        ensemble_method (str): Ensemble method to use ("rf" for random forest, etc.). Default: "rf".
        ensemble_options (dict, optional): Additional options for ensemble model creation.
        distance_measure (str): Distance metric for context comparison. Default: "canberra_distance".
        cluster_method (str): Clustering method for identifying co-regulator groups. 
                             Default: "community_detection".
        cluster_options (dict, optional): Additional options for clustering.

    Returns:
        If return_intermediate is False:
            pd.DataFrame: Combined results with co-regulator predictions tagged by target gene.

        If return_intermediate is True:
            tuple: (pd.DataFrame, dict) where:
                - DataFrame: Combined results with co-regulator predictions
                - dict: Intermediate computations keyed by target gene, including model, 
                        paths, contexts, distance matrix, runtime, and all parameters.

    Raises:
        ValueError: If target gene is not found in gene expression data.
    """
    model, feature_importance, matrix = GenerateGeneSimilarity(ge_data, source_gene,target_gene,ge_sparsity_threshold,ensemble_method,ensemble_options,distance_measure)

    results = identify_coregulators(
        matrix, target_gene, cluster_method, cluster_options, feature_importance=feature_importance)

    intermediates = {
        "model": model,
        "feature_importance": feature_importance,
        "matrix": matrix
    }

    return results, intermediates


def MineFrequentCoReg(cluster_df, min_support=0.004):
    """
    """
    _, frequent_coregs = gene_frequency_patterns(cluster_df, min_support)
    return frequent_coregs


__all__ = ["read", "create_model_input", "create_model", "tree_paths", "create_context",
           "transform_context", "compare_context", "identify_coregulators", "utils", "RunCoRegTor"]
