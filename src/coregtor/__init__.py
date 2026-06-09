from coregtor.forest import create_model_input, create_model, tree_paths
from coregtor.context import create_context, transform_context, compare_context
from coregtor.clusters import identify_coregulators
import coregtor.utils

import pandas as pd
import time

def get_model_input(ge_raw, target, source_genes):
    df = ge_raw.transpose().rename_axis("sample_name")
    if target not in df.columns:
        raise ValueError("Target gene not found")
    feature_genes = [g for g in source_genes if g in df.columns]
    if target in feature_genes:
        feature_genes.remove(target)
    X = df[feature_genes]
    Y = df[[target]]
    return X, Y


def RunCoRegTor(
    ge_data,
    source_gene,
    target_gene,
    ge_sparsity_threshold=0.10,
    ensemble_method="rf",
    ensemble_options=None,
    distance_measure="canberra_distance",
    cluster_method="community_detection",
    cluster_options=None
):
    # Safely handle default mutable dictionaries
    if ensemble_options is None:
        ensemble_options = {"max_depth": 5, "n_estimators": 1000, "n_jobs": -1}
    if cluster_options is None:
        cluster_options = {"edge_creation_method": "threshold_percentile",
                           "edge_creation_value": 0.5, "resolution": 0.25}

    errors = []
    if ge_data is None:
        errors.append("No Gene Expression Data provided. G_sample*gene")
    if source_gene is None:
        errors.append("No source gene list provided")
    if target_gene is None:
        errors.append("No target gene provided")

    if len(errors) != 0:
        raise ValueError(f"Error: {'; '.join(errors)}")
    
    t = time.perf_counter()
    X, Y = get_model_input(ge_data, target_gene, source_gene)
    #print(X)
    #print(Y)
    model = create_model(
        X, Y,
        method=ensemble_method, 
        options=ensemble_options, 
        sparsity_threshold=ge_sparsity_threshold
    )
    paths = tree_paths(model, X, Y)
    contexts = create_context(paths)
    transformed = transform_context(contexts)
    matrix = compare_context(transformed, method=distance_measure, transformation_type="gene_frequency")
    results = identify_coregulators(matrix, target_gene, cluster_method, cluster_options, "")

    time_spent = time.perf_counter() - t
    
    intermediates = {
        "model": model,
        "paths": paths,
        "contexts": contexts,
        "matrix": matrix,
        "run_time_second": time_spent,
        "result": results,
        "source_gene": source_gene,
        "target_gene": target_gene,
        "ge_sparsity_threshold": ge_sparsity_threshold,
        "ensemble_method": ensemble_method,
        "ensemble_options": ensemble_options,
        "distance_measure": distance_measure,
        "cluster_method": cluster_method,
        "cluster_options": cluster_options
    }
    
    return results, intermediates

    



__all__ = ["read", "create_model_input", "create_model", "tree_paths", "create_context",
           "transform_context", "compare_context", "identify_coregulators", "utils","RunCoRegTor"]
