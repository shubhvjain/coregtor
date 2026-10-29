from coregtor.forest import create_model_input, create_model, tree_paths
from coregtor.context import create_context, transform_context, compare_context
from coregtor.clusters import identify_coregulators, cluster_results_to_edges
from coregtor.frequent_coreg import gene_frequency_patterns
import os
import pandas as pd
import networkx as nx
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time

def generate_context(ge_data, source_gene, target_gene, ge_sparsity_threshold=0.10, ensemble_method="rf",ensemble_options=None):
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
    t = time.perf_counter()
    X, Y = create_model_input(ge_data, target_gene, source_gene,sparsity_threshold=ge_sparsity_threshold)
    model, fi = create_model(
        X, Y,
        method=ensemble_method,
        options=ensemble_options
    )
    paths = tree_paths(model, X, Y)
    contexts = create_context(paths)
    transformed = transform_context(contexts)
    time_taken =  time.perf_counter() - t
    return model, fi, paths, transformed, time_taken


def RunCoRegTor(
        ge_data: pd.DataFrame,
        source_gene: list,
        target_gene: str,
        ge_sparsity_threshold: float = 0.10,
        ensemble_method: str ="rf",
        ensemble_options=None,
        distance_measure: str = "canberra_distance",
        cluster_method: str = "community_detection",
        cluster_options=None ):
    """
    Run CoRegTor to predict co-regulators for a specific target gene.

    Args:
        ge_data (pd.DataFrame): 
            Gene expression data matrix (samples by genes).

        source_gene (list): 
            List of candidate regulator/predictor gene names present in `ge_data`. These genes serve as feature variables in the ensemble model.
        
        target_gene (srt): Target gene name. Must exists as a column in the gene expression data.

        ge_sparsity_threshold (float):  Minimum fraction of non-zero expression values required per column. Only sources that pass this threshold will be considered for model training. Default: 0.10. 
        
        ensemble_method (str, optional): 
            The ensemble model used to model the expression of `target_gene` using candidate 
            regulators in `source_gene`. 

            * 'rf' (Random Forest Regressor) (default)
            * 'et' (Extra Trees Regressor)

        ensemble_options (dict, optional): 
            Hyperparameters passed directly to the underlying scikit-learn estimator. See `RandomForestRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_ or   `ExtraTreesRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html>`_ for full parameter options. Default to generating 1000 estimators with a max depth of 5 utilizing all available CPUs

        distance_measure (str,optional): 
            The distance metric used to measure the pairwise dissimilarity between the transformed gene context of selected most predictive genes. 
            Supported metrics include:  'canberra_distance' (default), 'cosine_distance', 'euclidean_distance', 'manhattan_distance','weighted_jaccard', 'sorensen_distance', 'chebyshev_distance'
            
        cluster_method (str):
            Clustering method for identifying coregulatory clusters from the gene context similarity matrix. 
            Supported options:

            * 'community_detection' (Leiden Algorithm): Converts the similarity matrix into a network graph and detects communities by optimizing graph modularity.
            * 'hierarchical' (Agglomerative Hierarchical Clustering): Builds a dendrogram from distance metrics and slices it into discrete gene clusters.

        cluster_options (dict, optional): 
            Additional options for clustering.

            * For 'community_detection':
                * 'edge_creation_method': Method to build network edges ('threshold_percentile' or 'knn'). Default: 'threshold_percentile'.
                * 'edge_creation_value': Threshold cutoff or number of neighbors for graph construction. Default: 0.5.
                * 'resolution': Controls cluster granularity (higher values yield smaller modules). Default: 0.5.
                * 'objective_function': Optimization metric ('CPM' or 'modularity'). Default: 'CPM'.
                * 'n_iterations': Optimization passes for the Leiden algorithm. Default: 2.
            * For 'hierarchical':
                * 'linkage_method': Dendrogram linkage criteria ('average', 'complete', 'single', 'ward'). Default: 'average'.
                * 'auto_threshold': Cutoff strategy ('inconsistency' or 'elbow'). Default: 'inconsistency'.
                * 'threshold': Fixed distance threshold for slicing the tree.
                * 'n_clusters': Target number of clusters to extract.

    Returns:
        tuple: A 2-element tuple containing:

        * results (pd.DataFrame): Formatted summary DataFrame containing target gene predictions, assigned co-regulator module IDs, importance scores, and cluster metrics.
        * intermediates (dict): Dictionary containing internal pipeline artifacts for downstream validation and plotting, structured as:
            * model: Trained ensemble estimator object.
            * feature_importance: Feature importance for source genes.
            * path: Extracted root to leaf paths from ensemble trees.
            * matrix: Pairwise context distance matrix.
            * edges: Network edge list helpful to visualize the clusters.
            * time_taken: Total execution time in seconds.

    
    """
    model, feature_importance, paths, transformed_context, time_taken = generate_context(
        ge_data,
        source_gene,
        target_gene,
        ge_sparsity_threshold=ge_sparsity_threshold,
        ensemble_method=ensemble_method,
        ensemble_options=ensemble_options,
    )

    matrix = compare_context(
        transformed_context,
        method=distance_measure,
        transformation_type="gene_frequency",
    )

    results = identify_coregulators(
        matrix,
        target_gene,
        cluster_method,
        cluster_options,
        feature_importance=feature_importance,
    )

    intermediates = {
        "model": model,
        "feature_importance": feature_importance,
        "path": paths,
        "matrix": matrix,
        "edges": cluster_results_to_edges(results, feature_importance),
        "time_taken": time_taken,
    }

    return results, intermediates

def PlotExprData(df_matrix, tf_list, log_scale=True):
    """Plot mean expression versus variance for transcription factors. This can be helpful during the data preprocessing part. 

    Args:
        df_matrix : pandas.DataFrame
            Expression matrix where rows are samples and columns are genes.
        tf_list : list of str
            List of transcription factor gene names to filter and plot.
        log_scale : bool, optional
            If True (default), log2-transforms the expression values with a
            +1 pseudo-count before computing statistics.

    """
    # Log-transform if requested
    data = np.log2(df_matrix + 1) if log_scale else df_matrix.copy()

    # Filter for TFs present in the dataset (genes as columns)
    tfs = [gene for gene in tf_list if gene in data.columns]

    # Calculate stats directly for TFs
    means = data[tfs].mean(axis=0)
    vars_ = data[tfs].var(axis=0)

    # Plot
    plt.figure(figsize=(10,5))
    sns.scatterplot(x=means, y=vars_,s=20)

    unit = "Log2(expr + 1)" if log_scale else "expr"
    plt.xlabel(f"Mean Expression ({unit})")
    plt.ylabel(f"Variance ({unit})")
    plt.title("TF Mean vs Variance")
    plt.tight_layout()
    plt.show()


def PlotClusters(result, details, evidence=None, network_options=None):
    """
    Plot one network graph for each unique cluster returned by RunCoRegTor.

    Args:
        result: Output from RunCoRegTor.
        details: Dictionary returned alongside the main result. It must contain an
            "edges" DataFrame with columns: cluster_id, target, source, fi_score.
        evidence: Optional list of dicts such as:
            [{"target": "TP53", "sources": ["GENE_A", "GENE_B"], "title": "related", "details": "..."}]
        network_options: Optional dict with plotting settings such as:
            - figsize
            - node_size
            - font_size
            - title_prefix
            - save_path
            - show
            - evidence_colors

    Returns:
        list: One matplotlib figure per cluster.
    """
    if network_options is None:
        network_options = {}

    edges = details.get("edges") if isinstance(details, dict) else None
    if edges is None:
        raise ValueError("details must contain an 'edges' DataFrame.")
    if not isinstance(edges, pd.DataFrame):
        raise TypeError("details['edges'] must be a pandas DataFrame.")
    if edges.empty:
        return []

    if evidence is None:
        evidence = []
    evidence = evidence if isinstance(evidence, list) else []

    fig_list = []
    cluster_ids = sorted(edges["cluster_id"].dropna().unique().tolist())
    figsize = network_options.get("figsize", (9, 9))
    node_size = network_options.get("node_size", 700)
    font_size = network_options.get("font_size", 10)
    title_prefix = network_options.get("title_prefix", "Cluster")
    save_path = network_options.get("save_path")
    show = network_options.get("show", True)

    for cluster_id in cluster_ids:
        cluster_df = edges[edges["cluster_id"] == cluster_id].copy()
        if cluster_df.empty:
            continue

        target_gene = cluster_df["target"].dropna().iloc[0]
        
        # Extract all valid sources present in this specific cluster
        cluster_sources = set(cluster_df["source"].dropna().astype(str).unique())

        g = nx.Graph()
        g.add_node(target_gene, role="target")

        evidence_for_cluster = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            if item.get("target") != target_gene:
                continue
            sources = item.get("sources", [])
            if isinstance(sources, str):
                sources = [sources]
            
            cleaned_sources = [str(s).strip() for s in sources if str(s).strip()]
            
            # Skip this evidence if it has no sources or if ANY source is missing from the cluster
            if not cleaned_sources or not all(s in cluster_sources for s in cleaned_sources):
                continue

            evidence_for_cluster.append({
                "title": item.get("title", "Evidence"),
                "details": item.get("details", ""),
                "sources": cleaned_sources,
            })

        cmap = plt.colormaps["tab20"] if hasattr(plt, "colormaps") else plt.cm.get_cmap("tab20")
        evidence_colors_map = {}
        legend_info = []

        for idx, item in enumerate(evidence_for_cluster):
            color = cmap(idx % 20)
            sources_str = ",".join(item['sources'])
            legend_info.append((f"{item['title']}({sources_str})", color))
            for source in item["sources"]:
                evidence_colors_map[source] = color

        for _, row in cluster_df.iterrows():
            source = row.get("source")
            if pd.isna(source):
                continue
            source = str(source)
            g.add_node(source, role="source")
            weight = row.get("fi_score")
            if pd.isna(weight):
                weight = 1.0
            
            edge_color = evidence_colors_map.get(source, "gray")
            g.add_edge(target_gene, source, weight=float(weight), color=edge_color)

        fig, ax = plt.subplots(figsize=figsize)
        pos = nx.spring_layout(g, seed=42)

        node_colors = []
        for node in g.nodes():
            if node == target_gene:
                node_colors.append("gold")
            elif node in evidence_colors_map:
                node_colors.append(evidence_colors_map[node])
            else:
                node_colors.append("lightblue")

        edge_color_list = [g[u][v].get("color", "gray") for u, v in g.edges()]

        nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_size, ax=ax)
        nx.draw_networkx_labels(g, pos, font_size=font_size, ax=ax)
        nx.draw_networkx_edges(
            g,
            pos,
            width=[max(1.0, g[u][v].get("weight", 1.0) * 3) for u, v in g.edges()],
            edge_color=edge_color_list,
            alpha=0.8,
            ax=ax,
        )

        if evidence_for_cluster:
            legend_handles = []
            for title, color in legend_info:
                handle = plt.Line2D(
                    [], [],
                    color=color,
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    markerfacecolor=color,
                    markeredgecolor='black',
                    markersize=8,
                    label=title,
                )
                legend_handles.append(handle)

            ax.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=len(legend_handles),
                frameon=False,
                fontsize=9,
                columnspacing=1.5,
                handletextpad=0.6,
            )

        ax.set_title(f"{title_prefix} {target_gene}")
        ax.axis("off")
        fig.tight_layout()

        fig_list.append(fig)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, f"cluster_{cluster_id}.png"), dpi=200, bbox_inches="tight")

        if show:
            plt.show()

    return fig_list


def MineFrequentCoReg(cluster_df, min_support=0.004):
    """
    """
    _, frequent_coregs = gene_frequency_patterns(cluster_df, min_support)
    return frequent_coregs


__all__ = [ "RunCoRegTor","PlotClusters", "PlotExprData", "create_model_input", "create_model", "tree_paths", "create_context",
           "transform_context", "compare_context", "identify_coregulators", "utils" ]
