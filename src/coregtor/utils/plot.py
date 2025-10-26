"""
Plots for all parts of the pipeline 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram, cophenet
from scipy.spatial.distance import squareform
from sklearn.linear_model import QuantileRegressor
from typing import Tuple, List, Optional, Dict


#------------------
# Visualize modules
#------------------


def _ensure_distance_matrix(comparison_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the comparison matrix is a distance matrix.
    
    Converts similarity to distance if needed based on metadata.
    """
    is_distance = comparison_matrix.attrs.get('is_distance', False)
    
    if not is_distance:
        # Convert similarity to distance
        dist_matrix = 1 - comparison_matrix
        dist_matrix.attrs = comparison_matrix.attrs.copy()
        dist_matrix.attrs['is_distance'] = True
        return dist_matrix
    
    return comparison_matrix


def _prepare_for_scipy(distance_matrix: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Prepare distance matrix for scipy hierarchical clustering."""
    mat = distance_matrix.copy()
    np.fill_diagonal(mat.values, 0.0)
    condensed = squareform(mat.values, checks=False)
    labels = mat.index.tolist()
    return condensed, labels


def dendrogram(
    comparison_matrix: pd.DataFrame,
    method: str = "average",
    figsize: Tuple[int, int] = (15, 9),
    leaf_rotation: float = 90,
    leaf_font_size: int = 8,
    color_threshold: float = None,
    show: bool = True
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """Plot hierarchical clustering dendrogram from comparison matrix."""
    dist_matrix = _ensure_distance_matrix(comparison_matrix)
    condensed, labels = _prepare_for_scipy(dist_matrix)
    Z = linkage(condensed, method=method)
    
    fig, ax = plt.subplots(figsize=figsize)
    scipy_dendrogram(
        Z,
        labels=labels,
        leaf_rotation=leaf_rotation,
        leaf_font_size=leaf_font_size,
        color_threshold=color_threshold,
        ax=ax
    )
    
    ax.set_title(f"Hierarchical Clustering Dendrogram ({method} linkage)")
    ax.set_ylabel("Distance")
    ax.set_xlabel("Gene")
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, ax, Z


def cophenetic(
    comparison_matrix: pd.DataFrame,
    methods: List[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    quantile: float = 0.1,
    show_regression: bool = True,
    show: bool = True
) -> Tuple[plt.Figure, np.ndarray, Dict[str, float], Optional[float]]:
    """Plot cophenetic correlation analysis to assess clustering quality."""
    if methods is None:
        methods = ["average", "complete", "ward"]
    
    dist_matrix = _ensure_distance_matrix(comparison_matrix)
    condensed, labels = _prepare_for_scipy(dist_matrix)
    
    ccc_dict = {}
    linkage_matrices = {}
    
    for method in methods:
        Z = linkage(condensed, method=method)
        ccc, coph_dist = cophenet(Z, condensed)
        ccc_dict[method] = ccc
        linkage_matrices[method] = (Z, coph_dist)
    
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    
    suggested_threshold = None
    
    for ax, method in zip(axes, methods):
        Z, coph_dist = linkage_matrices[method]
        ccc = ccc_dict[method]
        
        df = pd.DataFrame({
            "original": condensed,
            "cophenetic": coph_dist
        })
        
        ax.scatter(df["cophenetic"], df["original"], alpha=0.4, s=10)
        ax.set_xlabel("Cophenetic distance")
        ax.set_ylabel("Original distance")
        ax.set_title(f"{method.capitalize()}\nCCC = {ccc:.3f}")
        ax.grid(True, alpha=0.3)
        
        if show_regression and method == methods[0]:
            X = df["cophenetic"].values.reshape(-1, 1)
            y = df["original"].values
            
            model = QuantileRegressor(quantile=quantile, alpha=0).fit(X, y)
            x_sorted = np.sort(X[:, 0])
            y_pred = model.predict(x_sorted.reshape(-1, 1))
            
            ax.plot(x_sorted, y_pred, 'r-', linewidth=2, 
                   label=f'{int(quantile*100)}th percentile')
            
            median_coph = np.median(df["cophenetic"])
            suggested_threshold = model.predict([[median_coph]])[0]
            ax.axvline(median_coph, color='orange', linestyle='--', alpha=0.5)
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    if show:
        plt.show()
    
    return fig, axes, ccc_dict, suggested_threshold
