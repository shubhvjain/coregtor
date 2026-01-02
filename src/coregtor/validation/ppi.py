"""
Validating through Protein Protein Interaction databases 
"""
from coregtor.dataset import hippie_get_edges, string_get_edges
import networkx as nx
from coregtor.utils.validation import generate_cluster_dict,build_network,generate_stats

def _HIPPIE_check(gene_list, ppi_db, **kwargs):
    """

    """
    # gather edges and generate a graph
    interactions_found = hippie_get_edges(gene_list, db=ppi_db, gene_mapping=kwargs.get(
        "gene_mapping", None), location=kwargs.get("location", None))
    stats = generate_stats(gene_list, interactions_found)
    return interactions_found, stats


def _STRING_check(gene_list, ppi_db=None,filter=None,**kwargs):
    """
    """
    interactions_found = string_get_edges(
        gene_list, db=ppi_db,location=kwargs.get("location", None))
    
    if filter:
        # Keep only rows where ALL columns in filter list have non-zero values
        interactions_found = interactions_found[
            (interactions_found[filter] != 0).all(axis=1)
        ]

    stats = generate_stats(gene_list, interactions_found)
    return interactions_found,stats


PPI_SOURCES = {
    "HIPPIE": _HIPPIE_check,
    "STRING": _STRING_check
}

def check_PPI_interactions(gene_list, ppi_source="HIPPIE", ppi_db=None, **kwargs):
    """
    Returns a list of interactions that exists between the given list of sources in the selected ppi database 
    and stats 
    """
    if ppi_source not in PPI_SOURCES:
        raise ValueError(f"Unknown  source: {ppi_source}")

    return PPI_SOURCES[ppi_source](gene_list, ppi_db, **kwargs)
