"""
Validation based on various databases related to transcription factors 
"""

from coregtor.utils.validation import generate_cluster_dict, build_network, generate_stats
import networkx as nx
from coregtor.dataset import get_CollecTRI_edges_to


def _Collectri_check(gene_sources, target_gene, db, **kwargs):
    """

    """
    if gene_sources is None or len(gene_sources) == 0:
        raise ValueError("No sources provided")
    if target_gene is None:
        raise ValueError("No target gene provided")

    # gather edges and generate a graph
    interactions_found = get_CollecTRI_edges_to(
        gene_sources, target_gene, db, location=kwargs.get("location", None))
    #print(interactions_found)
    #print(gene_sources)
    #print(target_gene)
    stats = generate_stats(gene_sources+[target_gene], interactions_found)
    return interactions_found, stats


TR_SOURCES = {
    "CollecTRI": _Collectri_check
}


def check_regulation_evidence(gene_sources, target_gene, db_source="CollecTRI", db=None, **kwargs):
    """
    Returns a list of interactions that exists between the given list of sources in the selected ppi database 
    and stats 
    """
    if db_source not in TR_SOURCES:
        raise ValueError(f"Unknown  source: {db_source}")
    return TR_SOURCES[db_source](gene_sources, target_gene, db, **kwargs)
