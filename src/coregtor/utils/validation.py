"""
utilities methods for validation module
"""
import networkx as nx
import pandas as pd

def generate_cluster_dict(result):
    """
    Generate nodes and clusters from clustering results.

    All the results are stored in the same network. They are marked with unique cluster id which they belong to
    
    Returns:
        clusters: Dict with cluster_id -> {genes: [...], n_genes: int}
    """
    nodes = {}
    clusters = {}
    
    for idx, row in result.iterrows():
        target = row['target_gene']
        cluster_id = row['cluster_id']
        genes = [g.strip() for g in row['gene_cluster'].split(',')]
        
        # Store cluster
        clusters[cluster_id] = {
            'genes': genes,
            'n_genes': row['n_genes'],
            'target_gene': target
        }
        
        # Track nodes with their cluster
        if target not in nodes:
            nodes[target] = {'cluster_id': cluster_id, 'node_type': 'target'}
        
        for gene in genes:
            if gene not in nodes:
                nodes[gene] = {'cluster_id': cluster_id, 'node_type': 'cluster_gene'}
    
    return  clusters


def build_network(nodes, edges_df, directed=False):
    """
    Build NetworkX graph from node list and edge dataframe.

    Args:
    nodes: list of gene names (strings)
    edges_df: DataFrame with columns ['node1', 'node2', and optionally 'score']
    directed: bool, whether graph should be directed (default False for PPI)

    Returns:
    NetworkX Graph or DiGraph
    """
    if nodes is None:
        raise ValueError("No nodes provided")
    # Choose graph type
    graph_type = nx.DiGraph() if directed else nx.Graph()

    # Build graph from edges
    G = nx.from_pandas_edgelist(
        edges_df,
        source='node1',
        target='node2',
        edge_attr=True,
        create_using=graph_type
    )

    # Add all nodes from the cluster
    G.add_nodes_from(nodes)
    return G


def generate_stats(gene_list, edges):
    """
    Stats include:
        - g_density : the number of edges found vs the total possible number of edges 
        - n_nodes : the number of nodes (genes) in the graph
        - n_edges :  the number of edges found between genes included self loops 
        - n_loop_edges : the number of self loops in the graph
        - n_cc : the number of connected components in the graph 
        - cc_max_size : the number of genes in the largest connected component 
        - cc_max_ratio : the ratio of the number of genes in the  connected components by the total number of genes in the graph
        - n_node_particp : participatory nodes i.e. the number of nodes with at least one edge
        - n_isolated : number of nodes with no edges 
        - n_minimal : number of nodes with a single edge (could be a self loop)
        - n_active : number of nodes with more than one edges  

    """
    G = build_network(gene_list, edges)

    stats = {}
    stats["g_density"] = round(nx.density(G), 4)  # density of the graph
    stats["n_nodes"] = len(gene_list)  # number of nodes in the graph
    stats["n_edges"] = len(edges)  # number of edges in the graph
    stats["n_loop_edges"] = nx.number_of_selfloops(
        G)  # number of looped edges in the graph

    # connected components cc
    components = list(nx.connected_components(G))
    num_components = len(components)
    stats["n_cc"] = num_components  # number of cc
    # print(components)
    largest_component = max(components, key=len)
    largest_component_size = len(largest_component)
    # the size of the largest connected component
    stats["cc_max_size"] = largest_component_size
    # print(max(components,key=len))
    stats["cc_max_ratio"] = round(stats["cc_max_size"] / stats["n_nodes"], 4)
    # stats["cc_list"] =  components # list of all components

    # node participation
    participating_nodes = [node for node in G.nodes() if G.degree(node) > 0]
    # the number of nodes that have at least one edge
    stats["n_node_particp"] = round(
        len(participating_nodes) / len(G.nodes()), 4)

    # Get degree for all nodes
    degrees = dict(G.degree())
    # Categorize nodes
    stats["n_isolated"] = len(
        [node for node, deg in degrees.items() if deg == 0])
    stats["n_minimal"] = len(
        [node for node, deg in degrees.items() if deg == 1])
    stats["n_active"] = len(
        [node for node, deg in degrees.items() if deg >= 2])
    # print(stats)

    return stats

