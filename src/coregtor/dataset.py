"""
module to handle biological databases from external sources 
"""

import os
import requests
import shutil
import pandas as pd
from pathlib import Path
from coregtor.config import DATA_SOURCES
import decoupler as dc

def convert_genes(input: list[str], 
                 input_type: str = "symbol", 
                 output_type: str = "ensembl_gene_id",
                 data: pd.DataFrame = None,
                 location = None) -> dict:
    """
    Convert gene identifiers from one type to another using BioMart data.
    
    Args:
        input: List of gene identifiers to convert
        input_type: Source identifier type. Options:
            - 'symbol': HGNC gene symbol (e.g., 'TP53')
            - 'ensembl_gene_id': Ensembl gene ID (e.g., 'ENSG00000141510')
            - 'entrez_id': NCBI Entrez Gene ID (e.g., '7157')
            - 'uniprot_id': UniProt/Swiss-Prot ID
            - 'refseq_id': RefSeq mRNA accession
        output_type: Target identifier type (same options as input_type)
        data: Pre-loaded BioMart DataFrame. If None, will load from location
        location: Directory where BioMart data is stored (only used if data is None)
    
    Returns:
        dict: Mapping of input identifiers to output identifiers
              Returns None for identifiers that couldn't be mapped
    
    """
    # Load data if not provided
    if data is None:
        print(f"Loading BioMart data from {location}...")
        data = biomart_gene_mapping(location)
    
    results = {}
    
    for gene_id in input:
        # Find matching rows
        mask = data[input_type].astype(str) == str(gene_id)
        matches = data[mask]
        
        if len(matches) > 0:
            # Take first match
            result = matches.iloc[0][output_type]
            # Handle NaN values
            results[gene_id] = result if pd.notna(result) else None
        else:
            results[gene_id] = None
    
    return results


def download(url, file):
    """Download a file from URL if it doesn't already exist."""
    file_path = Path(file)
    if file_path.exists():
        print(f"File already exists: {file_path}")
        return file_path

    file_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(file_path, 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)

        print(f"Successfully downloaded to {file_path}")
        return file_path

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        if file_path.exists():
            file_path.unlink()
        raise

def biomart_gene_mapping(location):
    """
    Download and load BioMart gene mapping data into a DataFrame.
    
    This includes mappings between:
    - Ensembl Gene IDs
    - HGNC gene symbols
    - Entrez Gene IDs (NCBI)
    - UniProt/Swiss-Prot IDs
    - RefSeq mRNA accessions
    - Gene descriptions
    
    Args:
        location: Directory where BioMart data should be stored
    
    Returns:
        pandas DataFrame with gene ID mappings
    """
    url = DATA_SOURCES["biomart_gene_mapping"]["url"]
    filename = DATA_SOURCES["biomart_gene_mapping"]["name"]
    destination = os.path.join(location, filename)
    
    # Download file if needed
    filepath = download(url, destination)
    
    # Load into DataFrame
    print(f"Loading BioMart gene mapping data into DataFrame...")
    df = pd.read_csv(filepath, sep='\t')
    
    # Rename columns for consistency
    df.columns = ['ensembl_gene_id', 'symbol', 'entrez_id', 
                  'uniprot_id', 'refseq_id', 'description']
    
    # Add 9606. prefix to Ensembl IDs for STRING compatibility
    df['ensembl_gene_id'] = '9606.' + df['ensembl_gene_id'].astype(str)
    df['entrez_id'] = pd.to_numeric(df['entrez_id'], errors='coerce').astype('Int64')
    #print(f"Loaded {len(df)} gene mappings")
    #print(f"Columns: {', '.join(df.columns)}")
    
    return df

def hippie_ppi(location):
    """
    Download and load HIPPIE protein-protein interaction database into a DataFrame.

    Args:
        location: Directory where HIPPIE data should be stored

    Returns:
        pandas DataFrame with columns: protein1, protein2, combined_score
    """

    url = DATA_SOURCES["hippie_ppi"]["url"]
    filename = DATA_SOURCES["hippie_ppi"]["name"]
    destination = os.path.join(location, filename)

    # Download file if needed
    filepath = download(url, destination)

    # Load into DataFrame
    print(f"Loading HIPPIE data into DataFrame...")
    df = pd.read_csv(filepath, sep='\t', header=None, names=[
        "uniprot_id_1", "entrez_id_1", "uniprot_id_2", "entrez_id_2", "score", "comments"])
    df['entrez_id_1'] = pd.to_numeric(df['entrez_id_1'], errors='coerce').astype('Int64')
    df['entrez_id_2'] = pd.to_numeric(df['entrez_id_2'], errors='coerce').astype('Int64')
    
    # print(f"Loaded {len(df)} protein interactions")
    return df

def _hippie_get_edges_within_sources(db: pd.DataFrame, sources: list[int]) -> pd.DataFrame:
    """Get all edges between genes in sources list (uses Entrez IDs)."""
    
    sources_set = set(sources)
    
    mask_forward = (
        db['entrez_id_1'].isin(sources_set) &
        db['entrez_id_2'].isin(sources_set)
    )
    
    edges = db[mask_forward].copy()
    
    return edges

def _hippie_get_edges_sources_to_target(db: pd.DataFrame, sources: list[int], target: int) -> pd.DataFrame:
    """Get edges from sources to target (uses Entrez IDs)."""
    
    sources_set = set(sources)
    
    # Forward direction: source -> target
    mask_forward = (
        db['entrez_id_1'].isin(sources_set) &
        (db['entrez_id_2'] == target)
    )
    
    # Reverse direction: target -> source
    mask_reverse = (
        (db['entrez_id_1'] == target) &
        db['entrez_id_2'].isin(sources_set)
    )
    
    edges = db[mask_forward | mask_reverse].copy()
    
    return edges

def hippie_get_edges(
    sources: list[str],
    target: str = None,
    db: pd.DataFrame = None,
    location: str = None,
    gene_mapping: pd.DataFrame = None,
    include_type: bool = True
) -> pd.DataFrame:
    """
    Get edges from HIPPIE database between sources and optionally to a target.
    
    Args:
        sources: List of gene symbols (e.g., ['TP53', 'BRCA1'])
        target: Optional target gene symbol
        db: Pre-loaded HIPPIE DataFrame (if None, will load from location)
        location: Directory where HIPPIE data is stored (required if db is None)
        gene_mapping: Pre-loaded BioMart gene mapping DataFrame (if None, will load from location)
        include_type: Whether to add 'edge_type' column
    
    Returns:
        DataFrame with columns: node1, node2, score, comments, [edge_type], edge_source
    """
    
    if db is None:
        if location is None:
            raise ValueError("Either db or location must be provided")
        db = hippie_ppi(location)
    
    if gene_mapping is None:
        if location is None:
            raise ValueError("Either gene_mapping or location must be provided")
        print("Loading gene mapping data...")
        gene_mapping = biomart_gene_mapping(location)
    
    # Create symbol to Entrez ID mapping (entrez_id already converted to Int64 in biomart_gene_mapping)
    symbol_to_entrez = gene_mapping.dropna(subset=['entrez_id']).set_index('symbol')['entrez_id'].to_dict()
    
    # Get Entrez IDs for sources
    source_entrez_ids = []
    for gene in sources:
        entrez_id = symbol_to_entrez.get(gene)
        if entrez_id and pd.notna(entrez_id):
            source_entrez_ids.append(int(entrez_id))
        else:
            print(f"Warning: Could not find Entrez ID for gene '{gene}'")
    
    # Get Entrez ID for target if provided
    target_entrez_id = None
    if target:
        target_entrez_id = symbol_to_entrez.get(target)
        if target_entrez_id and pd.notna(target_entrez_id):
            target_entrez_id = int(target_entrez_id)
        else:
            print(f"Warning: Could not find Entrez ID for target gene '{target}'")
    
    if len(source_entrez_ids) == 0:
        print("No valid Entrez IDs found for source genes")
        cols = ['node1', 'node2', 'score', 'comments']
        if include_type:
            cols.append('edge_type')
        cols.append('edge_source')
        return pd.DataFrame(columns=cols)
    
    all_edges = []
    
    # Get within-sources edges
    within_edges = _hippie_get_edges_within_sources(db, source_entrez_ids)
    if len(within_edges) > 0:
        if include_type:
            within_edges['edge_type'] = 'within_cluster'
        all_edges.append(within_edges)
    
    # Get sources-to-target edges
    if target_entrez_id:
        target_edges = _hippie_get_edges_sources_to_target(db, source_entrez_ids, target_entrez_id)
        if len(target_edges) > 0:
            if include_type:
                target_edges['edge_type'] = 'to_target'
            all_edges.append(target_edges)
    
    if len(all_edges) == 0:
        cols = ['node1', 'node2', 'score', 'comments']
        if include_type:
            cols.append('edge_type')
        cols.append('edge_source')
        return pd.DataFrame(columns=cols)
    
    edges = pd.concat(all_edges, ignore_index=True)
    
    # Rename columns
    edges = edges.rename(columns={
        'entrez_id_1': 'node1_entrez',
        'entrez_id_2': 'node2_entrez'
    })
    
    # Convert Entrez IDs back to gene symbols for output
    entrez_to_symbol = gene_mapping.dropna(subset=['entrez_id']).set_index('entrez_id')['symbol'].to_dict()
    
    edges['node1'] = edges['node1_entrez'].map(entrez_to_symbol).fillna(edges['node1_entrez'].astype(str))
    edges['node2'] = edges['node2_entrez'].map(entrez_to_symbol).fillna(edges['node2_entrez'].astype(str))
    
    # Add edge source
    edges["edge_source"] = "hippie_ppi"
    
    # Select and order columns
    result_cols = ['node1', 'node2', 'score', 'comments','edge_type','edge_source']
    #result_cols.append('edge_type')
    #result_cols.append('edge_source')
    
    return edges[result_cols]

def string_ppi(location):
    """
    Download and load STRING protein-protein interaction database.
    
    Returns:
        tuple: (ppi_df, protein_info_df)
    """
    # Download PPI data
    url = DATA_SOURCES["string_ppi"]["url"]
    filename = DATA_SOURCES["string_ppi"]["name"]
    destination = os.path.join(location, filename)
    filepath = download(url, destination)
    
    print(f"Loading STRING PPI data...")
    df_ppi = pd.read_csv(filepath, sep=" ", compression='gzip')
    
    # Download protein info (NEEDED for gene name mapping!)
    url_info = DATA_SOURCES["string_protein"]["url"]
    filename_info = DATA_SOURCES["string_protein"]["name"]
    destination_info = os.path.join(location, filename_info)
    filepath_info = download(url_info, destination_info)
    
    print(f"Loading STRING protein info...")
    df_info = pd.read_csv(filepath_info, sep="\t", compression='gzip')
    
    print(f"Loaded {len(df_ppi)} protein interactions")
    print(f"Loaded {len(df_info)} protein mappings")
    
    return df_ppi, df_info

def string_get_edges(
    sources: list[str],
    target: str = None,
    db: tuple[pd.DataFrame, pd.DataFrame] = None,
    location: str = None,
    include_type: bool = True,
    score_columns: list[str] = ["cooccurence","homology","coexpression", "coexpression_transferred", "experiments", "experiments_transferred","database", "database_transferred", "textmining", "textmining_transferred", "combined_score"
]
) -> pd.DataFrame:
    """
    Get edges from STRING database between sources and optionally to a target.
    
    Args:
        sources: List of gene symbols (e.g., ['TP53', 'BRCA1'])
        target: Optional target gene symbol
        db: Pre-loaded STRING DataFrames tuple (ppi_df, info_df)
        location: Directory where STRING data is stored
        include_type: Whether to add 'edge_type' column
        score_columns: List of score columns (default: ['combined_score'])
    
    Returns:
        DataFrame with columns: node1, node2, [score columns], [edge_type], edge_source
    """
    
    if db is None:
        if location is None:
            raise ValueError("Either db or location must be provided")
        db_ppi, db_info = string_ppi(location)
    else:
        db_ppi, db_info = db
    
    if score_columns is None:
        score_columns = ['combined_score']
    
    # Map gene symbols to STRING protein IDs (ENSP format)
    symbol_to_protein = db_info.set_index('preferred_name')['#string_protein_id'].to_dict()
    protein_to_symbol = db_info.set_index('#string_protein_id')['preferred_name'].to_dict()
    
    # Convert sources to STRING protein IDs
    sources_proteins = []
    for gene in sources:
        protein_id = symbol_to_protein.get(gene)
        if protein_id and pd.notna(protein_id):
            sources_proteins.append(protein_id)
        else:
            print(f"Warning: Could not find STRING protein ID for gene '{gene}'")
    
    # Convert target if provided
    target_protein = None
    if target:
        target_protein = symbol_to_protein.get(target)
        if not target_protein or pd.isna(target_protein):
            print(f"Warning: Could not find STRING protein ID for target '{target}'")
    
    if len(sources_proteins) == 0:
        cols = ['node1', 'node2'] + score_columns
        if include_type:
            cols.append('edge_type')
        cols.append('edge_source')
        return pd.DataFrame(columns=cols)
    
    sources_set = set(sources_proteins)
    all_edges = []
    
    # Get within-sources edges
    within_mask = (
        db_ppi['protein1'].isin(sources_set) &
        db_ppi['protein2'].isin(sources_set)
    )
    within_edges = db_ppi[within_mask].copy()
    if len(within_edges) > 0:
        if include_type:
            within_edges['edge_type'] = 'within_cluster'
        all_edges.append(within_edges)
    
    # Get sources-to-target edges
    if target_protein:
        target_mask = (
            (db_ppi['protein1'].isin(sources_set) & (db_ppi['protein2'] == target_protein)) |
            ((db_ppi['protein1'] == target_protein) & db_ppi['protein2'].isin(sources_set))
        )
        target_edges = db_ppi[target_mask].copy()
        if len(target_edges) > 0:
            if include_type:
                target_edges['edge_type'] = 'to_target'
            all_edges.append(target_edges)
    
    if len(all_edges) == 0:
        cols = ['node1', 'node2'] + score_columns
        if include_type:
            cols.append('edge_type')
        cols.append('edge_source')
        return pd.DataFrame(columns=cols)
    
    edges = pd.concat(all_edges, ignore_index=True)
    
    # Convert STRING protein IDs back to gene symbols
    edges['node1'] = edges['protein1'].map(protein_to_symbol).fillna(edges['protein1'])
    edges['node2'] = edges['protein2'].map(protein_to_symbol).fillna(edges['protein2'])
    
    edges['edge_source'] = 'string_ppi'
    
    # Select result columns
    result_cols = ['node1', 'node2'] + score_columns
    if include_type:
        result_cols.append('edge_type')
    result_cols.append('edge_source')
    
    return edges[result_cols]


def get_CollecTRI(location):
    """
    Download and load CollecTRI transcription factor regulatory network.
        
    Args:
        location: Directory where CollecTRI data should be stored
    
    Returns:
        pandas DataFrame 
    """
    filename = DATA_SOURCES["collectri"]["name"]  # Assumes this is in your config
    destination = os.path.join(location, filename)
    file_path = Path(destination)
    
    # Check if file exists locally
    if file_path.exists():
        print(f"Loading CollecTRI network from {file_path}...")
        edges = pd.read_csv(file_path, sep='\t')
        print(f"Loaded {len(edges)} regulatory interactions")
        return edges
    
    # File doesn't exist, download from decoupler
    print(f"Loading CollecTRI network from decoupler...")
    edges = dc.op.collectri(organism='human') #dc.get_collectri(organism='human', split_complexes=False)
    
    # Create parent directories if needed
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to local file
    print(f"Saving CollecTRI data to {file_path}...")
    edges.to_csv(file_path, sep='\t', index=False)
    
    print(f"Loaded {len(edges)} regulatory interactions")
    return edges



def get_CollecTRI_edges_to(
    sources: list[str],
    target: str,
    db: pd.DataFrame = None,
    location: str = None
) -> pd.DataFrame:
    """
    Get CollecTRI edges from sources to target gene.
    
    Args:
        sources: List of source gene symbols (TFs) (e.g., ['TP53', 'BRCA1'])
        target: Target gene symbol (e.g., 'MDM2')
        db: Pre-loaded CollecTRI DataFrame (if None, will load from decoupler)
        location: Directory where CollecTRI data is stored (required if db is None)
    
    Returns:
        DataFrame with columns: source, target, mor, reference, edge_source
    """
    if db is None:
        if location is None:
            raise ValueError("Either db or location must be provided")
        db = get_CollecTRI(location)
    
    sources_set = set(sources)
    
    # Filter edges where source is in sources list AND target matches
    mask = (db['source'].isin(sources_set)) & (db['target'] == target)
    edges = db[mask].copy()
    
    edges['edge_source'] = 'collectri'
    
    e = edges[['source', 'target', 'weight', 'resources', 'references','sign_decision']]
    e = e.rename(columns={"source":"node1","target":"node2"})
    return e
