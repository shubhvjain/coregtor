import os
from pathlib import Path
from datetime import datetime
import sqlite3
import pandas as pd

def get_exp_path(input,CONFIG):
    """
    Use this everywhere to get the path to the experiment results and temp folder. 
    output_path = env.EXP_OUTPUT_PATH/input
    r
    """
    if input.get("path",None) is not  None:
        output_path = Path(CONFIG.get("EXP_OUTPUT_PATH"))/ input.get("path")
        temp_path = Path(CONFIG.get("EXP_TEMP_PATH"))/ input.get("path")
    else:
        output_path = Path(CONFIG.get("EXP_OUTPUT_PATH"))/ input.get("id")
        temp_path = Path(CONFIG.get("EXP_TEMP_PATH"))/ input.get("id")
    return output_path,temp_path


def get_mappings(CONFIG, gene_list, source, target, batch_size=900):
    """
    Map gene identifiers from source format to target format.
    Assuming tfitpy setup was run and the source db file is available.
    Args:
        gene_list: List of identifiers to map
        source: Source attribute name in database (e.g., 'gene_name', 'gene_id')
        target: Target attribute name in database (e.g., 'gene_id', 'gene_name')
        batch_size: Number of items to process per query (default 900, under SQLite's 999 limit)
    Returns:
        Dictionary mapping {source_value: target_value, ...}
    """
    db_path = CONFIG["DATA_PATH"]/"gencode"/"gene_name_mapping.db"
    #print(db_path)
    #print(db_path.exists())
    con = sqlite3.connect(db_path)

    try:
        # Remove None/NaN and get unique values
        unique_values = [x for x in set(
            gene_list) if x is not None and pd.notna(x)]
        if len(unique_values) == 0:
            print("No values to map")
            return {}

        # Process in batches to avoid SQLite variable limit
        mapping_dict = {}
        for i in range(0, len(unique_values), batch_size):
            batch = unique_values[i:i + batch_size]
            placeholders = ','.join(['?'] * len(batch))
            query = f"""
                SELECT DISTINCT {source}, {target}
                FROM mappings
                WHERE {source} IN ({placeholders})
                AND {target} IS NOT NULL
            """

            # Execute query and update mapping dict
            mapping_df = pd.read_sql_query(query, con, params=batch)
            mapping_dict.update(
                dict(zip(mapping_df[source], mapping_df[target])))

        # Report statistics
        total = len(gene_list)
        unique_count = len(unique_values)
        mapped_count = len(mapping_dict)
        failed_count = unique_count - mapped_count
        null_in_results = sum(
            1 for gene in gene_list if gene not in mapping_dict)

        print(f"Mapping from {source} to {target}:")
        print(f"  Total input values: {total}")
        print(f"  Unique values: {unique_count}")
        print(f"  Successfully mapped: {mapped_count}")
        print(f"  Failed to map: {failed_count}")
        print(
            f"  Null results: {null_in_results} ({null_in_results/total*100:.2f}%)")

        return mapping_dict

    finally:
        con.close()

def get_tflist(CONFIG):
    """
    """
    tf_path = CONFIG["DATA_PATH"] / "tflist"/ "allTFs_hg38.txt"
    df = pd.read_csv(tf_path, names=["gene_name"], header=None)
    return  df["gene_name"].tolist()


def get_protein_coding_genes(gene_list, CONFIG):
    """
    Fetches all protein-coding genes once and intersects with input list in memory.
    Best for one-time calls with large input lists.
    """
    db_path = CONFIG["DATA_PATH"] / "gencode" / "gene_name_mapping.db"
    
    # 1. Fetch all protein-coding genes from the DB
    query = "SELECT DISTINCT gene_name FROM mappings WHERE gene_type = 'protein_coding'"
    
    try:
        with sqlite3.connect(db_path) as con:
            cursor = con.execute(query)
            pc_reference_set = {row[0] for row in cursor.fetchall()}
    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
        return []

    # 2. Use set intersection to find matches
    gene_list_set = set(gene_list)
    protein_coding = list(pc_reference_set.intersection(gene_list_set))
    
    return protein_coding

def read_dataset(details, config):
    """"""
    dtype = details.get("type", None)
    if dtype is None:
        raise ValueError("invalid dataset.type")

    #print(config.get("data_path"))
    if dtype == "gct":
        return read_gct(details.get("path"), config, convert_gene_names=details.get("convert_gene_names", True))


def read_gct(file_path, CONFIG=None, convert_gene_names=False):
    """
    Read Gene Cluster Text (GCT) format file into a pandas DataFrame.

    GCT is a tab-delimited format which include
    - Line 1: Version information
    - Line 2: Dimensions (genes x samples)  
    - Line 3+: Header with Name, Description, and sample columns
    - Data rows: Gene information and expression values 

    Assuming there is  "Description" column that has the name of genes have the gene names for each gene row.

    Args:
        file_path (str or Path) : Path to the GCT file.

    Returns:
        pd.DataFrame :  DataFrame with genes as rows and samples as columns. The index is gene_name (from Description column), columns are sample identifiers. Each cell had gene expression levels

    Notes:
    ------
    - Removes the 'Name' column (gene IDs) and uses 'Description' as gene names

    """

    file_path = Path(os.path.expandvars(file_path))
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Read GCT file, skipping version and dimension lines
        df = pd.read_csv(file_path, skiprows=2, sep="\t")
        gene = df["Name"].values.tolist()
        if convert_gene_names:
            if CONFIG is None:
                raise ValueError("data_path not provided")
            mps = get_mappings(CONFIG, gene_list=gene, source='gene_id',
                               target='gene_name')
            df["Name"] = df["Name"].map(mps)
        if "Name" not in df.columns or "Description" not in df.columns:
            raise ValueError(
                "GCT file must contain 'Name' and 'Description' columns")

        # remove Name column, rename Description to gene_name, set as index
        df = df.drop(columns=["Name"]).rename(
            columns={"Description": "gene_name"})
        df = df.set_index("gene_name")
        df = df.transpose().rename_axis("sample_name")
        return df
    except Exception as e:
        raise ValueError(f"Error reading GCT file {file_path}: {str(e)}")
