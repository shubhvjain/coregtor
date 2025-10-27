"""
this module helps in running experiments. this is not part of the main pipeline
"""


from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
from typing import Union
import os
import json
from datetime import datetime
import time
import numpy as np


# Type aliases for better readability
PathLike = Union[str, Path]


def read_GE_data(file_path: PathLike):
    """ Read Gene expression data from a file and prepare it for further processing. 

    This method reads the given file and generate the input for further analysis. The methods include many options for processing data from various formats. 

    Args:
        file_path (Path or str) : Path to the file. Valid file types : .gct
    Returns:
        pd.DataFrame : A pandas dataframe with columns are genes and rows are samples/cells. 

    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".gct":
        ge_data = read_gct(file_path)
    else:
        print("Unknown file type:", ext)

    # if tf_file_path:
    #   # transcription factor list provided filter the data
    #   tf_file_path = Path(tf_file_path).expanduser()
    #   if not tf_file_path.exists():
    #       raise FileNotFoundError(f"Transcription factor list not found: {tf_file_path}")

    #   # Read transcription factor gene list
    #   tf_genes = pd.read_csv(tf_file_path, names=["gene_name"], header=None)["gene_name"]

    #    # Filter genes to TF list only
    #  ge_data = ge_data[ge_data.index.isin(tf_genes)]

    ge_data = ge_data.transpose().rename_axis("sample_name")
    return ge_data


def read_gct(file_path: PathLike) -> pd.DataFrame:
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
    file_path = Path(file_path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Read GCT file, skipping version and dimension lines
        df = pd.read_csv(file_path, skiprows=2, sep="\t")

        # Validate required columns
        if "Name" not in df.columns or "Description" not in df.columns:
            raise ValueError(
                "GCT file must contain 'Name' and 'Description' columns")

        # remove Name column, rename Description to gene_name, set as index
        df = df.drop(columns=["Name"]).rename(
            columns={"Description": "gene_name"})
        df = df.set_index("gene_name")

        return df

    except Exception as e:
        raise ValueError(f"Error reading GCT file {file_path}: {str(e)}")


def save_exp_results(
    options,
    input_config,
    sim_matrix,
    results,
    runtimes,
    base_path=None,
    output_subdir="experiments"
):
    """
    Save experiment results to JSON file

    Parameters:
    -----------
    options : dict
        Experiment configuration options (including notes)
    input_config : dict
        Dictionary with data_file_path, data_description, tf_file_path
    sim_matrix : pandas.DataFrame
        Similarity matrix from analysis
    results : pandas.DataFrame
        Results from identify_coregulators
    base_path : Path, optional
        Base directory path (uses current directory if None)
    output_subdir : str
        Subdirectory name within base_path (default: "experiments")

    Returns:
    --------
    str : Path to saved file
    """

    # Create output directory inside base_path
    if base_path is None:
        base_path = Path(".")

    output_path = Path(base_path) / output_subdir
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate experiment ID and timestamp
    expid = int(time.time())
    file_created_on = datetime.now().isoformat()

    # Convert all Path objects in input_config to strings
    input_serialized = {}
    for key, value in input_config.items():
        if isinstance(value, Path):
            input_serialized[key] = str(value)
        else:
            input_serialized[key] = value

    # Prepare the experiment data
    exp_data = {
        "expid": expid,
        "title": f"Coregulators for gene {options['target_gene']}",
        "file_created_on": file_created_on,
        "options": options,
        "input": input_serialized,
        "runtimes": runtimes,
        "similarity_matrix": None,
        "results": None
    }

    # Convert similarity matrix to list
    if isinstance(sim_matrix, pd.DataFrame):
        exp_data["similarity_matrix"] = {
            "data": sim_matrix.values.tolist(),
            "index": sim_matrix.index.tolist(),
            "columns": sim_matrix.columns.tolist()
        }

    # Convert results to serializable format
    if isinstance(results, pd.DataFrame):
        exp_data["results"] = results.to_dict(orient='records')

    # Save to JSON file
    output_file = output_path / f"exp_{expid}_{options['target_gene']}.json"
    with open(output_file, 'w') as f:
        json.dump(exp_data, f, indent=2)

    print(f"Experiment saved successfully!")
    print(f"File: {output_file}")
    print(f"Experiment ID: {expid}")
    print(f"Target gene: {options['target_gene']}")

    return str(output_file)


def load_exp_file(file_path):
    """
    Load experiment results from JSON file

    Parameters:
    -----------
    file_path : str or Path
        Path to the experiment JSON file

    Returns:
    --------
    dict : Dictionary containing all experiment data with keys:
        - expid: Experiment ID
        - title: Experiment title
        - file_created_on: Creation timestamp
        - options: Experiment configuration (including notes)
        - input: Input file paths and descriptions
        - similarity_matrix: Reconstructed similarity matrix as DataFrame
        - results: Results as DataFrame
    """

    with open(file_path, 'r') as f:
        exp_data = json.load(f)

    # Reconstruct similarity matrix as DataFrame
    if exp_data["similarity_matrix"] and "data" in exp_data["similarity_matrix"]:
        sim_data = exp_data["similarity_matrix"]["data"]
        sim_index = exp_data["similarity_matrix"]["index"]
        sim_columns = exp_data["similarity_matrix"]["columns"]

        exp_data["similarity_matrix"] = pd.DataFrame(
            sim_data,
            index=sim_index,
            columns=sim_columns
        )

    # Convert results back to DataFrame
    if exp_data["results"] and isinstance(exp_data["results"], list):
        if len(exp_data["results"]) > 0:
            exp_data["results"] = pd.DataFrame(exp_data["results"])

    print(f"Experiment loaded successfully!")
    print(f"Title: {exp_data['title']}")
    print(f"Experiment ID: {exp_data['expid']}")
    print(f"Created on: {exp_data['file_created_on']}")
    print(f"Target gene: {exp_data['options']['target_gene']}")

    return exp_data


# Default configuration for experiments
DEFAULT_CONFIG = {
    "model_type": "rf",
    "model_options": {
        "n_estimators": 1000,
        "max_depth": 5,
        "random_state": 123
    },
    "create_context_method": "tree_paths",
    "transform_context_method": "gene_frequency",
    "context_similarity_method": "cosine"
}


def generate_tables_from_experiments(
    experiments_folder: Path,
    output_folder: Optional[Path] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate experiments and clusters tables from JSON experiment files.

    Parameters:
    -----------
    experiments_folder : Path
        Folder containing exp_*.json files
    output_folder : Path, optional
        Folder to save CSV files (if None, doesn't save)

    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        (experiments_df, clusters_df)
    """

    experiments_folder = Path(experiments_folder)
    exp_files = list(experiments_folder.glob("exp_*.json"))

    if not exp_files:
        raise ValueError(f"No experiment files found in {experiments_folder}")

    experiments_data = []
    clusters_data = []

    for exp_file in exp_files:
        with open(exp_file, 'r') as f:
            exp = json.load(f)

        # Extract experiment-level data
        exp_row = _extract_experiment_row(exp)
        experiments_data.append(exp_row)

        # Extract cluster-level data
        cluster_rows = _extract_cluster_rows(exp)
        clusters_data.extend(cluster_rows)

    # Create DataFrames
    experiments_df = pd.DataFrame(experiments_data)
    clusters_df = pd.DataFrame(clusters_data)

    # Sort by expid
    experiments_df = experiments_df.sort_values('expid').reset_index(drop=True)
    clusters_df = clusters_df.sort_values(
        ['expid', 'cluster_id']).reset_index(drop=True)

    # Save to CSV if output folder specified
    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        experiments_df.to_csv(
            output_folder / "experiment_summary.csv", index=False)
        clusters_df.to_csv(output_folder / "cluster_summary.csv", index=False)
        print(f"Tables saved to {output_folder}")

    return experiments_df, clusters_df


def _extract_experiment_row(exp: Dict) -> Dict:
    """Extract experiment-level data from JSON."""

    options = exp.get('options', {})
    input_config = exp.get('input', {})
    runtimes = exp.get('runtimes', {})
    results = exp.get('results', [])

    # Get model options with defaults
    model_type = options.get('model_type', DEFAULT_CONFIG['model_type'])
    model_options = options.get('model_options', {})
    default_model_options = DEFAULT_CONFIG['model_options']

    # Build clusters summary string
    clusters_summary = _build_clusters_summary(results)

    # Calculate aggregate metrics
    n_clusters = len(results)

    runtime_keys = [
        'create_model_input', 'create_model', 'tree_paths',
        'create_context', 'transform_context', 'compare_context',
        'identify_coregulators'
    ]
    total_runtime = sum(runtimes.get(k, 0)
                        for k in runtime_keys if runtimes.get(k) is not None)

    row = {
        # Metadata
        'expid': exp.get('expid'),
        'file_created_on': exp.get('file_created_on'),
        'target_gene': options.get('target_gene'),

        # Input configuration
        'input_data_file_path': input_config.get('data_file_path', ''),
        'input_data_description': input_config.get('data_description', ''),
        'input_tf_file_path': input_config.get('tf_file_path', ''),

        # Model options with defaults
        'model_type': model_type,
        'model_n_estimators': model_options.get('n_estimators', default_model_options['n_estimators']),
        'model_max_depth': model_options.get('max_depth', default_model_options['max_depth']),
        'model_random_state': model_options.get('random_state', default_model_options['random_state']),

        # Pipeline options with defaults
        'create_context_method': options.get('create_context_method', DEFAULT_CONFIG['create_context_method']),
        'transform_context_method': options.get('transform_context_method', DEFAULT_CONFIG['transform_context_method']),
        'context_similarity_method': options.get('context_similarity_method', DEFAULT_CONFIG['context_similarity_method']),
        'selected_threshold': options.get('selected_threshold'),

        # Results summary
        'n_clusters': n_clusters,
        'clusters_summary': clusters_summary,

        # Runtimes
        'runtime_total_pipeline': round(total_runtime, 4) if total_runtime > 0 else None,
        'runtime_create_model_input': round(runtimes.get('create_model_input'), 4) if runtimes.get('create_model_input') is not None else None,
        'runtime_create_model': round(runtimes.get('create_model'), 4) if runtimes.get('create_model') is not None else None,
        'runtime_tree_paths': round(runtimes.get('tree_paths'), 4) if runtimes.get('tree_paths') is not None else None,
        'runtime_create_context': round(runtimes.get('create_context'), 4) if runtimes.get('create_context') is not None else None,
        'runtime_transform_context': round(runtimes.get('transform_context'), 4) if runtimes.get('transform_context') is not None else None,
        'runtime_compare_context': round(runtimes.get('compare_context'), 4) if runtimes.get('compare_context') is not None else None,
        'runtime_identify_coregulators': round(runtimes.get('identify_coregulators'), 4) if runtimes.get('identify_coregulators') is not None else None,

        # Notes
        'notes': options.get('notes', '')
    }

    return row


def _extract_cluster_rows(exp: Dict) -> List[Dict]:
    expid = exp.get('expid')
    target_gene = exp.get('options', {}).get('target_gene')
    file_created_on = exp.get('file_created_on')
    selected_threshold = exp.get('options', {}).get('selected_threshold')
    input_data_file_path = exp.get('input', {}).get('data_file_path', '')
    results = exp.get('results', [])
    
    cluster_rows = []
    
    for result in results:
        row = {
            'expid': expid,
            'target_gene': target_gene,
            'file_created_on': file_created_on,
            'input_data_file_path': input_data_file_path,
            'selected_threshold': selected_threshold,
            'cluster_id': result.get('cluster_id'),
            'n_genes': result.get('n_genes'),
            'gene_cluster': result.get('gene_cluster', '')
        }
        cluster_rows.append(row)
    
    return cluster_rows



def _build_clusters_summary(results: List[Dict]) -> str:
    """Build clusters summary string in format: '0:gene1,gene2;1:gene3,gene4'"""

    if not results:
        return ""

    summary_parts = []
    for result in results:
        cluster_id = result.get('cluster_id', '')
        genes = result.get('gene_cluster', '')
        summary_parts.append(f"{cluster_id}:{genes}")

    return ";".join(summary_parts)


def get_default_config() -> Dict:
    """Return a copy of the default configuration."""
    return DEFAULT_CONFIG.copy()
