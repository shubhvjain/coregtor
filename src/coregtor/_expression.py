"""
Gene Expression Data Input/Output and Preprocessing Utilities
This module provides  utilities for reading, processing, and preparing gene expression data 
"""

import pandas as pd
from pathlib import Path
from typing import Union
import os

# Type aliases for better readability
PathLike = Union[str, Path]

def read(file_path:PathLike,tf_file_path:PathLike=None):
    """ Read Gene expression data from a file and prepare it for further processing. 
    
    This method reads the given file and generate the input for further analysis. The methods include many options for processing data from various formats. 
    
    Args:
        file_path (Path or str) : Path to the file. Valid file types : .gct
        tf_file_path (str or Path) :  (Optional) Path to file containing transcription factor gene names. Should be a single-column file with gene names (no header).
    Returns:
        pd.DataFrame : A pandas dataframe with columns are genes and rows are samples/cells. 

    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == ".gct":
      ge_data =  read_gct(file_path)
    else:
      print("Unknown file type:", ext)
    
    if tf_file_path:
      # transcription factor list provided filter the data 
      tf_file_path = Path(tf_file_path).expanduser()
      if not tf_file_path.exists():
          raise FileNotFoundError(f"Transcription factor list not found: {tf_file_path}")
        
      # Read transcription factor gene list
      tf_genes = pd.read_csv(tf_file_path, names=["gene_name"], header=None)["gene_name"]
        
      # Filter genes to TF list only
      ge_data = ge_data[ge_data.index.isin(tf_genes)]

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
            raise ValueError("GCT file must contain 'Name' and 'Description' columns")
        
        # remove Name column, rename Description to gene_name, set as index
        df = df.drop(columns=["Name"]).rename(columns={"Description": "gene_name"})
        df = df.set_index("gene_name")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error reading GCT file {file_path}: {str(e)}")



def create_model_input(raw_ge_data: pd.DataFrame,target_gene:str,tf_factors:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare gene expression data for training.

    This function splits the gene expression DataFrame into features (X) and target (Y) for supervised learning.
    
    Args:
      raw_ge_data (pd.DataFrame) : Gene expression data in samples x genes format 
      target_gene (str) :  Name of the target gene to predict. Must be present in raw_ge_data columns
      tf_factors (pd.DataFrame) :  A DataFrame containing transcription factor gene names. It must have a column named 'gene_name' listing the TF genes. This DataFrame is used to filter the input gene expression data. 
      
    Returns:
      tuple[pd.DataFrame, pd.DataFrame] :  X - Feature matrix (samples x genes) excluding target gene; Y - Target vector (samples x 1) containing only target gene expression
    
    """
    # Check if raw_ge_data is empty
    if raw_ge_data.empty:
        raise ValueError("The gene expression data (raw_ge_data) is empty.")

    # Check if target_gene is in raw_ge_data columns
    if target_gene not in raw_ge_data.columns:
        raise ValueError(f"Target gene '{target_gene}' not found in gene expression data columns.")

    # Extract the target vector Y
    Y = raw_ge_data[[target_gene]]

    # Get the list of TFs to keep
    tf_list = tf_factors['gene_name'].tolist()

    # Filter only TF columns present in raw_ge_data, excluding the target gene
    X_cols = [gene for gene in tf_list if gene in raw_ge_data.columns and gene != target_gene]
    X = raw_ge_data[X_cols]

    return X, Y
