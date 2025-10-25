
import pandas as pd
from pathlib import Path
from typing import Union
import os

# Type aliases for better readability
PathLike = Union[str, Path]

def read_GE_data(file_path:PathLike):
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
      ge_data =  read_gct(file_path)
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
            raise ValueError("GCT file must contain 'Name' and 'Description' columns")
        
        # remove Name column, rename Description to gene_name, set as index
        df = df.drop(columns=["Name"]).rename(columns={"Description": "gene_name"})
        df = df.set_index("gene_name")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error reading GCT file {file_path}: {str(e)}")

