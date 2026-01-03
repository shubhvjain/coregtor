from coregtor.clusters import identify_coregulators

import pandas as pd

data = pd.read_csv("tests/test_sim_matrix.csv")

# Extract gene names from FIRST ROW (skip first column)
genes = data.columns[1:].tolist()

# Extract numeric data (skip first row and first column)
data_fixed = data.iloc[1:, 1:].astype(float)

# Create square similarity matrix
m = pd.DataFrame(
    data_fixed.values,
    index=genes,
    columns=genes
)

# Set similarity flag
m.attrs['is_distance'] = False

print("Matrix ready:", m.shape)
print(m.head())

# Run clustering
result = identify_coregulators(
    sim_matrix=m,
    target_gene="TP53",
    method= "validation_index",
    method_options = {"index":"silhouette",
    "min_module_size":2}   
)

print(result)

print(result["clusters_df"])


result["clusters_df"].to_csv("tests/res1.csv") 
result["best_df"].to_csv("tests/res2.csv")