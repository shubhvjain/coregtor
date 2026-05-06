# import os
from pathlib import Path
from datetime import datetime
import sqlite3
import json
import pandas as pd
from coregtor.workflow.util import get_protein_coding_genes, get_tflist, read_dataset


def setup_experiment(exp, config):
    # Construct the full path to the experiment folder
    exp_dir = Path(config["out_path"]) / exp["id"]
    temp_dir = Path(config["temp_path"]) / exp["id"]
    done_file = temp_dir / "init_done.txt"
    input_file = exp_dir / "input.json"
    db_file = exp_dir / "status.db"

    # Check if already initialized
    if done_file.exists():
        print("Exp folder already exists and is initialized")
        return

    """
    1. load the data 
    2. compute source and target list
    3. create status.db
    4. setup the folder structure
    5. create the input.json file in output folder
    """
    # Create folders if it doesn't exist
    exp_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    #print(exp)
    dataset = read_dataset(exp["dataset"], config)
    tf_list = get_tflist(config)

    input = {**exp}
    # source_list
    
    input["source_genes"] = get_filtered_genes(
        dataset, exp.get("sources"), tf_list,config)
    input["target_genes"] = get_filtered_genes(
        dataset, exp.get("targets"), tf_list,config)

    with open(input_file, "w") as f:
        json.dump(input, f)

    conn = sqlite3.connect(db_file)
    conn.execute("""
        CREATE TABLE genes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            gene        TEXT,
            status      TEXT DEFAULT 'pending',
            worker      TEXT,
            started_at  REAL,
            finished_at REAL,
            error       TEXT
        )
    """)
    targets = input["target_genes"]
    conn.executemany(
        "INSERT INTO genes (gene, status) VALUES (?, 'pending')",
        [(g,) for g in targets]
    )
    conn.commit()
    conn.close()

    Path(done_file).write_text(f"Init at {datetime.now()}")

    return


def get_filtered_genes(df, options, tflist, config):
    """
    Selects and filters genes from df based on type and stats.
    """
    #print(options)
    g_type = options.get("type", "all")

    # 1. Selection
    if g_type == "tf":
        pool = [g for g in tflist if g in df.columns]
    elif g_type == "no_tf":
        pool = [g for g in df.columns if g not in tflist]
    elif g_type == "pc_only":
        pool = get_protein_coding_genes(df.columns.tolist(),config)
    elif g_type == "custom":
        pool = [g for g in options.get("items", []) if g in df.columns]
    else:
        raise ValueError("Invalid selection type provided")

    if not pool:
        return []

    # 2. Thresholding (Non-zero occupancy)
    nz_thresh = options.get("non_zero_threshold")
    if nz_thresh is not None:
        occupancy = (df[pool] > 0).mean()
        pool = occupancy[occupancy >= nz_thresh].index.tolist()

    # 3. Ranking (Top N by Mean/Var)
    n = options.get("top_expressed_n")
    if n is not None and len(pool) > n:
        sub = df[pool]
        stats = pd.DataFrame({"m": sub.mean(), "v": sub.var()})
        pool = stats.sort_values(
            ["m", "v"], ascending=False).head(n).index.tolist()

    return pool
