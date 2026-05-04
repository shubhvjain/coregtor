from datetime import datetime
import json
import sqlite3
import time 

from coregtor.workflow.util import read_dataset,get_tflist

from coregtor.pipeline import CoRegTorPipeline

def run_batch(exp, config, items=100, batch_id=None):
    """
    1. check if exp is initialized
    2. claim pending gene (check in status.db)
    3. initialize a new db for the batch
    4. get the required datasets 
    5. for each target:
        5.1. run the single target
        5.2  update the status in batch.db
    """
    if batch_id is None:
        batch_id = f"Batch-{datetime.now().strftime("%Y%m%d-%H%M%S")}"
        print(batch_id)

    out_path = config["out_path"] / exp["id"]
    temp_path = config["temp_path"] / exp["id"]

    db_file = out_path / "status.db"

    if not temp_path/ "init_check.txt":
        raise ValueError("Exp not yet initialized, run init first")
    
    targets = claim_pending_genes(db_file, batch_id, items)

    targets = [g for g in targets if g and g.strip()]
    
    if not targets:
        print("No pending genes, nothing to do")
        return

    print(f"Claimed {len(targets)} genes by batch:  {batch_id}")
    
    batch_db = temp_path / f"{batch_id}.db"
    init_batch_db(batch_db,targets)

    # get input.json
    input_data_path = out_path / "input.json"
    input_data = None
    with open(input_data_path,"r") as f:
        input_data = json.load(f)
    

    dataset = read_dataset(exp["dataset"],config)
    tf_list = get_tflist(config)

    pipeline_options = {
        **exp,
        "target_genes": input_data["target_genes"],
        "paths":{
            "temp": str(temp_path/"results"),
            "out": str(out_path)
        }
    }


    pipeline = CoRegTorPipeline(
        expression_data=dataset,
        tflist=tf_list,
        options=pipeline_options,
        exp_title=exp["id"]
    )
    print(f"Starting {batch_db} with {len(targets)} genes")
    for gene in targets:
        try:
            print(gene)
            pipeline.run_single_target(gene)
            mark_batch_gene_done(batch_db, gene)
            print("  done")
        except Exception as e:
            print("  error:", e)
            mark_batch_gene_failed(batch_db, gene, str(e))
    update_batch_complete(batch_db,db_file)
    print("done")




def claim_pending_genes(db_path, batch_id, batch_size = 500):
    STALE_TIMEOUT = 60 * 60 * 2

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")

    try:
        conn.execute("BEGIN IMMEDIATE")  # blocks other writers, allows readers

        # Reset stale claimed genes
        conn.execute(
            "UPDATE genes SET status='pending', worker=NULL WHERE status='claimed' AND started_at < ?",
            (time.time() - STALE_TIMEOUT,)
        )

        rows = conn.execute(
            "SELECT gene FROM genes WHERE status='pending' LIMIT ?",
            (batch_size,)
        ).fetchall()

        if not rows:
            conn.execute("COMMIT")
            return []

        genes = [r[0] for r in rows]
        conn.executemany(
            "UPDATE genes SET status='claimed', worker=?, started_at=? WHERE gene=? AND status='pending'",
            [(batch_id, time.time(), g) for g in genes]
        )
        conn.execute("COMMIT")

    except sqlite3.OperationalError:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    return genes  # no need for re-query, BEGIN IMMEDIATE guarantees exclusivity


def init_batch_db(db_path,targets):
    """
    generate a new database for the batch being run 
    """
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS genes (
            gene TEXT PRIMARY KEY,
            status TEXT DEFAULT 'pending',
            finished_at REAL,
            error TEXT
        )
    """)
    conn.executemany(
        "INSERT OR IGNORE INTO genes (gene, status) VALUES (?, 'pending')",
        [(g,) for g in targets]
    )
    conn.commit()
    conn.close()
    

def mark_batch_gene_done(batch_db, gene):
    conn = sqlite3.connect(batch_db)
    conn.execute(
        "UPDATE genes SET status='done', finished_at=? WHERE gene=?",
        (time.time(), gene)
    )
    conn.commit()
    conn.close()


def mark_batch_gene_failed(batch_db, gene, error):
    conn = sqlite3.connect(batch_db)
    conn.execute(
        "UPDATE genes SET status='failed', finished_at=?, error=? WHERE gene=?",
        (time.time(), error[:500], gene)
    )
    conn.commit()
    conn.close()


def update_batch_complete(batch_db, db_path, retries: int = 5):
    wconn = sqlite3.connect(batch_db, timeout=10)
    rows = wconn.execute(
        "SELECT gene, status, error FROM genes WHERE status IN ('done', 'failed')"
    ).fetchall()
    wconn.close()

    done = [(g,) for g, s, _ in rows if s == "done"]
    failed = [(e, g) for g, s, e in rows if s == "failed"]

    for attempt in range(retries):
        try:
            conn = sqlite3.connect(db_path, timeout=60)
            conn.execute("PRAGMA journal_mode=WAL")
            if done:
                conn.executemany(
                    "UPDATE genes SET status='done', finished_at=? WHERE gene=?",
                    [(time.time(), g) for (g,) in done]
                )
            if failed:
                conn.executemany(
                    "UPDATE genes SET status='failed', finished_at=?, error=? WHERE gene=?",
                    [(time.time(), (e or "")[:500], g) for e, g in failed]
                )
            conn.commit()
            conn.close()
            batch_db.unlink()
            return
        except Exception as e:
            print(f"  flush attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(2 ** attempt)

    print(f"WARNING: flush failed after {retries} attempts. Worker DB kept at {batch_db}")
    print("Run 'consolidate' to retry later")