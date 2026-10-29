"""
CoRegTor CLI: all the logic behind the `coregtor` commands.
(cli.py only parses arguments and calls the public functions here.)

TWO WAYS TO USE THE TOOL
------------------------
A) Many targets, resumable, can be split over several jobs:

       coregtor bulk   ...   -> step 1: train models, save one temp file per target
       coregtor status ...   -> check how far step 1 is
       coregtor result ...   -> step 2: turn the temp files into the final CSV

   Step 1 is slow (model training), step 2 is fast. Keeping the temp files
   means you can later produce more results (different distance / clustering
   settings) without retraining anything.

B) Few targets, you only want the final file:

       coregtor run ...      -> does everything in memory, writes ONLY the
                                final CSV. No temp files at all.

WHAT THE INPUT FILE LOOKS LIKE
------------------------------
The data is expected to be already prepared (no filtering / gene mapping here):

    {
      "id": "exp1",
      "inputs": {
        "data": "expr.csv", # genes = columns, first column = sample id
        "source": "sources.txt", # one source (regulator) gene per line
        "target_file": "targets.txt" # or  "target_list": ["GENE1", "GENE2"]
      },
      "results": [{"id": "res1", "distance": "cosine_distance", "method": "hierarchical"}]
    }

The full format is the SCHEMA in coregtor/workflow/pipeline.py.

WHAT GETS WRITTEN WHERE  (mode A)
---------------------------------
    <temp>/<exp id>/
        status.db        the work queue: one row per target gene
        Batch-XXXX.db    scratch db of one running batch (deleted when merged)
        <gene>.pkl       per-target checkpoint written by the pipeline
    <output>/<exp id>/
        result_<name>.csv    the final result

HOW THE WORK QUEUE WORKS  (mode A)
----------------------------------
Every target is one row in status.db with a status:

    pending --(bulk claims it)--> claimed --> done
                                          \\-> failed

`bulk` can run several times at once (e.g. a cluster job array). Each call
claims its own genes inside one locked transaction, so two calls never get the
same gene. If a call dies, its genes stay "claimed"; after STALE_TIMEOUT they
are handed out again.
"""
import json
import os
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

# A "claimed" gene whose batch never reported back within this duration is assumed to belong to a crashed job and goes back to "pending".
STALE_TIMEOUT = 60 * 60 * 2  # 2 hours

# A batch db touched more recently than this is assumed to belong to a running job.
ACTIVE_BATCH_AGE = 60 * 10  # 10 minutes

STATUS_DB = "status.db"

# Default fallback directories if --output or --temp are omitted.
DEFAULT_OUTPUT = "~/coregtor/results"
DEFAULT_TEMP = "~/coregtor/temp"


class CoregtorCLIError(Exception):
    """A problem the user can fix (bad input file, missing step, ...).

    cli.py prints these as a short message instead of a Python traceback.
    """


# ==================================================
# Path Resolution & Setup Helpers
# ==================================================

@contextmanager
def db_connection(db_path, timeout=30, wal=False):
    """Context manager for SQLite connections ensuring clean resource cleanup."""
    conn = sqlite3.connect(db_path, timeout=timeout)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")  # WAL mode: readers don't block writers
    try:
        yield conn
    finally:
        conn.close()

def expand_path(value):
    """Expand environment variables and user home directory (~) in a path."""
    text = os.path.expandvars(str(value))
    if "$" in text:  # expandvars leaves unknown variables untouched
        raise CoregtorCLIError(f"Path '{value}' uses an unset environment variable.")
    return Path(text).expanduser()


def load_input(path):
    """Read and parse the experiment JSON definition, resolving relative input paths."""
    path = expand_path(path)
    if not path.exists():
        raise CoregtorCLIError(f"Input file not found: {path}")

    try:
        exp = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise CoregtorCLIError(f"Input file is not valid JSON: {e}") from e

    # Make data file paths absolute, relative to the folder of the input JSON file.
    # (Missing or wrong fields are reported later by the pipeline's schema validation.)
    inputs = exp.get("inputs", {})
    for key in ("data", "source", "target_file"):
        value = inputs.get(key)
        if value:
            p = expand_path(value)
            inputs[key] = str(p if p.is_absolute() else (path.parent / p).resolve())

    return exp


def resolve_bases(output, temp):
    """Return expanded output and temp base folders using defaults if unset."""
    return expand_path(output or DEFAULT_OUTPUT), expand_path(temp or DEFAULT_TEMP)


def get_paths(exp, output, temp):
    """Return (output_dir, temp_dir) tailored to this specific experiment ID."""
    out_base, temp_base = resolve_bases(output, temp)
    return out_base / exp["id"], temp_base / exp["id"]


def check_result_name(exp, name):
    """Verify that `name` is one of the experiment's result ids."""
    # "default" is the result the schema adds when the input file has no "results" key
    results = exp.get("results", [{"id": "default"}])
    names = [r["id"] for r in results]
    if name not in names:
        raise CoregtorCLIError(f"Result '{name}' not in input file. Available: {names}")


def make_pipeline(exp, output, temp, **run_opts):
    """Instantiate and validate a CoRegTorBulkPipeline object."""
    # Imported here so quick commands (like `status`) don't load the heavy dependencies
    from coregtor.workflow.pipeline import CoRegTorBulkPipeline

    out_base, temp_base = resolve_bases(output, temp)
    run_config = {
        "temp_path": str(temp_base),
        "output_path": str(out_base),
        **run_opts,
    }
    try:
        return CoRegTorBulkPipeline(exp, run_config)
    except Exception as e:
        raise CoregtorCLIError(f"Could not set up pipeline: {e}") from e


# ============================================
#  status.db: the shared work queue (mode A)
# ============================================

def init_status_db(db_path, targets):
    """Create status.db with every target marked 'pending'.

    Returns True if this call created the database, False if it already existed
    (including when another job created it at the same moment).
    """
    db_path = Path(db_path)
    if db_path.exists():
        return False

    # Build the database under a name unique to this process, then publish it
    # under its real name. os.link fails if the real name is already taken,
    # so only one process can ever win.
    tmp = db_path.with_name(f"{db_path.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)

    try:
        with db_connection(tmp) as conn:
            conn.execute("""
                CREATE TABLE genes (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    gene        TEXT UNIQUE,
                    status      TEXT DEFAULT 'pending',  -- pending | claimed | done | failed
                    worker      TEXT,
                    started_at  REAL,
                    finished_at REAL,
                    error       TEXT
                )
            """)
            conn.executemany(
                "INSERT INTO genes (gene, status) VALUES (?, 'pending')",
                [(g,) for g in targets],
            )
            conn.commit()

        try:
            os.link(tmp, db_path)
            return True
        except FileExistsError:
            return False
    finally:
        tmp.unlink(missing_ok=True)


def check_targets_match(db_path, targets):
    """Ensure target genes in the input file match targets recorded in status.db."""
    with db_connection(db_path) as conn:
        in_db = {r[0] for r in conn.execute("SELECT gene FROM genes")}

    if in_db != set(targets):
        raise CoregtorCLIError(
            f"Targets in input file ({len(set(targets))}) differ from status.db ({len(in_db)}). "
            "Delete experiment temp folder to reset."
        )


def claim_pending_genes(db_path, batch_id, batch_size):
    """Claim up to `batch_size` pending targets under an exclusive lock transaction.
    
    Releases claims older than STALE_TIMEOUT before allocating new genes.
    """
    with db_connection(db_path, wal=True) as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")  # Acquire write lock immediately
            
            # 1. Reset stale claims from dead/crashed workers
            conn.execute(
                "UPDATE genes SET status='pending', worker=NULL "
                "WHERE status='claimed' AND started_at < ?",
                (time.time() - STALE_TIMEOUT,),
            )

            # 2. Select target genes
            genes = [
                r[0]
                for r in conn.execute(
                    "SELECT gene FROM genes WHERE status='pending' LIMIT ?", (batch_size,)
                )
            ]

            # 3. Mark selected targets as claimed
            if genes:
                now = time.time()
                conn.executemany(
                    "UPDATE genes SET status='claimed', worker=?, started_at=? "
                    "WHERE gene=? AND status='pending'",
                    [(batch_id, now, g) for g in genes],
                )
            conn.execute("COMMIT")
            return genes
        except Exception:
            conn.execute("ROLLBACK")
            raise


def init_batch_db(db_path, targets):
    """Create a private batch DB to track active worker progress locally without lock contention."""
    with db_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS genes (
                gene TEXT PRIMARY KEY,
                status TEXT DEFAULT 'pending',  -- pending | done | failed
                finished_at REAL,
                error TEXT
            )
        """)
        conn.executemany(
            "INSERT OR IGNORE INTO genes (gene, status) VALUES (?, 'pending')",
            [(g,) for g in targets],
        )
        conn.commit()


def mark_batch_gene(batch_db, gene, status, error=None):
    """Record completion or failure for a single gene in the private batch database."""
    with db_connection(batch_db) as conn:
        conn.execute(
            "UPDATE genes SET status=?, finished_at=?, error=? WHERE gene=?",
            (status, time.time(), (error or "")[:500] or None, gene),
        )
        conn.commit()


def flush_batch(batch_db, status_db, retries=5):
    """Merge completed batch DB results into central status.db, then delete batch DB."""
    batch_db = Path(batch_db)

    # Read local accomplishments
    with db_connection(batch_db, timeout=10) as bconn:
        rows = bconn.execute(
            "SELECT gene, status, error FROM genes WHERE status IN ('done', 'failed')"
        ).fetchall()

    now = time.time()
    done = [(now, g) for g, s, _ in rows if s == "done"]
    failed = [(now, (e or "")[:500], g) for g, s, e in rows if s == "failed"]

    # Write results to shared status.db with retry logic on database lock conflicts
    for attempt in range(retries):
        try:
            with db_connection(status_db, timeout=60, wal=True) as conn:
                conn.executemany(
                    "UPDATE genes SET status='done', finished_at=? WHERE gene=?", done
                )
                conn.executemany(
                    "UPDATE genes SET status='failed', finished_at=?, error=? WHERE gene=?",
                    failed,
                )
                conn.commit()
            batch_db.unlink()  # Unlink batch database after successful sync
            return True
        except Exception as e:
            print(f"  could not update status.db (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(2**attempt)

    print(f"WARNING: could not update status.db. Batch db kept at {batch_db}.")
    print("Run 'coregtor consolidate' later to retry.")
    return False


def _status_counts(db_path):
    """Fetch status summary counts (e.g., {'pending': 10, 'done': 90})."""
    with db_connection(db_path) as conn:
        return dict(conn.execute("SELECT status, COUNT(*) FROM genes GROUP BY status"))


# ============================================
# 3. Commands (these are what cli.py calls)
# ============================================
def bulk(exp, output, temp, batch_size=100, batch_id=None, rerun=False):
    """`coregtor bulk`: process one batch of targets and save temp files.

    Steps:
      1. Build the pipeline (loads + validates the data).
      2. First call ever: create status.db with all targets as 'pending'.
         Later calls: check the targets still match.
      3. Claim up to `batch_size` pending targets.
      4. For each one, train the model and save its temp file (<gene>.pkl).
         Progress is recorded in the batch's private db.
      5. Merge the batch db into status.db (also when interrupted with Ctrl-C).

    Run it again (or start more jobs) until `status` shows nothing pending.
    `rerun=True` regenerates temp files even if they already exist.
    """
    # 1
    pipeline = make_pipeline(exp, output, temp, rerun=rerun)
    _, temp_dir = get_paths(exp, output, temp)
    status_db = temp_dir / STATUS_DB

    # 2
    if init_status_db(status_db, pipeline.targets):
        print(f"Created {STATUS_DB} with {len(pipeline.targets)} targets")
    else:
        check_targets_match(status_db, pipeline.targets)

    # 3. the batch id marks which batch owns which genes in status.db
    batch_id = batch_id or f"Batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    claimed = claim_pending_genes(status_db, batch_id, batch_size)
    if not claimed:
        print("No pending genes, nothing to do")
        return
    print(f"{batch_id}: claimed {len(claimed)} genes")

    batch_db = temp_dir / f"{batch_id}.db"
    init_batch_db(batch_db, claimed)

    # 4 + 5
    try:
        for i, gene in enumerate(claimed, 1):
            print(f"[{i}/{len(claimed)}] {gene}", end=" ... ", flush=True)

            gene_status, error = "done", None
            try:
                # True on success, False if the model step failed
                if not pipeline.process_single_target_temp(gene):
                    gene_status, error = "failed", "model/context step failed"
            except Exception as e:  # one bad gene must not stop the batch
                gene_status, error = "failed", str(e)

            mark_batch_gene(batch_db, gene, gene_status, error)
            print(gene_status if error is None else f"{gene_status}: {error}")
    finally:
        flush_batch(batch_db, status_db)

    print("Batch finished")


def consolidate(exp, output, temp):
    """`coregtor consolidate`: merge leftover batch dbs into status.db.

    Batch dbs that were modified very recently are skipped, because they
    probably belong to a job that is still running.
    """
    _, temp_dir = get_paths(exp, output, temp)
    status_db = temp_dir / STATUS_DB
    if not status_db.exists():
        raise CoregtorCLIError("Nothing to consolidate, run 'coregtor bulk' first")

    leftovers = sorted(temp_dir.glob("Batch-*.db"))
    if not leftovers:
        print("Nothing to consolidate")
        return

    for db in leftovers:
        age = time.time() - db.stat().st_mtime
        if age < ACTIVE_BATCH_AGE:
            print(f"Skipping {db.name} (changed {age:.0f}s ago, probably still running)")
            continue
        print(f"Merging {db.name}")
        flush_batch(db, status_db)


def status(exp, output, temp, name=None):
    """Execute `coregtor status`: Display execution progress and metrics."""
    out_dir, temp_dir = get_paths(exp, output, temp)
    status_db = temp_dir / STATUS_DB
    if not status_db.exists():
        raise CoregtorCLIError("No status found, run 'coregtor bulk' first")

    counts = _status_counts(status_db)
    total = sum(counts.values())
    print(f"Experiment: {exp['id']}")

    for s in ("pending", "claimed", "done", "failed"):
        n = counts.get(s, 0)
        pct = f"{n / total * 100:5.1f}%" if total else "  -  "
        print(f"  {s:<8} {n:>7}  {pct}")
    print(f"  {'total':<8} {total:>7}")

    with db_connection(status_db) as conn:
        stale = conn.execute(
            "SELECT COUNT(*) FROM genes WHERE status='claimed' AND started_at < ?",
            (time.time() - STALE_TIMEOUT,),
        ).fetchone()[0]
        errors = conn.execute(
            "SELECT gene, error FROM genes WHERE status='failed' LIMIT 5"
        ).fetchall()

    if stale:
        print(f"  {stale} claimed genes look abandoned; next bulk call will re-queue them")

    leftovers = list(temp_dir.glob("Batch-*.db"))
    if leftovers:
        print(f"  {len(leftovers)} batch db(s) not merged yet, run 'coregtor consolidate'")

    for gene, err in errors:
        print(f"  failed: {gene}: {err}")

    if name:
        check_result_name(exp, name)
        out_file = out_dir / f"result_{name}.csv"
        if out_file.exists():
            print(f"\nResult '{name}': generated -> {out_file}")
        else:
            print(
                f"\nResult '{name}': not generated yet ({counts.get('done', 0)} genes ready)"
            )


def result(exp, output, temp, name, rerun=False, jobs=None):
    """`coregtor result`: build the final CSV from the temp files made by `bulk`.

    Reads each target's temp file, computes the distance matrix, clusters it, and combines all targets into <output>/<exp id>/result_<name>.csv.
    Targets whose temp file is missing or failed are skipped. `rerun=True` overwrites an existing result file. `jobs` is the number of parallel workers for this step.
    """
    check_result_name(exp, name)
    _, temp_dir = get_paths(exp, output, temp)

    status_db = temp_dir / STATUS_DB
    if status_db.exists():
        counts = _status_counts(status_db)
        unfinished = counts.get("pending", 0) + counts.get("claimed", 0)
        if unfinished:
            print(
                f"Warning: {unfinished} genes are not processed yet; "
                f"the result will cover only the {counts.get('done', 0)} finished ones"
            )

    opts = {"rerun": rerun}
    if jobs:
        opts["n_jobs"] = jobs
    pipeline = make_pipeline(exp, output, temp, **opts)

    out_file = pipeline.output_dir / f"result_{name}.csv"
    if out_file.exists() and not rerun:
        print(f"{out_file} already exists, use --rerun to overwrite")
        return

    print(f"Generating result '{name}'...")
    pipeline.generate_results(result_id=name)

    if out_file.exists():
        print(f"Done.. Result saved at {out_file}")
    else:
        raise CoregtorCLIError("No result file written (missing or failed temp files).")


def run(exp, output, name, rerun=False):
    """`coregtor run`: compute one result start to finish, no temp files.

    For each target the pipeline calls RunCoRegTor and keeps only the small
    cluster table. At the end all tables are combined into
    <output>/<exp id>/result_<name>.csv.

    The trade-off compared to bulk + result: nothing is saved, so a crash or
    Ctrl-C loses all progress, and a different result needs a full recompute.
    Good for a handful of targets; use `bulk` for many.
    """
    check_result_name(exp, name)

    from coregtor.workflow.pipeline import CoRegTorRunPipeline
    run_config = {"output_path": str(expand_path(output or DEFAULT_OUTPUT))}
    pipeline = CoRegTorRunPipeline(exp, run_config)

    out_file = pipeline.result_file(name)
    if out_file.exists() and not rerun:
        print(f"{out_file} already exists, use --rerun to overwrite")
        return

    res_config = pipeline.get_result_config(name)
    tables, failed = [], []

    for i, gene in enumerate(pipeline.targets, 1):
        print(f"[{i}/{len(pipeline.targets)}] {gene}", end=" ... ", flush=True)
        try:
            clusters = pipeline.run_target(gene, res_config)
        except Exception as e:  # one bad target must not stop the others
            failed.append(gene)
            print(f"error: {e}")
            continue

        if clusters is None:
            failed.append(gene)
            print("no clusters")
        else:
            tables.append(clusters)
            print("done")

    if failed:
        shown = ", ".join(failed[:10]) + (" ..." if len(failed) > 10 else "")
        print(f"{len(failed)} targets gave no result: {shown}")
    if not tables:
        raise CoregtorCLIError("No result file was written (every target failed)")

    pipeline.save_result(name, tables)
    print(f"Done. Result saved at {out_file}")