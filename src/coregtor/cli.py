#!/usr/bin/env python3
"""CoRegTor CLI."""

import argparse
from pathlib import Path
import sys
import os
from coregtor.pipeline import CoRegTorPipeline
from coregtor.utils.exp import read_GE_data
import subprocess
import sys
import argparse
import sys
import pandas as pd

def run_bulk(extra_args):
    """
    access to snakemake workflow for bulk processing
    """
    snakefile = Path(__file__).parent / "workflow" / "Snakefile"
    
    # Base command
    cmd = [
        "snakemake",
        "--snakefile", str(snakefile),
        "--cores", "all",
        "--nolock"
    ]
    
    # Append the extra arguments (init, batch, --config, etc.)
    if extra_args:
        cmd.extend(extra_args)
        
    print(f"Executing: {' '.join(cmd)}") # Helpful for debugging
    subprocess.run(cmd, check=True)

def run():
    """run small set of targets"""

    print("coming soon!")


def main():
    parser = argparse.ArgumentParser(prog="coregtor")
    subparsers = parser.add_subparsers(dest="command")

    # The 'run' command for standard/simple use
    run_parser = subparsers.add_parser("run", help="Predict co-regulators for genes")
    run_parser.add_argument("--targets", required=True, help="Comma-separated list of genes")
    run_parser.add_argument("--source", choices=["all", "tf"], default="all")
    run_parser.add_argument("--data", required=True, help="Path to expression data")
    run_parser.add_argument("--data-type", required=True, help="e.g., gtex, tcga")
    run_parser.add_argument("--format", default="csv", choices=["csv", "tsv"])
    run_parser.add_argument("--force", action="store_true", help="Skip the bulk warning")

    # The 'bulk' command explicitly for Snakemake
    bulk_parser = subparsers.add_parser("bulk", help="Run optimized Snakemake pipeline")
    
    # Use REMAINDER to catch everything after 'bulk' (init, batch, etc.)
    bulk_parser.add_argument("snakemake_args", nargs=argparse.REMAINDER, 
                             help="Subcommands and arguments to pass to Snakemake")


    test_parser = subparsers.add_parser("hi", help="Just a test")


    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Process gene list
    

    if args.command == "run":
        # Check if we should suggest the bulk pipeline
        gene_list = [g.strip() for g in args.targets.split(",")]
        num_genes = len(gene_list)
        if num_genes > 50 and not args.force:
            print(f"Warning: {num_genes} genes requested.", file=sys.stderr)
            print("Processing more than 50 genes is faster via the bulk pipeline.", file=sys.stderr)
            print("Use 'coregtor bulk' or add '--force' to continue here.", file=sys.stderr)
            sys.exit(1)
        
        run(args, gene_list)

    elif args.command == "bulk":
        run_bulk(args.snakemake_args)

    elif args.command == "hi":
        print("hi")
        print("bye")



if __name__ == "__main__":
    main()