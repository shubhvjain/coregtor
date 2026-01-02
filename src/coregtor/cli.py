#!/usr/bin/env python3
"""CoRegTor CLI."""

import argparse
import json
from pathlib import Path
import sys
import os
from coregtor.pipeline import Pipeline
from coregtor.utils.exp import read_GE_data
import pandas as pd
def get_a_path(pth):
    return Path(os.path.expanduser(os.path.expandvars(pth))) 


def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    parser = argparse.ArgumentParser(description="CoRegTor Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    gen_parser = subparsers.add_parser("generate_config")
    gen_parser.add_argument("--output", "-o", default="config.json")
    
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", "-c", required=True)
    run_parser.add_argument("--title", "-t", required=True)
    
    args = parser.parse_args()
    
    if args.command == "generate_config":
        # Pipeline generates dict  CLI saves file
        default_config = Pipeline._generate_default_config_dict()
        Path(args.output).write_text(json.dumps(default_config, indent=2))
        print(f"Config generated: {args.output}")
        
    elif args.command == "run":
        config = json.loads(Path(args.config).read_text())
        
        expression_data = read_GE_data(get_a_path(config["input"]["expression"]))
        tflist = pd.read_csv(get_a_path(config["input"]["tflist"]),names=["gene_name"], header=None)
        # exp_title = "test1"
        pipeline = Pipeline(expression_data,tflist,config,exp_title=args.title)
        pipeline.run()
        # details = pipeline.run_details()

        # output_path = get_a_path(config["output_dir"]) / f"{pipeline.title}.json"
        # with open(output_path, "w") as f:
        #     json.dump(details, f, indent=1)


if __name__ == "__main__":
    main()
