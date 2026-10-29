#!/usr/bin/env python3
"""CoRegTor CLI.

    coregtor bulk        --input exp.json --batch 100 --output OUT --temp TMP
    coregtor status      --input exp.json --output OUT --temp TMP [--name res1]
    coregtor result      --input exp.json --output OUT --temp TMP --name res1
    coregtor run         --input exp.json --output OUT --result res1
    coregtor consolidate --input exp.json --output OUT --temp TMP
"""
import argparse
import sys

from coregtor.workflow import util  


def add_input(p):
    p.add_argument("--input", required=True, help="Experiment input file (JSON/YAML)")


def add_paths(p, temp=True, output_required=False):
    p.add_argument("--output", required=output_required, default=util.DEFAULT_OUTPUT,
                   help="Base output folder, results go to <output>/<exp id> (default ./output)")
    if temp:
        p.add_argument("--temp", default=util.DEFAULT_TEMP,
                       help="Base temp folder, temp files go to <temp>/<exp id> (default ./temp)")


def build_parser():
    parser = argparse.ArgumentParser(prog="coregtor")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("bulk", help="Generate temp files for a batch of pending targets")
    add_input(p)
    add_paths(p)
    p.add_argument("--batch", type=int, default=100, help="Targets to process in this call (default 100)")
    p.add_argument("--batch-id", help="Custom batch id (default: timestamp)")
    p.add_argument("--rerun", action="store_true", help="Regenerate temp files even if they exist")

    p = sub.add_parser("status", help="Show progress of the bulk run")
    add_input(p)
    add_paths(p)
    p.add_argument("--name", help="Also show whether this result has been generated")

    p = sub.add_parser("result", help="Generate a result file from the stored temp files")
    add_input(p)
    add_paths(p)
    p.add_argument("--name", required=True, help="Result id from the input file")
    p.add_argument("--rerun", action="store_true", help="Overwrite an existing result file")
    p.add_argument("--jobs", type=int, help="Parallel jobs for result generation")

    p = sub.add_parser("run", help="Generate a result directly, no temp files kept")
    add_input(p)
    add_paths(p, temp=False, output_required=False)
    p.add_argument("--result", required=False, default="default",help="Result id from the input file")
    p.add_argument("--rerun", action="store_true", help="Overwrite an existing result file")
    p.add_argument("--jobs", type=int, help="Parallel jobs for result generation")

    p = sub.add_parser("consolidate", help="Merge leftover batch databases into status.db")
    add_input(p)
    add_paths(p)

    return parser


def main(argv=None):
    """Parse the command line and run the chosen command. Returns the exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
 
    if not args.command:  # user typed just `coregtor`
        parser.print_help()
        return 0
 
    try:
        exp = util.load_input(args.input)
        # print(exp)
        temp = getattr(args, "temp", None)  # `run` has no --temp
 
        if args.command == "bulk":
            util.bulk(exp, args.output, temp, batch_size=args.batch,
                      batch_id=args.batch_id, rerun=args.rerun)
        elif args.command == "status":
            util.status(exp, args.output, temp, name=args.name)
        elif args.command == "result":
            util.result(exp, args.output, temp, name=args.name,
                        rerun=args.rerun, jobs=args.jobs)
        elif args.command == "run":
            util.run(exp, args.output, name=args.result, rerun=args.rerun)
        elif args.command == "consolidate":
            util.consolidate(exp, args.output, temp)
 
    except util.CoregtorCLIError as e:
        # Problems the user can fix: short message, no traceback
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130  # conventional exit code for Ctrl-C
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())